"""
The pycity_scheduling framework


Copyright (C) 2022,
Institute for Automation of Complex Power Systems (ACS),
E.ON Energy Research Center (E.ON ERC),
RWTH Aachen University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import numpy as np
import pyomo.environ as pyomo
import pycity_base.classes.demand.electrical_demand as ed

from pycity_scheduling.classes.electrical_entity import ElectricalEntity


class CurtailableLoad(ElectricalEntity, ed.ElectricalDemand):
    """
    Extension of pyCity_base class ElectricalDemand for scheduling purposes.

    Parameters
    ----------
    environment : Environment
        Common Environment instance.
    p_el_nom : float
        Nominal electric power in [kW].
    max_curtailment : float
        Maximal Curtailment of the load
    max_low : int, optional
        Maximum number of timesteps the curtailable load can stay under
        nominal load
    min_full : int, optional
        Minimum number of timesteps the curtailable load has to stay at
        nominal operation level when switching to the nominal operation
        level

    Notes
    -----
    - CLs offer sets of constraints for operation. In the `convex` mode the following
      constraints and bounds are generated by the CL:

    .. math::
        p_{el\\_nom} * max\\_curtailment \\geq p_{el} \\geq 0 \\\\
        \\sum_{j=i}^{i+max\\_low+min\\_full} p_{el\\_j} \\geq p_{el\\_nom} *
        (min\\_full + max\\_low * max\\_curtailment)

    - The last constraint is replaced in integer mode with the following constraints:

    .. math::
        p_{el} \\geq p_{state} * p_{el\\_nom} \\\\
        \\sum_{j=i}^{i+max\\_low} p_{state\\_j} \\geq 1 \\\\
        \\sum_{j=i}^{i+min\\_full-1} p_{state\\_j} \\geq
        (p_{state\\_i} - p_{state\\_i-1}) * min\\_full

    - These constraints take also the previous values before the current optimization
      horizon into account using the current schedule. Values before :math:`t=0` are
      assumed to be perfect.
    """

    def __init__(self, environment, p_el_nom, max_curtailment, max_low=None, min_full=None):
        shape = environment.timer.timesteps_total
        super().__init__(environment, 0, np.zeros(shape))
        self._long_id = "CUL_" + self._id_string

        self.p_el_nom = p_el_nom
        self.max_curt = max_curtailment
        if max_low is not None or min_full is not None:
            assert max_low is not None
            assert min_full is not None
            assert min_full >= 1
            assert max_low >= 0
        self.max_low = max_low
        self.min_full = min_full
        self.p_el_curt = self.p_el_nom * self.max_curt
        self.new_var("p_state", dtype=np.bool, func=lambda model:
                     self.schedule["p_el"][self.op_slice] > 0.99*p_el_nom)

    def populate_model(self, model, mode="convex"):
        """
        Add device block to pyomo ConcreteModel

        Call parent's `populate_model` method and set variables upper bounds to
        the loadcurve and lower bounds to `self.p_el_Min`.

        Parameters
        ----------
        model : pyomo.ConcreteModel
        mode : str, optional
            Specifies which set of constraints to use.

            - `convex`  : Use linear constraints
            - `integer`  : Uses integer variables for max_low and min_full constraints if necessary
        """
        super(CurtailableLoad, self).populate_model(model, mode)
        m = self.model

        if mode in ["convex", "integer"]:
            m.p_el_vars.setlb(self.p_el_curt)
            m.p_el_vars.setub(self.p_el_nom)

            if self.max_low is None:
                # if max_low is not set the entity can choose p_state freely.
                # as a result no constraints are required
                pass
            elif self.max_low == 0:
                # if max_low is zero the p_state_vars would have to always be one
                # this results in operation at always 100%.
                # the following bound is enough to represent this behaviour
                m.p_el_vars.setlb(self.p_el_nom)
            elif mode == "integer":
                # generate integer constraints for max_low min_full values

                # create binary variables representing the state if the device is operating at full level
                m.p_state_vars = pyomo.Var(m.t, domain=pyomo.Binary)

                # coupling the state variable to the electrical variable
                # since operation infinitely close to 100% can be chosen by the entity to create a state
                # of zero, coupling in one direction is sufficient.

                def p_state_rule(model, t):
                    return model.p_state_vars[t] * self.p_el_nom <= model.p_el_vars[t]
                m.p_state_integer_constr = pyomo.Constraint(m.t, rule=p_state_rule)

                # creat constraints which can be used by update_model to take previous states into account.
                # update_schedule only needs to modify RHS which should be faster than deleting and creating
                # new constraints

                max_overlap = max(self.max_low, self.min_full)

                # add constraints forcing the entity to operate at least once at 100% between every range
                # of max_low + 1 in the op_horizon
                m.previous_n_states = pyomo.Param(pyomo.RangeSet(1, max_overlap), mutable=True, initialize=True)

                def p_full_rule(model, t):
                    e = 0
                    entries = 0
                    for t_backwards in range(t-self.max_low, t + 1, 1):
                        if t_backwards >= 0:
                            e += model.p_state_vars[t_backwards]
                        else:
                            e += model.previous_n_states[-t_backwards]
                        entries += 1

                    assert entries == self.max_low + 1
                    return e >= 1
                m.P_full_integer_constr = pyomo.Constraint(m.t, rule=p_full_rule)

                # add constraints to operate at a minimum of min_full timesteps at 100% when switching
                # from the state 0 to the state 1
                if self.min_full > 1:
                    def p_on_rule(model, t):
                        e = 0
                        entries = 0
                        for t_backwards in range(t - self.min_full + 1, t + 1, 1):
                            if t_backwards >= self.op_horizon:
                                e += 1
                            elif t_backwards >= 0:
                                e += model.p_state_vars[t_backwards]
                            else:
                                e += model.previous_n_states[-t_backwards]
                            entries += 1
                        assert entries == self.min_full
                        t_flank_start = t - self.min_full
                        t_flank_end = t - self.min_full + 1
                        if t_flank_start >= 0:
                            flank_start = model.p_state_vars[t_flank_start]
                        else:
                            flank_start = model.previous_n_states[-t_flank_start]
                        if t_flank_end >= 0:
                            flank_end = model.p_state_vars[t_flank_end]
                        else:
                            flank_end = model.previous_n_states[-t_flank_end]
                        return (flank_end - flank_start) * self.min_full <= e

                    m.P_on_integer_constr = pyomo.Constraint(pyomo.RangeSet(0, self.op_horizon + self.min_full - 2),
                                                             rule=p_on_rule)

            else:
                # generate relaxed constraints with max_low min_full values
                width = self.min_full + self.max_low

                m.previous_n_consumptions = pyomo.Param(pyomo.RangeSet(1, width-1),
                                                        mutable=True,
                                                        initialize=self.p_el_nom)

                def p_average_rule(model, t):
                    e = 0
                    entries = 0
                    for t_backwards in range(t - width + 1, t + 1, 1):
                        if t_backwards >= 0:
                            e += model.p_el_vars[t_backwards]
                        else:
                            e += model.previous_n_consumptions[-t_backwards]
                        entries += 1
                    assert entries == width
                    return e >= self.p_el_nom * self.min_full + self.p_el_curt * self.max_low

                m.P_average_constr = pyomo.Constraint(m.t, rule=p_average_rule)

        else:
            raise ValueError(
                "Mode %s is not implemented by class CurtailableLoad." % str(mode)
            )
        return

    def update_model(self, mode="convex"):
        super(CurtailableLoad, self).update_model(mode)
        m = self.model
        timestep = self.timer.current_timestep
        if hasattr(m, "previous_n_states"):

            for key in m.previous_n_states.sparse_iterkeys():
                if timestep - key >= 0:
                    previous_value = self.p_state_schedule[timestep - key] * 1.0
                else:
                    # for timesteps below zero a perfect state is assumed.
                    previous_value = True
                m.previous_n_states[key] = previous_value
        if hasattr(m, "previous_n_consumptions"):
            for key in m.previous_n_consumptions.sparse_iterkeys():
                if timestep - key >= 0:
                    previous_value = self.p_el_schedule[timestep - key]
                else:
                    # for timesteps below zero a perfect state is assumed.
                    previous_value = self.p_el_nom

                m.previous_n_consumptions[key] = previous_value
        return
