import numpy as np
import pyomo.environ as pyomo
import pycity_base.classes.demand.ElectricalDemand as ed

from .electrical_entity import ElectricalEntity


class CurtailableLoad(ElectricalEntity, ed.ElectricalDemand):
    """
    Extension of pyCity_base class ElectricalDemand for scheduling purposes.
    """

    def __init__(self, environment, P_El_Nom, max_curtailment,
                 max_low=None, min_full=None):
        """Initialize a curtailable load.

        Parameters
        ----------
        environment : Environment
            Common Environment instance.
        P_El_Nom : float
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
        """
        shape = environment.timer.timestepsTotal
        super().__init__(environment, 0, np.zeros(shape))
        self._long_ID = "CUL_" + self._ID_string

        self.P_El_Nom = P_El_Nom
        self.max_curt = max_curtailment
        if max_low is not None or min_full is not None:
            assert max_low is not None
            assert min_full is not None
            assert min_full >= 1
            assert max_low >= 0
        self.max_low = max_low
        self.min_full = min_full
        self.P_El_Curt = self.P_El_Nom * self.max_curt
        self.new_var("P_State", dtype=np.bool, func=lambda model, t: pyomo.value(model.P_El_vars[t] > 0.99*P_El_Nom))
        self.constr_previous_state = []
        self.constr_previous = []
        self.constr_previous_start = None

    def populate_model(self, model, mode="convex"):
        """Add variables to Gurobi model

        Call parent's `populate_model` method and set variables upper bounds to
        the loadcurve and lower bounds to s`elf.P_El_Min`.

        Parameters
        ----------
        model : gurobi.Model
        mode : str, optional
            Specifies which set of constraints to use
            - `convex`  : Use linear constraints
            - `integer`  : Uses integer variables for max_low and min_full constraints if necessary
        """
        super(CurtailableLoad, self).populate_model(model, mode)
        m = self.model

        if mode in ["convex", "integer"]:
            m.P_El_vars.setlb(self.P_El_Curt)
            m.P_El_vars.setub(self.P_El_Nom)

            if self.max_low is None:
                # if max_low is not set the entity can choose P_State freely.
                # as a result no constraints are required
                pass
            elif self.max_low == 0:
                # if max_low is zero the P_State_vars would have to  always be one
                # this results in operation at always 100%.
                # the following bound is enough to represent this behaviour
                m.P_El_vars.setlb(self.P_El_Nom)
            elif mode == "integer":
                # generate integer constraints for max_low min_full values

                # create binary variables representing the state if the device is operating at full level
                m.P_State_vars = pyomo.Var(m.t, domain=pyomo.Binary)

                # coupling the state variable to the electrical variable
                # since operation infinitly close to 100% can be chosen by the entity to create a state
                # of zero, coupling in one direction is sufficient.

                def p_state_rule(model, t):
                    return model.P_State_vars[t] * self.P_El_Nom <= model.P_El_vars[t]
                m.P_state_constr = pyomo.Constraint(m.t, rule=p_state_rule)

                # creat constraints which can be used by update_model to take previous states into account.
                # update_schedule only needs to modify RHS which should be faster than deleting and creating
                # new constraints

                # REMOVED
                max_overlap = max(self.max_low, self.min_full)
                #max_overlap = min(max_overlap, self.op_horizon)  # cap overlap constraints at op_horizon
                #for t in range(1, max_overlap + 1):
                #    self.constr_previous_state.append(model.addConstr(
                #        gurobi.quicksum(self.P_State_vars[:t]) >= -gurobi.GRB.INFINITY
                #    ))

                # add constraints forcing the entity to operate at least once at 100% between every range
                # of max_low + 1 in the op_horizon
                m.previous_n_states = pyomo.Param(pyomo.RangeSet(1, max_overlap), mutable=True, initialize=True)

                def p_full_rule(model, t):
                    e = 0
                    entries = 0
                    for t_backwards in range(t-self.max_low, t + 1, 1):
                        if t_backwards >= 0:
                            e += model.P_State_vars[t_backwards]
                        else:
                            e += model.previous_n_states[-t_backwards]
                        entries += 1

                    assert entries == self.max_low + 1
                    return e >= 1
                m.P_full_constr = pyomo.Constraint(m.t, rule=p_full_rule)

                # add constraints to operate at a minimum of min_full timesteps at 100% when switching
                # from the state 0 to the state 1
                #if self.min_full > 1:
                #    next_states = self.P_State_vars[1:self.min_full]
                #    self.constr_previous_start = model.addConstr(
                #        self.P_State_vars[0] * len(next_states)
                #        - gurobi.quicksum(next_states) <=
                #        gurobi.GRB.INFINITY  # self.P_State_vars[t-1] set via update_model
                #    )
                #    for t in self.op_time_vec[:-2]:
                #        next_states = self.P_State_vars[t + 2: t + self.min_full + 1]
                #        assert 1 <= len(next_states) <= self.min_full - 1
                #        model.addConstr(
                #            (self.P_State_vars[t + 1] - self.P_State_vars[t]) * len(next_states) <=
                #            gurobi.quicksum(next_states)
                #        )
                if self.min_full > 1:
                    def p_on_rule(model, t):
                        e = 0
                        entries = 0
                        for t_backwards in range(t - self.min_full + 1, t + 1, 1):
                            if t_backwards >= self.op_horizon:
                                e += 1
                            elif t_backwards >= 0:
                                e += model.P_State_vars[t_backwards]
                            else:
                                e += model.previous_n_states[-t_backwards]
                            entries += 1
                        assert entries == self.min_full
                        t_flank_start = t - self.min_full
                        t_flank_end = t - self.min_full + 1
                        if t_flank_start >= 0:
                            flank_start = model.P_State_vars[t_flank_start]
                        else:
                            flank_start = model.previous_n_states[-t_flank_start]
                        if t_flank_end >= 0:
                            flank_end = model.P_State_vars[t_flank_end]
                        else:
                            flank_end = model.previous_n_states[-t_flank_end]
                        return (flank_end - flank_start) * self.min_full <= e

                    m.P_on_constr = pyomo.Constraint(pyomo.RangeSet(0, self.op_horizon + self.min_full - 2), rule=p_on_rule)

            else:
                # generate relaxed constraints with max_low min_full values
                width = self.min_full + self.max_low

                m.previous_n_consumptions = pyomo.Param(pyomo.RangeSet(1, width-1), mutable=True, initialize=self.P_El_Nom)
                #for t in self.op_time_vec[:-width + 1]:
                #    next_vars = self.P_El_vars[t:t + width]
                #    assert len(next_vars) == width
                #    model.addConstr(
                #        gurobi.quicksum(next_vars) >=
                #        self.P_El_Nom * self.min_full + self.P_El_Curt * self.max_low
                #    )

                def p_average_rule(model, t):
                    e = 0
                    entries = 0
                    for t_backwards in range(t - width + 1, t + 1, 1):
                        if t_backwards >= 0:
                            e += model.P_El_vars[t_backwards]
                        else:
                            e += model.previous_n_consumptions[-t_backwards]
                        entries += 1
                    assert entries == width
                    return e >= self.P_El_Nom * self.min_full + self.P_El_Curt * self.max_low

                m.P_average_constr = pyomo.Constraint(m.t, rule=p_average_rule)

                # creat constraints which can be used by update_model to take previous P_El values into
                # account. update_schedule only needs to modify RHS which should be faster than deleting
                # and creating new constraints

                # REMOVED
                #max_overlap = min(self.max_low + self.min_full - 1, self.op_horizon)
                #for overlap in range(0, max_overlap):
                #    self.constr_previous.append(model.addConstr(
                #        gurobi.quicksum(self.P_El_vars[:overlap + 1]) >= -gurobi.GRB.INFINITY
                #    ))
        else:
            raise ValueError(
                "Mode %s is not implemented by CHP." % str(mode)
            )


    def update_model(self, model, mode="convex"):
        super(CurtailableLoad, self).update_model(model, mode)
        m = self.model
        timestep = self.timer.currentTimestep
        if hasattr(m, "previous_n_states"):

            for key in m.previous_n_states.sparse_iterkeys():
                if timestep - key >= 0:
                    previous_value = self.P_State_Schedule[timestep - key] * 1.0
                else:
                    # for timesteps below zero a perfect state is assumed.
                    previous_value = True
                m.previous_n_states[key] = previous_value
        if hasattr(m, "previous_n_consumptions"):
            for key in m.previous_n_consumptions.sparse_iterkeys():
                if timestep - key >= 0:
                    previous_value = self.P_El_Schedule[timestep - key]
                else:
                    # for timesteps below zero a perfect state is assumed.
                    previous_value = self.P_El_Nom

                m.previous_n_consumptions[key] = previous_value