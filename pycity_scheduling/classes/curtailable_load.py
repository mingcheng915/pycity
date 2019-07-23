import warnings
import gurobipy as gurobi
import pycity_base.classes.demand.ElectricalDemand as ed
import numpy as np

from .electrical_entity import ElectricalEntity


class CurtailableLoad(ElectricalEntity, ed.ElectricalDemand):
    """
    Extension of pycity class ElectricalDemand for scheduling purposes.
    """

    def __init__(self, environment, P_El_Nom, max_curtailment,
                 max_off=None, min_on=None):
        """Initialize a curtailable load.

        Parameters
        ----------
        environment : Environment
            Common Environment instance.
        P_El_Nom : float
            Nominal electric power in [kW].
        max_curtailment : float
            Maximal Curtailment of the load
        max_off : int, optional
            Maximum amount of timesteps the curtailable load can stay under
            nominal load
        min_on : int, optional
            Minimum amount of timesteps the curtailable load has to stay on
            when switching to the nominal operation level
        """
        shape = environment.timer.timestepsTotal
        super(CurtailableLoad, self).__init__(environment, 0, np.zeros(shape))
        self._long_ID = "CUL_" + self._ID_string

        self.P_El_Nom = P_El_Nom
        self.max_curt = max_curtailment
        if max_off is not None or min_on is not None:
            assert max_off is not None
            assert min_on is not None
            assert min_on >= 1
            assert max_off >= 0
            assert self.simu_horizon > min_on + max_off
        self.max_off = max_off
        self.min_on = min_on
        self.P_El_Curt = self.P_El_Nom * self.max_curt
        self.P_State_vars = []
        self.P_State_schedule = np.empty(self.simu_horizon, bool)
        self.constr_previous_state = []
        self.constr_previous = []

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
            - `integer`  : Uses integer variables for max_off and min_on constraints if necessary
        """
        super(CurtailableLoad, self).populate_model(model, mode)

        if mode == "convex" or mode == "integer":
            for t in self.op_time_vec:
                self.P_El_vars[t].lb = self.P_El_Curt
                self.P_El_vars[t].ub = self.P_El_Nom
            if self.max_off is None:
                pass
            elif self.max_off == 0:
                # max off = zero results in constant nominal load
                for t in self.op_time_vec:
                    self.P_El_vars[t].lb = self.P_El_Nom
            elif mode == "integer":
                # generate integer constraints for max_off min_on values
                for t in self.op_time_vec:
                    self.P_State_vars.append(model.addVar(
                        vtype=gurobi.GRB.BINARY,
                        name="%s_Mode_at_t=%i"
                             % (self._long_ID, t + 1)
                    ))
                model.update()
                for t in self.op_time_vec:
                    # connect to P_El_vars
                    model.addConstr(
                        self.P_State_vars[t] * self.P_El_Nom <= self.P_El_vars[t]
                    )
                max_overlap = max(self.max_off, self.min_on - 1)
                for t in range(1, max_overlap + 1):
                    self.constr_previous_state.append(model.addConstr(
                        gurobi.quicksum(self.P_State_vars[:t]) >= -gurobi.GRB.INFINITY
                    ))

                for t in self.op_time_vec:
                    next_states = self.P_State_vars[t:t + self.max_off + 1]
                    assert 1 <= len(next_states) <= self.max_off + 1
                    if len(next_states) == self.max_off + 1:
                        model.addConstr(
                            gurobi.quicksum(next_states) >= 1
                        )
                if self.min_on > 1:
                    for t in self.op_time_vec[:-2]:
                        next_states = self.P_State_vars[t + 2: t + self.min_on + 1]
                        assert 1 <= len(next_states) <= self.min_on - 1
                        model.addConstr(
                            (self.P_State_vars[t + 1] - self.P_State_vars[t]) * len(next_states) <=
                            gurobi.quicksum(next_states)
                        )
            else:
                # generate relaxed constraints with max_off min_on values
                width = self.min_on + self.max_off
                for t in self.op_time_vec[:-width]:
                    next_vars = self.P_El_vars[t:t + width]
                    assert len(next_vars) == width
                    model.addConstr(
                        gurobi.quicksum(next_vars) >=
                        self.P_El_Nom * self.min_on + self.P_El_Curt * self.max_off
                    )

                for t in range(1, self.max_off + self.min_on):
                    self.constr_previous.append(model.addConstr(
                        gurobi.quicksum(self.P_El_vars[:t]) >= -gurobi.GRB.INFINITY
                    ))
        else:
            raise ValueError(
                "Mode %s is not implemented by CHP." % str(mode)
            )

    def upate_model(self, model, mode="convex"):
        super(CurtailableLoad, self).update_model(model, mode)
        timestep = self.timer.currentTimestep
        if timestep != 0:
            if len(self.constr_previous_state) > 0:
                for constr in self.constr_previous_state:
                    constr.RHS = -gurobi.GRB.INFINITY
                if self.P_State_schedule[timestep - 1]:
                    for t in range(max(0, timestep - self.min_on), timestep - 2, 1):
                        if not self.P_State_schedule[t] and self.P_State_schedule[t + 1]:
                            remaining_ons = self.min_on - (timestep - (t + 1))
                            self.constr_previous_state[remaining_ons - 1].RHS = remaining_ons
                            break
                else:
                    off_ts = 1
                    while self.P_State_schedule[
                        timestep - off_ts - 1] and off_ts <= self.max_off and off_ts <= timestep:
                        off_ts += 1
                    overlap = self.max_off - off_ts
                    self.constr_previous_state[overlap - 1].RHS = 1

            elif len(self.constr_previous) > 0:
                width = self.min_on + self.max_off
                for t in range(max(0, timestep - width + 1), timestep, step=1):
                    self.constr_previous[width - (timestep - t) - 1].RHS = \
                        self.P_El_Nom * self.min_on + self.P_El_Curt * self.max_off - \
                        sum(self.P_El_Schedule[t:timestep])

    def update_schedule(self):
        super(CurtailableLoad, self).update_schedule()
        if len(self.constr_previous_state) > 0:
            timestep = self.timer.currentTimestep
            self.P_State_schedule[timestep:timestep + self.op_horizon] \
                = [np.isclose(var.X, self.P_El_Nom) for var in self.P_El_vars]

    def get_objective(self, coeff=None):
        """Objective function for entity level scheduling.

        Return the objective function of the curtailable load wheighted with
        coeff. Quadratic term minimizing the deviation from the loadcurve.

        Parameters
        ----------
        coeff : float, optional
            Coefficient for the objective function.
            Uses day-ahead prices by default
        Returns
        -------
        gurobi.QuadExpr :
            Objective function.
        """
        obj = gurobi.QuadExpr()
        if len(self.P_State_vars) > 0:
            timestep = self.environment.timer.currentTimestep
            if coeff is None:
                da_prices = self.environment.prices.da_prices
            elif type(coeff) is np.ndarray:
                da_prices = coeff.tolist()
            elif type(coeff) == list:
                da_prices = coeff
            elif np.isscalar(coeff):
                da_prices = [coeff] * (self.simu_horizon + self.min_on)
            else:
                raise ValueError(coeff)
            assert len(da_prices) >= self.simu_horizon + self.min_on
            next_coeffs = [-x for x in da_prices[timestep+self.op_horizon:timestep+self.op_horizon+self.min_on]]
            assert len(next_coeffs) == self.min_on
            obj.addTerms(
                next_coeffs,
                self.P_State_vars[-self.min_on:]
            )
        return obj

    def update_deviation_model(self, model, timestep, mode=""):
        """Update deviation model for the current timestep."""
        if mode == 'full':
            self.P_El_Act_var.lb = self.P_El_Curt
            self.P_El_Act_var.ub = self.P_El_Nom
        else:
            self.P_El_Act_var.lb = self.P_El_Schedule[timestep]
            self.P_El_Act_var.ub = self.P_El_Schedule[timestep]
