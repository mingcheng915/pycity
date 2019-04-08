import gurobipy as gurobi
import pycity_base.classes.demand.ElectricalDemand as ed

from .electrical_entity import ElectricalEntity


class CurtailableLoad(ElectricalEntity):
    """
    Extension of pycity class ElectricalDemand for scheduling purposes.
    """

    def __init__(self, environment, P_El_Nom: float, max_low_time: int = None, min_on_time: int = 1,
                 min_operation_level: float=0):
        """Initialize a curtailable load.

        Parameters
        ----------
        environment : Environment
            Common Environment instance.
        P_El_Nom : float
            Nominal elctric power in [kW].
        max_low_time : int
            Amount of time slots the Curtailable Load is allowed to stay under nominal load.
        min_on_time : int
            Minimum Amount of time slots the Curtailable Load has to operate on nominal load after operating under
            nominal load.
        min_operation_level : float
            Minimum load the Device is allowed to operate at
        """
        super(CurtailableLoad, self).__init__(
            environment.timer
        )
        self._long_ID = "CUL_" + self._ID_string

        self.P_El_Nom = P_El_Nom

        self.max_low = max_low_time
        self.min_on = min_on_time
        assert 0 <= min_operation_level <= 1
        self.min_level = min_operation_level

    def populate_model(self, model, mode=""):
        """Add variables to Gurobi model

        Call parent's `populate_model` method and set variables upper bounds to
        the loadcurve and lower bounds to s`elf.P_El_Min`.

        Parameters
        ----------
        model : gurobi.Model
        mode : str, optional
        """
        super(CurtailableLoad, self).populate_model(model, mode)
        time_shift = self.timer.currentTimestep
        self.P_El_bvars=[]
        constrs = []
        for t in self.op_time_vec:
            self.P_El_vars[t].lb = self.min_level * self.P_El_Nom
            self.P_El_vars[t].ub = self.P_El_Nom
            bvar = model.addVar(
                        vtype=gurobi.GRB.BINARY,
                        name="%s_binary_at_t=%i"
                             % (self._long_ID, t+1)
                    )
            #connect bvar to P_El_vars
            constrs.append(model.addConstr(
                bvar*self.P_El_Nom <= self.P_El_vars[t]
            ))
            self.P_El_bvars.append(bvar)
        model.update()
        #add constr that requires returning to nominal power consumpiton
        if self.max_low is None:
            pass
        elif self.max_low == 0:
            for t in self.op_time_vec:
                self.P_El_vars[t].lb = self.P_El_Nom
            return
        elif self.max_low > 0:
            for t in self.op_time_vec:
                if t <= max(self.op_time_vec) - self.max_low:
                    assert len(self.P_El_bvars[t:t+self.max_low+1]) == self.max_low+1
                    constrs.append(model.addConstr(
                        1 <= gurobi.quicksum(self.P_El_bvars[t:t+self.max_low+1])
                    ))
        else:
            raise ValueError

        #add constraint to keep at minimum x steps on
        if self.min_on <= 1:
            pass
        elif self.min_on > 1:
            for t in self.op_time_vec[:-2]:
                to_be_on = self.P_El_bvars[t+2:t+self.min_on+1]
                if self.min_on-1 <= len(self.P_El_bvars[t+2:]):
                    assert len(to_be_on) == self.min_on-1
                else:
                    assert self.P_El_bvars[t + 2:] == to_be_on
                constrs.append(model.addConstr(
                    (self.P_El_bvars[t + 1]-self.P_El_bvars[t])*self.P_El_bvars[t + 1]*len(to_be_on) <=
                    gurobi.quicksum(to_be_on)
                ))
        else:
            raise ValueError

        #add constraint to not allow switching of electrical consumption without returning to nominal consumption
        for t in self.op_time_vec[:-1]:
            constrs.append(model.addConstr(
                0 == (self.P_El_bvars[t]+self.P_El_bvars[t+1]-1)*(self.P_El_vars[t]-self.P_El_vars[t+1])
            ))

    def get_objective(self, coeff=1):
        """Objective function for entity level scheduling.

        Return the objective function of the curtailable load wheighted with
        coeff. Quadratic term minimizing the deviation from the loadcurve.

        Parameters
        ----------
        coeff : float, optional
            Coefficient for the objective function.

        Returns
        -------
        gurobi.QuadExpr :
            Objective function.
        """
        return 0
