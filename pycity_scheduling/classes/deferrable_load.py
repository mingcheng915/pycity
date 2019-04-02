import numpy as np
import gurobipy as gurobi
import pycity_base.classes.demand.ElectricalDemand as ed
import warnings

from ..exception import PyCitySchedulingInitError
from ..util import compute_blocks
from .electrical_entity import ElectricalEntity


class DeferrableLoad(ElectricalEntity, ed.ElectricalDemand):
    """
    Extension of pycity class ElectricalDemand for scheduling purposes.
    """

    def __init__(self, environment, P_El_Nom, E_Min_Consumption, time):
        """Initialize DeferrableLoad.

        Parameters
        ----------
        environment : Environment
            Common Environment instance.
        P_El_Nom : float
            Nominal elctric power in [kW].
        E_Min_Consumption : float
             Minimal power to be consumed over the time in [kWh].
        time : array of binaries
            Indicator when deferrable load can be turned on.
            `time[t] == 0`: device is off in t
            `time[t] == 1`: device *can* be turned on in t
            `time` must contain at least one `0` otherwise the model will
            become infeasible.
        """
        shape = environment.timer.timestepsTotal
        super(DeferrableLoad, self).__init__(environment.timer, environment, 0,
                                             np.zeros(shape))

        self._long_ID = "DL_" + self._ID_string

        if len(time) != 86400 / self.timer.timeDiscretization:
            raise PyCitySchedulingInitError(
                "The `time` argument must hold as many values as one day "
                "has timesteps."
            )
        self.P_El_Nom = P_El_Nom
        self.E_Min_Consumption = E_Min_Consumption
        self.E_Min_Slots = E_Min_Consumption / (P_El_Nom * environment.timer.time_slot)
        if self.E_Min_Slots != np.ceil(self.E_Min_Slots):
            warnings.warn(
                "Minimal Power Consumption can't be reached with Nominal Power for {}. Rounding Power Consumption up."
                    .format(self._long_ID)
            )
            np.int(np.ceil(self.E_Min_Slots))
        else:
            self.E_Min_Slots = np.int(self.E_Min_Slots)
        self.time = time
        self.P_El_bvars = []
        self.P_El_Sum_constrs = []

    def populate_model(self, model, mode=""):
        """Add variables and constraints to Gurobi model

        Call parent's `populate_model` method and set the upper bounds to the
        nominal power or zero depending on `self.time`. Also set a constraint
        for the minimum load. If mode == 'binary' add binary variables to model
        load as one block that can be shifted in time.

        Parameters
        ----------
        model : gurobi.Model
        """
        super(DeferrableLoad, self).populate_model(model)

        # device is on or off
        self.P_El_bvars = []

        # Add variables:
        for t in self.op_time_vec:
            if self.time[t] == 1:
                self.P_El_bvars.append(
                    model.addVar(
                        vtype=gurobi.GRB.BINARY,
                        name="%s_binary_at_t=%i"
                             % (self._long_ID, t+1)
                    )
                )
            else:
                self.P_El_bvars.append(0)
        model.update()

        # Set additional constraints:
        for t in self.op_time_vec:
            model.addConstr(
                self.P_El_vars[t] == self.P_El_bvars[t] * self.P_El_Nom
            )

        model.addConstr(
            self.P_El_bvars[0] + gurobi.quicksum(
                (self.P_El_bvars[t] - self.P_El_bvars[t+1])
                * (self.P_El_bvars[t] - self.P_El_bvars[t+1])
                for t in range(self.op_horizon - 1))
            <= 2
        )

        self.P_El_Sum_constrs.append(
            model.addConstr(
                gurobi.quicksum(self.P_El_bvars)
                == self.E_Min_Slots
            )
        )

    def update_model(self, model, mode=""):
        super(DeferrableLoad, self).update_model(model)

        timestep = self.timer.currentTimestep
        # raises GurobiError if constraints are from a prior scheduling
        # optimization
        try:
            for constr in self.P_El_Sum_constrs:
                model.remove(constr)
        except gurobi.GurobiError:
            pass
        del self.P_El_Sum_constrs[:]
        # consider already completed consumption
        completed_load = 0
        for p in self.P_El_Schedule[:timestep][::-1]:
            assert (p == self.P_El_Nom or p == 0)
            completed_load += int(p/self.P_El_Nom)

        #if load is already running, set initial state to ON
        if 0 < completed_load < self.E_Min_Slots:
            self.P_El_bvars[0].lb = 1
        else:
            if type(self.P_El_bvars[0]) != int:
                self.P_El_bvars[0].lb = 0
        self.P_El_Sum_constrs.append(
            model.addConstr(
                gurobi.quicksum(self.P_El_bvars)
                == self.E_Min_Slots - completed_load
            )
        )

    def get_objective(self, coeff=1):
        """Objective function for entity level scheduling.

        Return the objective function of the deferrable load wheighted with
        coeff. Quadratic term minimizing the deviation from the optiaml
        loadcurve.

        Parameters
        ----------
        coeff : float, optional
            Coefficient for the objective function.

        Returns
        -------
        gurobi.QuadExpr :
            Objective function.
        """
        return gurobi.QuadExpr()
