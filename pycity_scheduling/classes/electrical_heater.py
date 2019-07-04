import gurobipy as gurobi
import pycity_base.classes.supply.ElectricalHeater as eh

from .thermal_entity import ThermalEntity
from .electrical_entity import ElectricalEntity


class ElectricalHeater(ThermalEntity, ElectricalEntity, eh.ElectricalHeater):
    """
    Extension of pycity class ElectricalHeater for scheduling purposes.
    """

    def __init__(self, environment, P_Th_nom, eta=1, lower_activation_limit=0):
        """Initialize ElectricalHeater.

        Parameters
        ----------
        environment : pycity_scheduling.classes.Environment
            Common to all other objects. Includes time and weather instances.
        P_Th_nom : float
            Nominal thermal power output in [kW].
        eta : float, optional
            Efficiency of the electrical heater.
        lower_activation_limit : float, optional
            Must be in [0, 1]. Lower activation limit of the electrical heater
            as a percentage of the rated power. When the electrical heater is
            running its power nust be zero or between the lower activation
            limit and its rated power.
            `lower_activation_limit = 0`: Linear behavior
            `lower_activation_limit = 1`: Two-point controlled
        """
        # Flow temperature of 55 C
        super(ElectricalHeater, self).__init__(environment, P_Th_nom*1000, eta,
                                               55, lower_activation_limit)
        self._long_ID = "EH_" + self._ID_string
        self.P_Th_Nom = P_Th_nom

    def populate_model(self, model, mode=""):
        """Add variables to Gurobi model.

        Call parent's `populate_model` method and set thermal variables upper
        bounds to `self.P_Th_Nom`. Also add constraint to bind electrical
        demand to thermal output.

        Parameters
        ----------
        model : gurobi.Model
        mode : str, optional
        """
        ThermalEntity.populate_model(self, model, mode)
        ElectricalEntity.populate_model(self, model, mode)

        for var in self.P_Th_vars:
            var.lb = -self.P_Th_Nom
            var.ub = 0

        for t in self.op_time_vec:
            model.addConstr(
                - self.P_Th_vars[t] == self.eta * self.P_El_vars[t]
            )

    def get_objective(self, coeff=1):
        """Objective function for entity level scheduling.

        Return the objective function of the electrical heater wheighted with
        coeff. Sum of self.P_El_vars.

        Parameters
        ----------
        coeff : float, optional
            Coefficient for the objective function.

        Returns
        -------
        gurobi.LinExpr :
            Objective function.
        """
        obj = gurobi.LinExpr()
        obj.addTerms(
            [coeff] * self.op_horizon,
            self.P_El_vars
        )
        return obj

    def update_schedule(self):
        """Update the schedule with the scheduling model solution."""
        ThermalEntity.update_schedule(self)
        ElectricalEntity.update_schedule(self)

    def populate_deviation_model(self, model, mode=""):
        """Add variables for this entity to the deviation model.

        Adds variables, sets the correct bounds to the thermal variable and
        adds a coupling constraint.
        """
        ThermalEntity.populate_deviation_model(self, model, mode)
        ElectricalEntity.populate_deviation_model(self, model, mode)

        self.P_Th_Act_var.lb = -self.P_Th_Nom
        self.P_Th_Act_var.ub = 0
        model.addConstr(
            - self.P_Th_Act_var == self.eta * self.P_El_Act_var
        )

    def update_actual_schedule(self, timestep):
        """Update the actual schedule with the deviation model solution."""
        ThermalEntity.update_actual_schedule(self, timestep)
        ElectricalEntity.update_actual_schedule(self, timestep)

    def save_ref_schedule(self):
        """Save the schedule of the current reference scheduling."""
        ThermalEntity.save_ref_schedule(self)
        ElectricalEntity.save_ref_schedule(self)

    def reset(self, schedule=True, actual=True, reference=False):
        """Reset entity for new simulation.

        Parameters
        ----------
        schedule : bool, optional
            Specify if to reset schedule.
        actual : bool, optional
            Specify if to reset actual schedule.
        reference : bool, optional
            Specify if to reset reference schedule.
        """
        ThermalEntity.reset(self, schedule, actual, reference)
        ElectricalEntity.reset(self, schedule, actual, reference)
