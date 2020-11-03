"""
The pycity_scheduling framework


@institution:
Institute for Automation of Complex Power Systems (ACS)
E.ON Energy Research Center
RWTH Aachen University

@author:
Sebastian Schwarz, M.Sc.
Sebastian Alexander Uerlich, B.Sc.
Univ.-Prof. Antonello Monti, Ph.D.
"""


import numpy as np
import pycity_base.classes.demand.space_cooling as sc

from pycity_scheduling.classes.thermal_entity_cooling import ThermalEntityCooling


class SpaceCooling(ThermalEntityCooling, sc.SpaceCooling):
    """
    Extension of pyCity_base class SpaceCooling for scheduling purposes.

    As for all uncontrollable loads, the `p_th_schedule` contains the forecast
    of the load.

    Parameters
    ----------
    environment : Environment object
        Common to all other objects. Includes time and weather instances
    method : integer, optional
        - `0` : Provide load curve directly
        - `1` : Use thermal standard load profile (not implemented yet!)
    loadcurve : Array-like, optional
        Load curve for all investigated time steps
        Requires `method=0`
    living_area : Float, optional
        Living area of the apartment in m^2
        Requires `method=1`
    specific_demand : Float, optional
        Specific thermal demand of the building in kWh/(m^2 a)
        Requires `method=1`
    profile_type : str, optional
        Thermal SLP profile name
        Requires `method=1`
        - `HEF` : Single family household
        - `HMF` : Multi family household
        - `GBA` : Bakeries
        - `GBD` : Other services
        - `GBH` : Accomodations
        - `GGA` : Restaurants
        - `GGB` : Gardening
        - `GHA` : Retailers
        - `GHD` : Summed load profile business, trade and services
        - `GKO` : Banks, insurances, public institutions
        - `GMF` : Household similar businesses
        - `GMK` : Automotive
        - `GPD` : Paper and printing
        - `GWA` : Laundries

    Notes
    -----
    The following constraint is added for removing the bounds from the TEC:

    .. math::
        p_{th\\_cool} = load\\_curve
    """

    def __init__(self, environment, method=0, loadcurve=1, living_area=0, specific_demand=0, profile_type='HEF'):

        super().__init__(environment, method, loadcurve*1000, living_area, specific_demand, profile_type)
        self._long_ID = "SC_" + self._ID_string

        ts = self.timer.time_in_year(from_init=True)
        p = self.loadcurve[ts:ts+self.simu_horizon] / 1000
        self.p_th_cool_schedule = p

    def update_model(self, mode=""):
        """Add device block to pyomo ConcreteModel.

        Set variable bounds to equal the given demand, as pure space cooling does
        not provide any flexibility.

        Parameters
        ----------
        mode : str, optional
        """
        m = self.model
        timestep = self.timestep

        for t in self.op_time_vec:
            m.p_th_cool_vars[t].setlb(self.p_th_cool_schedule[timestep + t])
            m.p_th_cool_vars[t].setub(self.p_th_cool_schedule[timestep + t])
        return

    def new_schedule(self, schedule):
        super().new_schedule(schedule)
        self.copy_schedule(schedule, "default", "p_th_cool")
        return

    def update_schedule(self, mode=""):
        pass

    def reset(self, name=None):
        pass