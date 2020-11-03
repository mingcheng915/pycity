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

from pycity_scheduling.classes import *
from pycity_scheduling.algorithms import *


# This is a very simple power scheduling example using the distributed Exchange ADMM algorithm.


def main(do_plot=False):
    print("\n\n------ Example 05: Algorithm Exchange-ADMM ------\n\n")

    # Define timer, price, weather, and environment objects:
    t = Timer(op_horizon=2, step_size=3600)
    p = Prices(timer=t)
    w = Weather(timer=t)
    e = Environment(timer=t, weather=w, prices=p)

    # City district with district operator objective "peak-shaving":
    cd = CityDistrict(environment=e, objective='peak-shaving')

    # Schedule two sample buildings. The buildings' objectives are defined as "price".

    # Building no. one comes with fixed load, space heating, electrical heater, pv unit, thermal energy storage, and
    # electrical energy storage:
    bd1 = Building(environment=e, objective='price')
    cd.addEntity(entity=bd1, position=[0, 0])
    bes = BuildingEnergySystem(environment=e)
    bd1.addEntity(bes)
    ths = ThermalHeatingStorage(environment=e, e_th_max=40, soc_init=0.5)
    bes.addDevice(ths)
    eh = ElectricalHeater(environment=e, p_th_nom=10)
    bes.addDevice(eh)
    ap = Apartment(environment=e)
    bd1.addEntity(ap)
    load = np.array([10.0, 10.0])
    fi = FixedLoad(e, method=0, demand=load)
    ap.addEntity(fi)
    sh = SpaceHeating(environment=e, method=0, loadcurve=load)
    ap.addEntity(sh)
    pv = Photovoltaic(environment=e, method=1, peak_power=4.6)
    bes.addDevice(pv)
    bat = Battery(environment=e, e_el_max=4.8, p_el_max_charge=3.6, p_el_max_discharge=3.6)
    bes.addDevice(bat)

    # Building no. two comes with deferrable load, curtailable load, space heating, chp unit, thermal energy storage and
    # an electrical vehicle:
    bd2 = Building(environment=e, objective='price')
    cd.addEntity(entity=bd2, position=[0, 0])
    bes = BuildingEnergySystem(environment=e)
    bd2.addEntity(bes)
    ths = ThermalHeatingStorage(environment=e, e_th_max=35, soc_init=0.5)
    bes.addDevice(ths)
    chp = CombinedHeatPower(environment=e, p_th_nom=20.0)
    bes.addDevice(chp)
    ap = Apartment(environment=e)
    bd2.addEntity(ap)
    load = np.array([20.0, 20.0])
    dl = DeferrableLoad(environment=e, p_el_nom=2.0, e_consumption=2.0, load_time=[1, 1])
    ap.addEntity(dl)
    cl = CurtailableLoad(environment=e, p_el_nom=1.6, max_curtailment=0.8)
    ap.addEntity(cl)
    sh = SpaceHeating(environment=e, method=0, loadcurve=load)
    ap.addEntity(sh)
    ev = ElectricalVehicle(environment=e, e_el_max=37.0, p_el_max_charge=22.0, soc_init=0.65, charging_time=[0, 1])
    ap.addEntity(ev)

    # Perform the scheduling:
    opt = ExchangeADMM(city_district=cd, rho=2.0, eps_primal=0.001, eps_dual=0.01)
    results = opt.solve()
    cd.copy_schedule("admm")

    # Print some ADMM results:
    print("ADMM - Number of iterations:")
    print(results["iterations"][-1])
    print("ADMM - Norm vector 'r' over iterations:")
    print(results["r_norms"])
    print("ADMM - Norm vector 's' over iterations:")
    print(results["s_norms"])
    print("")

    # Print the building's schedules:
    print("Schedule building no. one:")
    print(list(bd1.p_el_schedule))
    print("Schedule building no. two:")
    print(list(bd2.p_el_schedule))
    print("Schedule of the city district:")
    print(list(cd.p_el_schedule))
    return


# Conclusions:
# If the distributed exchange ADMM optimization algorithm is applied, the two buildings are scheduled in a way
# so that both the local and system level objectives are satisfied. Local flexibility is used to achieve the system
# level objective. The scheduling results are close to the ones of the central algorithm, which demonstrates the
# correctness of the distributed algorithm.


if __name__ == '__main__':
    # Run example:
    main(do_plot=True)