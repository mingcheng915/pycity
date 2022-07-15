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

from pycity_scheduling.classes import *
from pycity_scheduling.algorithms import *


# This is a very simple power scheduling example using the distributed Exchange MIQP ADMM algorithm.
# It compares the differences of the schedules and the performance of the Exchange MIQP ADMM algorithm in the
# 'unconstrained' and 'constrained' x_update_mode. It also compares the solution of the algorithm in both
# x_update_modes with the solution of the Central Optimization algorithm.


def main(do_plot=False):
    print("\n\n------ Example 22: Algorithm Exchange-MIQP-ADMM ------\n\n")

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
    dl = DeferrableLoad(environment=e, p_el_nom=2.0, e_consumption=2.0, load_time=[1, 1])
    ap.addEntity(dl)
    cl = CurtailableLoad(environment=e, p_el_nom=1.6, max_curtailment=0.8)
    ap.addEntity(cl)
    load = np.array([20.0, 20.0])
    sh = SpaceHeating(environment=e, method=0, loadcurve=load)
    ap.addEntity(sh)
    ev = ElectricalVehicle(environment=e, e_el_max=37.0, p_el_max_charge=22.0, soc_init=0.65, charging_time=[0, 1])
    ap.addEntity(ev)

    # Perform the scheduling with the Central Optimization algorithm as reference:
    opt = CentralOptimization(city_district=cd, mode="integer")
    results = opt.solve()
    cd.copy_schedule("central")

    # Print the building's schedules:
    print("")
    print("")
    print("Central Optimization Algorithm")
    print("")
    print("Schedule building no. one:")
    print(list(bd1.p_el_schedule))
    print("Schedule building no. two:")
    print(list(bd2.p_el_schedule))
    print("Schedule of the city district:")
    print(list(cd.p_el_schedule))

    # Perform the scheduling with the Exchange MIQP ADMM algorithm and a constrained x_update
    opt = ExchangeMIQPADMM(city_district=cd, mode='integer', x_update_mode='constrained', eps_primal=0.01,
                           eps_dual=0.01, eps_primal_i=0.01, eps_dual_i=0.01)
    results = opt.solve()
    cd.copy_schedule("exchange_miqp_admm")

    # Print some results:
    print("")
    print("")
    print("Exchange MIQP ADMM, constrained x-update - Number of iterations:", results["iterations"][-1])
    print("")

    # Print the building's schedules:
    print("Schedule building no. one:")
    print(list(bd1.p_el_schedule))
    print("Schedule building no. two:")
    print(list(bd2.p_el_schedule))
    print("Schedule of the city district:")
    print(list(cd.p_el_schedule))

    # Perform the scheduling with the Exchange MIQP ADMM algorithm and a unconstrained x_update
    """
    opt = ExchangeMIQPADMM(city_district=cd, mode='integer', x_update_mode='unconstrained', eps_primal=0.001,
                           eps_dual=0.001, eps_primal_i=0.001, eps_dual_i=0.001)
   """
    opt = UnconstrainedLight(city_district=cd, mode='integer', x_update_mode='unconstrained', eps_primal=0.001,
                           eps_dual=0.001, eps_primal_i=0.001, eps_dual_i=0.001)
    results = opt.solve()
    cd.copy_schedule("exchange_miqp_admm")

    # Print some results:
    print("")
    print("")
    print("Exchange MIQP ADMM, unconstrained x-update - Number of iterations:", results["iterations"][-1])
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
# If the distributed Exchange MIQP ADMM optimization algorithm is applied, the two buildings are scheduled in a way
# so that both the local and system level objectives are satisfied even in integer mode.
# The Exchange MIQP ADMM algorithm has two different modes: In the "constrained" mode the x-update of the algorithm
# is executed by the solver under the constraints of the optimization problem. In the "unconstrained" mode
# the constraints of the optimization problem are considered by an augmented term in the x-update of the algorithm.
# Therefore, the solver only has to minimize an polynomial objective function as x-update. This example shows that
# the solution of the Exchange MIQP ADMM algorithm is almost identical in both the constrained and the unconstrained
# mode but the algorithm needs more iterations in the unconstrained mode.
# Anyway, both scheduling results are close to the ones of the Central Optimization algorithm, which demonstrates
# the correctness of the distributed algorithm.


if __name__ == '__main__':
    # Run example:
    main(do_plot=True)
