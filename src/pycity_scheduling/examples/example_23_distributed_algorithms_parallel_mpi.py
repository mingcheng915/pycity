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
import matplotlib.pyplot as plt

import pycity_scheduling.util.factory as factory
import pycity_scheduling.util.debug as debug
import pycity_scheduling.util.mpi_interface as mpi
from pycity_scheduling.util.metric import self_consumption, autarky
from pycity_scheduling.algorithms import *
from pycity_scheduling.classes import *


# In this example, the power schedule for a city district scenario is determined by means of real parallel distributed
# optimization algorithm implementations using the Message Passing Interface (MPI) standard. The scenario is built upon
# the district setup as defined in example 'example_13_district_generator.py', but it contains just a few buildings for
# the sake of demonstration.

def main(do_plot=False):
    # First, instantiate the MPI interface and disable printing for all MPI processes other than with rank 0.
    mpi_interface = mpi.MPIInterface()
    mpi_interface.disable_multiple_printing()

    print("\n\n------ Example 23: Distributed Algorithms Parallel MPI ------\n\n")

    # Create an environment using the factory's "generate_standard_environment" method. The environment
    # automatically encapsulates time, weather, and price data/information.
    env = factory.generate_standard_environment(initial_date=(2018, 12, 6), step_size=3600, op_horizon=24)

    # Create single-family houses:
    num_sfh = 3

    # 50% SFH.2002, 30% SFH.2010, 20% SFH.2016 (based on TABULA):
    sfh_distribution = {
        'SFH.2002': 0.5,
        'SFH.2010': 0.3,
        'SFH.2016': 0.2,
    }

    # 50% of the single-family houses are equipped with heat pump, 10% with boiler, and 40% with electrical heater:
    sfh_heating_distribution = {
        'HP': 0.5,
        'BL': 0.1,
        'EH': 0.4,
    }

    # All single-family houses are equipped with a fixed load, 20% have a deferrable load, and 30% have an electric
    # vehicle. Moreover, 50% of all single-family houses have a battery unit and 100% have a rooftop photovoltaic unit
    # installation.
    # The values are rounded in case they cannot be perfectly matched to the given number of buildings.
    sfh_device_probs = {
        'FL': 1,
        'DL': 0.2,
        'EV': 0.3,
        'BAT': 0.5,
        'PV': 1.0,
    }

    # Create multi-family houses (number of apartments according to TABULA):
    num_mfh = 0

    # 60% MFH.2002, 20% SFH.2010, 20% SFH.2016 (based on TABULA):
    mfh_distribution = {
        'MFH.2002': 0.6,
        'MFH.2010': 0.2,
        'MFH.2016': 0.2,
    }

    # 40% of the multi-family houses are equipped with heat pump, 20% with boiler, and 40% with electrical heater:
    mfh_heating_distribution = {
        'HP': 0.4,
        'BL': 0.2,
        'EH': 0.4,
    }

    # All apartments inside a multi-family houses are equipped with a fixed load, 20% have a deferrable load, and 20%
    # have an electric vehicle. Moreover, 40% of all multi-family houses have a battery unit and 100% have a rooftop
    # photovoltaic unit installation.
    # The values are rounded in case they cannot be perfectly matched to the given number of buildings.
    mfh_device_probs = {
        'FL': 1,
        'DL': 0.2,
        'EV': 0.2,
        'BAT': 0.4,
        'PV': 1.0,
    }

    # Finally, create the desired city district using the factory's "generate_tabula_district" method. The district's
    # district operator's objective is defined as "peak-shaving" and the buildings' objectives are defined as
    # "peak-shaving", too.
    district = factory.generate_tabula_district(env, num_sfh, num_mfh,
                                                sfh_distribution,
                                                sfh_heating_distribution,
                                                sfh_device_probs,
                                                mfh_distribution,
                                                mfh_heating_distribution,
                                                mfh_device_probs,
                                                district_objective='peak-shaving',
                                                building_objective='peak-shaving'
                                                )

    # Hierarchically print the district and all buildings/assets:
    debug.print_district(district, 1)

    # Perform the city district scheduling using the central optimization algorithm as a reference:
    print("\n### Central Algorithm ###\n")
    opt = CentralOptimization(district)
    results = opt.solve()
    district.copy_schedule("central")

    # Print the building's schedules and some metrics:
    count = 1
    for building in district.get_lower_entities():
        print("Schedule building {}:".format(str(building)))
        print(list(building.p_el_schedule))
    print("Schedule of the city district:")
    print(list(district.p_el_schedule))
    print("")
    print("Self-consumption rate: {: >4.2f}".format(self_consumption(district)))
    print("Autarky rate: {: >4.2f}".format(autarky(district)))

    # Perform the city district scheduling using the Dual Decomposition optimization algorithm:
    print("\n### Dual Decomposition Algorithm ###\n")
    opt = DualDecompositionMPI(district, mpi_interface, rho=0.1, eps_primal=0.01)
    results = opt.solve()
    district.copy_schedule("dual_decomposition")

    # Print the building's schedules:
    count = 1
    for building in district.get_lower_entities():
        print("Schedule building {}:".format(str(building)))
        print(list(building.p_el_schedule))
    print("Schedule of the city district:")
    print(list(district.p_el_schedule))
    print("")
    print("Self-consumption rate: {: >4.2f}".format(self_consumption(district)))
    print("Autarky rate: {: >4.2f}".format(autarky(district)))

    # Perform the city district scheduling using the Exchange ADMM optimization algorithm:
    print("\n### Exchange ADMM Algorithm ###\n")
    opt = ExchangeADMMMPI(district, mpi_interface, rho=2.0, eps_primal=0.001, eps_dual=0.01)
    results = opt.solve()
    district.copy_schedule("exchange_admm")

    # Print the building's schedules:
    count = 1
    for building in district.get_lower_entities():
        print("Schedule building {}:".format(str(building)))
        print(list(building.p_el_schedule))
    print("Schedule of the city district:")
    print(list(district.p_el_schedule))
    print("")
    print("Self-consumption rate: {: >4.2f}".format(self_consumption(district)))
    print("Autarky rate: {: >4.2f}".format(autarky(district)))

    return


# Conclusions:
# Using MPI, we can facilitate the real parallel execution of distributed optimization algorithms - even on
# loosely-coupled machines. The parallel implementations allow for significant computational speedups and are therefore
# suitable to tackle large-scale optimization problems that cannot be solved by an off-the-shelf centralized
# computation directly.


if __name__ == '__main__':
    # Run example:
    main(do_plot=True)
