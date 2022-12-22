from pycity_scheduling.algorithms import *
import matplotlib.pyplot as plt
import pycity_scheduling.util.factory as factory
import pycity_scheduling.util.mpi_interface as mpi
import cloudpickle
from pycity_scheduling.util.district_analyzer import DistrictAnalyzer

# Set the parameters for the algorithm
max_iterations = 3500
mode = 'integer'
r_exch = 0.05
s_exch = 0.05
r_dual = 0.05
s_dual = 0.05


# Generate City District and start the algorithm
def main(do_plot=False):
    mpi_interface = mpi.MPIInterface()
    mpi_interface.disable_multiple_printing()

    print("\n\n------ Evaluate Exchange MIQP ADMM ------\n\n")
    # First, create an environment using the factory's "generate_standard_environment" method. The environment
    # automatically encapsulates time, weather, and price data/information.
    env = factory.generate_standard_environment(initial_date=(2020, 1, 27), step_size=900, op_horizon=96)

    # Create N single-family houses:
    num_sfh = 199

    # 50% SFH.2002, 30% SFH.2010, 20% SFH.2016 (based on TABULA):
    sfh_distribution = {
        'SFH.1200': 0.033,
        'SFH.1860': 0.096,
        'SFH.1919': 0.112,
        'SFH.1949': 0.085,
        'SFH.1958': 0.150,
        'SFH.1969': 0.145,
        'SFH.1979': 0.070,
        'SFH.1984': 0.11,
        'SFH.1995': 0.102,
        'SFH.2002': 0.077,
        'SFH.2010': 0.01,
        'SFH.2016': 0.01,
    }

    # 50% of the single-family houses are equipped with heat pump, 10% with boiler, and 40% with electrical heater:
    sfh_heating_distribution = {
        'HP': 0.4,
        'BL': 0.0,
        'EH': 0.2,
        'CHP': 0.4,
    }

    # All single-family houses are equipped with a fixed load, 20% have a deferrable load, and 30% have an electric
    # vehicle. Moreover, 50% of all single-family houses have a battery unit and 80% have a rooftop photovoltaic unit
    # installation.
    # The values are rounded in case they cannot be perfectly matched to the given number of buildings.
    sfh_device_probs = {
        'FL': 1,
        'DL': 0,
        'EV': 0.33,
        'BAT': 0.5,
        'PV': 0.7,
    }

    # Create 0 multi-family houses (number of apartments according to TABULA):
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
    # have an electric vehicle. Moreover, 40% of all multi-family houses have a battery unit and 80% have a rooftop
    # photovoltaic unit installation.
    # The values are rounded in case they cannot be perfectly matched to the given number of buildings.
    mfh_device_probs = {
        'FL': 1,
        'DL': 0.2,
        'EV': 0.2,
        'BAT': 0.4,
        'PV': 0.8,
    }

    # Finally, create the desired city district using the factory's "generate_tabula_district" method. The district's/
    # district operator's objective is defined as "peak-shaving" and the buildings' objectives are defined as "price".
    district = factory.generate_tabula_district(env, num_sfh, num_mfh,
                                                sfh_distribution,
                                                sfh_heating_distribution,
                                                sfh_device_probs,
                                                mfh_distribution,
                                                mfh_heating_distribution,
                                                mfh_device_probs,
                                                district_objective='peak-shaving',
                                                building_objective='none',
                                                seed=0,
                                                mpi_interface=mpi_interface)

    """
    v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 99])
    p = np.array([0, 0, 3, 25, 82, 174, 321, 532, 815, 1180, 1580, 1810, 1980, 2050, 2050, 2050, 2050, 2050, 2050, 2050,
                  2050, 2050, 2050, 2050, 2050, 2050, 0, 0])
    wec = WindEnergyConverter(env, velocity=v, power=p, hub_height=78.0)
    district.addEntity(wec, [0, 0])
    """

    #d_a_pre = DistrictAnalyzer(city_district=district)
    #d_a_pre.complete_pre_analyze()
    #"""
    # Perform the city district scheduling using the Exchange MIQP ADMM Unconstrained Light algorithm:

    opt = ExchangeMIQPADMMMPI(city_district=district, mode=mode, mpi_interface=mpi_interface,
                              x_update_mode='constrained', eps_primal=r_exch, eps_dual=s_exch, eps_primal_i=r_dual,
                              eps_dual_i=s_dual, max_iterations=max_iterations, fix=True)
    
    """
    opt = ExchangeMIQPADMMFix(city_district=district, mode=mode,
                              x_update_mode='constrained', eps_primal=r_exch, eps_dual=s_exch, eps_primal_i=r_dual,
                              eps_dual_i=s_dual, max_iterations=max_iterations)
    
    opt = ExchangeMIQPADMM(city_district=district, mode=mode,
                           x_update_mode='constrained', eps_primal=r_exch, eps_dual=s_exch, eps_primal_i=r_dual,
                           eps_dual_i=s_dual, max_iterations=max_iterations, fix=True)

    """
    results = opt.solve()
    print("Returned")
    district.copy_schedule("district_schedule")
    if mpi_interface.get_rank() == 0 or mpi_interface.get_size() == 1:
        print("Analyse...")
        d_a = DistrictAnalyzer(city_district=district)
        d_a.save_schedules()
        d_a.count_violated_constraints()
        print("Pickle erstellen...")
        file = open('store_analyzer_miqp', 'wb')
        cloudpickle.dump(d_a, file)
        file.close()

    # Plot the scheduling results:
    plt.figure(1)
    plt.plot(district.p_el_schedule)
    plt.ylabel("City District Power [kW]")
    plt.title("Complex City District Scenario - Schedule")
    plt.savefig('schedule.png')
    # Plot the objective values over the iterations
    plt.figure(2)
    plt.plot(results["obj_value"])
    plt.ylabel("Objetive value")
    plt.title("Objetive Value")
    plt.savefig('obj_value.png')
    # Plot the residuals over the iterations
    plt.figure(3)
    plt.plot(results["r_norms"], label='Primal residual')
    plt.plot(results["s_norms"], label='Dual residual')
    plt.ylabel("Residual Norms")
    plt.title("Exchange Problem Residuals")
    plt.savefig('exchange_res.png')
    plt.figure(4)
    plt.plot(results["r_sub_ave"], label='Primal residual')
    plt.plot(results["s_sub_ave"], label='Dual residual')
    plt.ylabel("Residual Norms")
    plt.title("Sub-Problem Residuals")
    plt.savefig('sub_res.png')

    # Print some results
    print("Iterations Exchange MIQP ADMM: ", results["iterations"][-1])
    print("Objective value: ", results["obj_value"][-1])

    if do_plot:
        plt.show()
    return


if __name__ == '__main__':
    # Run example:
    main(do_plot=False)
