"""
Python class to analyze the generated city district in terms of the total generated and consumed power
"""
import numpy as np
import pyomo.environ as pyomo
import pyomo.kernel as pmo
import matplotlib.pyplot as plt
from pycity_scheduling.classes import *


class DistrictAnalyzer:
    """
    Python class to analyze the generated city district in terms of the total generated and consumed power
    """
    figure_number = 5

    def __init__(self, city_district):
        self.city_district = city_district
        self.op_horizon = city_district.op_horizon
        self.initial_time_step = self.city_district.environment.timer.time_in_year()
        self.end_time_step = self.initial_time_step + self.op_horizon - 1
        self.PVPower = self._get_total_pv_generation()
        self.FLPower = self._get_total_fixed_load()
        self.HeatDemand, self.FossileHeat, self.ElectricalHeat,\
            self.ElectricalPowerForHeat = self._get_heat_demand()
        self.CHPGeneration = self._get_total_chp_generation()
        self.DLSchedule = np.zeros(self.op_horizon)
        self.BatSchedule = np.zeros(self.op_horizon)
        self.EVSchedule = np.zeros(self.op_horizon)
        self.CHPSchedule = np.zeros(self.op_horizon)
        self.EHSchedule = np.zeros(self.op_horizon)
        self.HPSchedule = np.zeros(self.op_horizon)
        self.final_demand_schedule = np.zeros(self.op_horizon)
        self.final_generation_schedule = np.zeros(self.op_horizon)
        self.district_schedule = np.zeros(self.op_horizon)

    def _get_total_pv_generation(self):
        """
        Function that returns the total PV generated power of the city district in kW per time step
        """
        annual_power = np.zeros(35040)
        total_horizon_power = np.zeros(self.op_horizon)
        for en in self.city_district.get_all_entities():
            if isinstance(en, Photovoltaic):
                annual_power = np.add(annual_power, en.getPower())
        for i in range(self.op_horizon):
            total_horizon_power[i] += annual_power[self.initial_time_step + i] / 1000
        return total_horizon_power

    def _get_total_fixed_load(self):
        """
        Function that returns the total electrical power demand caused by fixed loads in kW per time step
        """
        total_fixed_load = np.zeros(self.op_horizon)
        for en in self.city_district.get_all_entities():
            if isinstance(en, FixedLoad):
                total_fixed_load = np.add(total_fixed_load, en.p_el_schedule)
        return total_fixed_load

    def _get_total_deferrable_load(self):
        """
        Function that returns the total electrical energy demand caused by deferrable loads in kWh
        """
        total_deferrable_load = 0
        for en in self.city_district.get_all_entities():
            if isinstance(en, DeferrableLoad):
                total_deferrable_load += en.e_consumption
        return total_deferrable_load

    def _get_total_battery_capacity(self):
        """
        Function that returns the total storage capacities of classical batteries and electric vehicles in kWh.
        Note, at the end of a simultated period, the soc has to be grater or equal to the initial soc per default.
        Therefore, the calculated total battery capacity in this function is a maximum!
        """
        total_battery_capacity = 0
        for en in self.city_district.get_all_entities():
            if isinstance(en, Battery):
                total_battery_capacity += en.e_el_max*(1-en.soc_init)
        return total_battery_capacity

    def _get_heat_demand(self):
        """
        Function that returns the total heat demand in kWh per time step and the corresponding electrical demand
        in kWh per time step
        """
        total_heat_demand = np.zeros(self.op_horizon)
        total_fossil_generated_heat = np.zeros(self.op_horizon)
        total_electrical_generated_heat = np.zeros(self.op_horizon)
        total_el_power_for_heat = np.zeros(self.op_horizon)
        for building in self.city_district.get_lower_entities():
            building_heat_demand = np.zeros(self.op_horizon)
            eta = 0
            cop = np.zeros(self.op_horizon)
            device = ''
            for en in building.get_all_entities():
                if isinstance(en, SpaceHeating):
                    total_heat_demand = np.add(total_heat_demand, en.p_th_heat_schedule)
                    building_heat_demand = en.p_th_heat_schedule
                if isinstance(en, ElectricalHeater):
                    device = 'EH'
                    eta = en.eta
                if isinstance(en, HeatPump):
                    device = 'HP'
                    cop = en.cop
                if isinstance(en, CombinedHeatPower):
                    device = 'CHP'
            if device == 'EH':
                total_electrical_generated_heat = np.add(total_electrical_generated_heat, building_heat_demand)
                building_el_heat_demand = building_heat_demand*eta
            elif device == 'HP':
                total_electrical_generated_heat = np.add(total_electrical_generated_heat, building_heat_demand)
                building_el_heat_demand = np.multiply(building_heat_demand, 1/cop)
            elif device == 'CHP' or device == 'BL':
                total_fossil_generated_heat = np.add(total_fossil_generated_heat, building_heat_demand)
                building_el_heat_demand = np.zeros(self.op_horizon)
            else:
                building_el_heat_demand = np.zeros(self.op_horizon)
                print("Device not listed")
            total_el_power_for_heat = np.add(total_el_power_for_heat, building_el_heat_demand)

        return total_heat_demand, total_fossil_generated_heat, total_electrical_generated_heat, total_el_power_for_heat

    def _get_total_chp_generation(self):
        """
        Pre analyze
        """
        total_chp_generation = np.zeros(self.op_horizon)
        for en in self.city_district.get_all_entities():
            p_el_chp = 0
            power = np.zeros(self.op_horizon)
            if isinstance(en, CombinedHeatPower):
                # Estimation: CHP is gonna run with 80 % of the nominal power in average
                p_el_chp = en.p_el_nom * 0.8
                for i in range(self.op_horizon):
                    power[i] = p_el_chp
            total_chp_generation = np.add(total_chp_generation, power)
        return total_chp_generation

    def plot_electrical_generation_profile(self, plot_now=False):
        """
        Pre analyze
        """
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.PVPower, label='PV', linestyle='dashed')
        plt.plot(self.CHPGeneration, label='CHP', linestyle='dashed')
        plt.plot(self.PVPower+self.CHPGeneration, label='total')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Electrical Generation Profile')
        plt.grid()
        if plot_now == True:
            plt.show()
        return

    def plot_electrical_demand_profile(self, plot_now=False):
        """
        Pre analyze
        """
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.FLPower+self.ElectricalPowerForHeat, label='Total demand')
        plt.plot(self.FLPower, label='FL', linestyle='dashed')
        plt.plot(self.ElectricalPowerForHeat, label='Electrical Power for Heat', linestyle='dashed')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Electrical Demand Profile')
        plt.grid()
        if plot_now == True:
            plt.show()

    def plot_heat_demand_profile(self, plot_now=False):
        """
        Pre analyze
        """
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.HeatDemand, label='Heat demand')
        plt.plot(self.FossileHeat, label='Fossil generated heat', linestyle='dashed')
        plt.plot(self.ElectricalHeat, label='Electrical generated heat', linestyle='dashed')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Heat Demand and Generation Profile')
        plt.grid()
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.ElectricalPowerForHeat, label='ELectrical power')
        plt.plot(self.ElectricalHeat, label='Electrical generated heat')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Electrical Power Demand for Electrically Generated Heat')
        plt.grid()
        if plot_now == True:
            plt.show()
        return

    def plot_electrical_power_imbalance(self, plot_now=False):
        """
        Pre analyze
        """
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(-self.PVPower-self.CHPGeneration, label='Electrical Power Generation', linestyle='dashed')
        plt.plot(self.FLPower+self.ElectricalPowerForHeat, label='Electrical Power Demand', linestyle='dashed')
        plt.plot(-self.PVPower-self.CHPGeneration+self.FLPower+self.ElectricalPowerForHeat, label='Power Imbalance')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Electrical Power Imbalance')
        plt.grid()
        if plot_now == True:
            plt.show()
        return

    def complete_pre_analyze(self):
        """
        Function to plot an estimation of the  city districts schedules if no optimization was done.
        """
        total_generation = self.PVPower + self.CHPGeneration
        total_load = self.FLPower + self.ElectricalPowerForHeat
        total_flexibility = self._get_total_deferrable_load() + self._get_total_battery_capacity()
        imbalance = np.subtract(total_load, total_generation)
        net_imbalance = 0
        for i in range(self.op_horizon):
            net_imbalance += imbalance[i]
        print("net-imbalance: ", net_imbalance)
        print("Flexibility_potential", total_flexibility)

        self.plot_heat_demand_profile()
        self.plot_electrical_demand_profile()
        self.plot_electrical_generation_profile()
        self.plot_electrical_power_imbalance()
        plt.show()
        return

    def add_vectors(self, numpy_vector, pyomo_vector):
        """
        Function to add an indexed pyomo variable with a numpy array elementwise
        """
        for i in range(self.op_horizon):
            numpy_vector[i] += pyomo.value(pyomo_vector[i])
        return numpy_vector

    def convert_pyomo_to_numpy(self, pyomo_vector):
        """
        Function that converts an indexed pyomo variable to a numpy array
        """
        temp = np.zeros(self.op_horizon)
        for i in range(self.op_horizon):
            temp[i] += pyomo.value(pyomo_vector[i])
        return temp

    def _get_flexible_device_schedules(self):
        """
        Function that calculates and saves the aggregated schedules for the flexible entity types DL, Bat, EV
        """
        DL_schedule = np.zeros(self.op_horizon)
        Bat_schedule = np.zeros(self.op_horizon)
        EV_schedule = np.zeros(self.op_horizon)
        for en in self.city_district.get_all_entities():
            if isinstance(en, DeferrableLoad):
                DL_schedule = self.add_vectors(DL_schedule, en.model.p_el_vars)
            if isinstance(en, Battery) and isinstance(en, ElectricalVehicle):
                EV_schedule = self.add_vectors(EV_schedule, en.model.p_el_vars)
            if isinstance(en, Battery) and not isinstance(en, ElectricalVehicle):
                Bat_schedule = self.add_vectors(Bat_schedule, en.model.p_el_vars)
        self.DLSchedule = DL_schedule
        self.BatSchedule = Bat_schedule
        self.EVSchedule = EV_schedule
        return

    def _get_heat_device_schedules(self):
        """
        Function that calculates and saves the aggregated schedules for the heating entities CHP, EH, HP
        """
        CHP_schedule = np.zeros(self.op_horizon)
        EH_schedule = np.zeros(self.op_horizon)
        HP_schedule = np.zeros(self.op_horizon)
        for en in self.city_district.get_all_entities():
            if isinstance(en, CombinedHeatPower):
                CHP_schedule = self.add_vectors(CHP_schedule, en.model.p_el_vars)
            if isinstance(en, ElectricalHeater):
                EH_schedule = self.add_vectors(EH_schedule, en.model.p_el_vars)
            if isinstance(en, HeatPump):
                HP_schedule = self.add_vectors(HP_schedule, en.model.p_el_vars)
        self.CHPSchedule = CHP_schedule
        self.EHSchedule = EH_schedule
        self.HPSchedule = HP_schedule
        return

    def plot_flexible_device_schedules(self, plot_now=False):
        self._get_flexible_device_schedules()
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.DLSchedule, label='DL', linestyle='dashed')
        plt.plot(self.EVSchedule, label='EV', linestyle='dashed')
        plt.plot(self.BatSchedule, label='Bat', linestyle='dashed')
        plt.plot(self.BatSchedule+self.DLSchedule+self.EVSchedule, label='total')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Flexible Devices Profile')
        plt.grid()
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.BatSchedule+self.DLSchedule+self.EVSchedule, label='total')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Aggregated Flexible Devices Profile')
        plt.grid()
        if plot_now:
            plt.show()
        return

    def _extract_demand_and_supply(self, schedule):
        """
        Function to seperate the schedules of Batteries (including Evs) into schedules that contain the demanding and
        the supplying time-steps of flexible loads
        """
        demand_schedule = np.zeros(self.op_horizon)
        supply_schedule = np.zeros(self.op_horizon)
        for i in range(self.op_horizon):
            if schedule[i] > 0:
                demand_schedule[i] = schedule[i]
            else:
                supply_schedule[i] = schedule[i]
        return [demand_schedule, supply_schedule]

    def plot_demand_profile(self, plot_now=False):
        self._get_heat_device_schedules()
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.FLPower, label='Fixed Load', linestyle='dashed')
        plt.plot(self._extract_demand_and_supply(self.EVSchedule)[0], label='EV charge', linestyle='dashed')
        plt.plot(self._extract_demand_and_supply(self.BatSchedule)[0], label='BAT charge', linestyle='dashed')
        plt.plot(self.DLSchedule, label='Deferrable Load', linestyle='dashed')
        plt.plot(self.EHSchedule, label='EH', linestyle='dashed')
        plt.plot(self.HPSchedule, label='HP', linestyle='dashed')
        self.final_demand_schedule = self.FLPower + self._extract_demand_and_supply(self.EVSchedule)[0] + \
                                     self._extract_demand_and_supply(self.BatSchedule)[0] + self.DLSchedule + \
                                     self.EHSchedule + self.HPSchedule
        plt.plot(self.final_demand_schedule, label='total')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Electrical Power Demand Profile')
        plt.grid()
        if plot_now:
            plt.show()
        return

    def plot_generation_profile(self, plot_now=False):
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(-self.PVPower, label='PV', linestyle='dashed')
        plt.plot(self._extract_demand_and_supply(self.EVSchedule)[1], label='EV discharge', linestyle='dashed')
        plt.plot(self._extract_demand_and_supply(self.BatSchedule)[1], label='BAT discharge', linestyle='dashed')
        plt.plot(self.CHPSchedule, label='CHP', linestyle='dashed')
        self.final_generation_schedule = -self.PVPower + self._extract_demand_and_supply(self.EVSchedule)[1] + \
                                     self._extract_demand_and_supply(self.BatSchedule)[1] + self.CHPSchedule
        plt.plot(self.final_generation_schedule, label='total')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Electrical Power Generation Profile')
        plt.grid()
        if plot_now:
            plt.show()
        return

    def complete_post_analyze(self):
        self.plot_flexible_device_schedules()
        self.plot_demand_profile()
        self.plot_generation_profile()
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.final_demand_schedule, label='Total demand', linestyle='dashed')
        plt.plot(self.final_generation_schedule, label='Total generation', linestyle='dashed')
        plt.plot(self.final_generation_schedule+self.final_demand_schedule, label= 'Total Imbalance')
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Electrical Imbalance Profile')
        plt.grid()
        #plt.show()
        return

    def print_schedules(self):
        return

    def _get_entity_schedules(self, entity_class):
        """
        Function that returns all schedules of one type of entity, e.g. Batteries
        """
        entity_list = []
        index = 0
        for en in self.city_district.get_all_entities():
            if entity_class == Battery and isinstance(en, Battery) and not isinstance(en, ElectricalVehicle):
                schedule = self.convert_pyomo_to_numpy(en.model.p_el_vars)
                entity_list.append([])
                entity_list[index].append(en)
                entity_list[index].append(schedule)
                index += 1
            elif entity_class != Battery and isinstance(en, entity_class):
                schedule = self.convert_pyomo_to_numpy(en.model.p_el_vars)
                entity_list.append([])
                entity_list[index].append(en)
                entity_list[index].append(schedule)
                index += 1

        return entity_list

    def plot_entity_schedules(self, entity_class):
        """
        Function that plots all schedules of one type of entity, e.g. Batteries
        """
        entity_list = self._get_entity_schedules(entity_class)
        for i in range(len(entity_list)):
            plt.figure(DistrictAnalyzer.figure_number)
            DistrictAnalyzer.figure_number += 1
            plt.plot(entity_list[i][1])
            plt.xlabel("Time step")
            plt.ylabel("Power in kW")
            plt.title(entity_list[i][0])
            plt.grid()
        return

    def plot_city_district_schedule(self):
        """
        Function that plots the city districts' final schedule
        """
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(self.city_district.p_el_schedule)
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('City District Schedule')
        plt.grid()
        return

    def _compare_schedules(self, schedule_a, schedule_b):
        """
        Function that returns the average absolute deviation between two schedules in kW per time step
        """
        sum = 0
        for i in range(self.op_horizon):
            diff = schedule_a[i] - schedule_b[i]
            if diff < 0:
                diff = diff*(-1)
            print(diff)
            sum += diff
        average_deviation = sum / self.op_horizon
        return average_deviation

    def save_schedules(self):
        self._get_heat_device_schedules()
        self._get_flexible_device_schedules()
        self.district_schedule = np.add(self.city_district.p_el_schedule, np.zeros(self.op_horizon))
        return

    def compare_aggregated_entities(self, first_optimization,
                                    first_algorithm_name, second_algorithm_name, save_plots=False,):
        """
        Function to compare the aggregated schedules of each entity type, if different algorithms were used for the
        optimization of the same city district.
        It is mandatory that both optimizations are started in the same script and that an own DistrictAnalyzer object
        is created for each optimization. After the first executed optimization, the DistrictAnalyzer class method
        save_aggregated_schedules() has to be called to save the schedules for the later comparing.
        """
        if not np.array_equal(self.CHPSchedule, np.zeros(self.op_horizon)):
            plt.figure(DistrictAnalyzer.figure_number)
            DistrictAnalyzer.figure_number += 1
            plt.plot(first_optimization.CHPSchedule, label=first_algorithm_name)
            plt.plot(self.CHPSchedule, label=second_algorithm_name)
            plt.legend()
            plt.xlabel("Time step")
            plt.ylabel("Power in kW")
            plt.title('Aggregated CHP-Schedule Comparison')
            plt.grid()
            if save_plots:
                plt.savefig('aggregated_chp_schedules.png')

        if not np.array_equal(self.HPSchedule, np.zeros(self.op_horizon)):
            plt.figure(DistrictAnalyzer.figure_number)
            DistrictAnalyzer.figure_number += 1
            plt.plot(self.HPSchedule, label=second_algorithm_name)
            plt.plot(first_optimization.HPSchedule, label=first_algorithm_name)
            plt.legend()
            plt.xlabel("Time step")
            plt.ylabel("Power in kW")
            plt.title('Aggregated HP-Schedule Comparison')
            plt.grid()
            if save_plots:
                plt.savefig('aggregated_hp_schedules.png')

        if not np.array_equal(self.EHSchedule, np.zeros(self.op_horizon)):
            plt.figure(DistrictAnalyzer.figure_number)
            DistrictAnalyzer.figure_number += 1
            plt.plot(self.EHSchedule, label=second_algorithm_name)
            plt.plot(first_optimization.EHSchedule, label=first_algorithm_name)
            plt.legend()
            plt.xlabel("Time step")
            plt.ylabel("Power in kW")
            plt.title('Aggregated EH-Schedule Comparison')
            plt.grid()
            if save_plots:
                plt.savefig('aggregated_eh_schedules.png')

        if not np.array_equal(self.DLSchedule, np.zeros(self.op_horizon)):
            plt.figure(DistrictAnalyzer.figure_number)
            DistrictAnalyzer.figure_number += 1
            plt.plot(self.DLSchedule, label=second_algorithm_name)
            plt.plot(first_optimization.DLSchedule, label=first_algorithm_name)
            plt.legend()
            plt.xlabel("Time step")
            plt.ylabel("Power in kW")
            plt.title('Aggregated DL-Schedule Comparison')
            plt.grid()
            if save_plots:
                plt.savefig('aggregated_dl_schedules.png')

        if not np.array_equal(self.EVSchedule, np.zeros(self.op_horizon)):
            plt.figure(DistrictAnalyzer.figure_number)
            DistrictAnalyzer.figure_number += 1
            plt.plot(self.EVSchedule, label=second_algorithm_name)
            plt.plot(first_optimization.EVSchedule, label=first_algorithm_name)
            plt.legend()
            plt.xlabel("Time step")
            plt.ylabel("Power in kW")
            plt.title('Aggregated EV-Schedule Comparison')
            plt.grid()
            if save_plots:
                plt.savefig('aggregated_ev_schedules.png')
        return

    def compare_city_ditrict_schedules(self, first_optimization, first_algorithm_name, second_algorithm_name,
                                       save_plots=False):
        """
        Function that compares the final city district schedules if two different optimization algorithms were used.

        """
        plt.figure(DistrictAnalyzer.figure_number)
        DistrictAnalyzer.figure_number += 1
        plt.plot(first_optimization.district_schedule, label=first_algorithm_name)
        plt.plot(self.city_district.p_el_schedule, label=second_algorithm_name)
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Power in kW")
        plt.title('Compare Final Schedules')
        plt.grid()
        if save_plots:
            plt.savefig('compare_district_schedules.png')
        return

    def compare_entity_schedules(self, entity_class, first_optimization, first_algorithm_name, second_algorithm_name):
        first_entity_list = first_optimization._get_entity_schedules(entity_class)
        second_entity_list = self._get_entity_schedules(entity_class)
        for i in range(len(first_entity_list)):
            plt.figure(DistrictAnalyzer.figure_number)
            DistrictAnalyzer.figure_number += 1
            plt.plot(first_entity_list[i][1], label=first_algorithm_name)
            plt.plot(second_entity_list[i][1], label=second_algorithm_name)
            plt.xlabel("Time step")
            plt.ylabel("Power in kW")
            plt.title(first_entity_list[i][0])
            plt.grid()
        return

    def _get_constraints(self):
        equality_list = []
        inequality_list = []
        counter = 0
        for en in self.city_district.get_all_entities():
            for constraint in en.model.component_objects(pyomo.Constraint):
                counter +=1
                for index in constraint:
                    # check if the constraint is an equality constraint and write it in the form Ax-b=0
                    if pyomo.value(constraint[index].lower) == pyomo.value(constraint[index].upper):
                        expr = constraint[index].body - constraint[index].lower
                        equality_list.append(expr)

                    # if the constraint is not an equality constraint it has to be an inequality constraint
                    # the next three checks are about to write that constraint in the form Cx-d >= 0
                    elif pyomo.value(constraint[index].upper) is None:
                        expr = constraint[index].body - constraint[index].lower
                        inequality_list.append([constraint, expr])

                    elif pyomo.value(constraint[index].lower) is None:
                        expr = -constraint[index].body + constraint[index].upper
                        inequality_list.append([constraint, expr])

                    else:
                        expr = -constraint[index].body + constraint[index].upper
                        inequality_list.append([constraint, expr])
                        expr = constraint[index].body - constraint[index].lower
                        inequality_list.append([constraint, expr])
        print("Equal: ", len(equality_list))
        print("Inequal: ", len(inequality_list))
        print("Gesamt: ", counter)

        return equality_list, inequality_list

    def count_violated_constraints(self):
        equality_list, inequality_list = self._get_constraints()
        equality_counter = 0
        inequality_counter = 0
        # check if constraints are violated. Allow a numerical tolerance of 1e-3 which equals 1 Watt in the application
        for i in range(len(equality_list)):
            if abs(0-pyomo.value(equality_list[i])) > 1e-2:
                print("Equality violation")
                print(equality_list[i], "   ", pyomo.value(equality_list[i]))
                print("")
                equality_counter += 1
        for i in range(len(inequality_list)):
            if pyomo.value(inequality_list[i][1]) < -1e-2:
                inequality_counter += 1
                print("Inequality violation: ", inequality_list[i][0])
                print(inequality_list[i][1], "   ", pyomo.value(inequality_list[i][1]))
                print("")

        print("Number of inequality constraints:", len(inequality_list)," Number of violations: ", inequality_counter)
        print("Number of equality constraints:", len(equality_list)," Number of violations: ", equality_counter)
        return

    def check_start_vars_DL(self):
        for en in self.city_district.get_all_entities():
            if isinstance(en, DeferrableLoad):
                for variable in en.model.component_objects(pyomo.Var):
                    if variable[0] == 0 or variable[0] == 1:
                        variable.pprint()
        return
























