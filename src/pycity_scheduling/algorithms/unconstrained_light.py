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
import pyomo.core as pyo_core
import pyomo.kernel as pmo
import pyomo.environ as pyomo

from pycity_scheduling.classes import (CityDistrict, Building, Photovoltaic, WindEnergyConverter)
from pycity_scheduling.util import extract_pyomo_values
from pycity_scheduling.algorithms.algorithm import IterationAlgorithm, DistributedAlgorithm, SolverNode
from pycity_scheduling.solvers import DEFAULT_SOLVER, DEFAULT_SOLVER_OPTIONS


class UnconstrainedLight(IterationAlgorithm, DistributedAlgorithm):
    """Implementation of the Exchange ADMM Algorithm.

    Uses the Exchange MIQP ADMM algorithm.

    Parameters
    ----------
    city_district : CityDistrict
    solver : str, optional
        Solver to use for solving (sub)problems.
    solver_options : dict, optional
        Options to pass to calls to the solver. Keys are the name of
        the functions being called and are one of `__call__`, `set_instance_`,
        `solve`.
        `__call__` is the function being called when generating an instance
        with the pyomo SolverFactory.  Additionally to the options provided,
        `node_ids` is passed to this call containing the IDs of the entities
        being optimized.
        `set_instance` is called when a pyomo Model is set as an instance of
        a persistent solver. `solve` is called to perform an optimization. If
        not set, `save_results` and `load_solutions` may be set to false to
        provide a speedup.
    mode : str, optional
        Specifies which set of constraints to use.
        - `convex`  : Use linear constraints
        - `integer`  : May use non-linear constraints
    x_update_mode: str, optional
        Specifies how the constraints are considered
        - 'constrained' : Constraints are considered by the solver by providing an constrained x-update to it
        - 'unconstrained' : Constraints are considered by an ADMM method with an augmented term
    eps_primal : float, optional
        Primal stopping criterion for the exchange problem solved by the Exchange MIQP ADMM algorithm.
    eps_dual : float, optional
        Dual stopping criterion for the exchange problem solved by the Exchange MIQP ADMM algorithm.
    eps_primal_i : float, optional
        Primal stopping criterion for the constrained sub-problems solved by the Exchange MIQP ADMM algorithm.
    eps_dual_i : float, optional
        Dual stopping criterion for the constrained sub-problems solved by the Exchange MIQP ADMM algorithm.
    rho : float, optional
        Stepsize for the ADMM algorithm.
    max_iterations : int, optional
        Maximum number of ADMM iterations.
    robustness : tuple, optional
        Tuple of two floats. First entry defines how many time steps are
        protected from deviations. Second entry defines the magnitude of
        deviations which are considered.
    """
    def __init__(self, city_district, solver=DEFAULT_SOLVER, solver_options=DEFAULT_SOLVER_OPTIONS, mode="integer",
                 x_update_mode='unconstrained', eps_primal=0.1, eps_dual=0.1, eps_primal_i=0.1, eps_dual_i=0.1,
                 rho=2, max_iterations=3000, robustness=None):
        super(UnconstrainedLight, self).__init__(city_district, solver, solver_options, mode)
        self.city_district = city_district
        self.mode = mode
        self.x_update_mode = x_update_mode
        self.eps_primal = eps_primal
        self.eps_dual = eps_dual
        self.eps_primal_i = eps_primal_i
        self.eps_dual_i = eps_dual_i
        self.max_iterations = max_iterations
        self.counter = 0
        # create solver nodes for each entity
        self.nodes = [
            SolverNode(solver, solver_options, [entity], mode, robustness=robustness)
            for entity in self.entities
        ]
        # load data from the model
        self.constraint_list = self._get_constraints()
        self.variable_list = self._get_variables()
        self.op_horizon = self.entities[0].op_horizon
        # create pyomo parameters for each entity
        for i, node, entity in zip(range(len(self.entities)), self.nodes, self.entities):
            node.model.beta = pyomo.Param(mutable=True, initialize=1)
            node.model.x_exch_ = pyomo.Param(entity.model.t, mutable=True, initialize=0)
            node.model.u_exch = pyomo.Param(entity.model.t, mutable=True, initialize=0)
            node.model.last_p_el_schedules = pyomo.Param(entity.model.t, mutable=True, initialize=0)
            node.model.rho = pyomo.Param(mutable=True, initialize=rho, within=pyomo.Reals)
        # Lists that contain the dual parameters of all subsystems
        self.x_k, self.u_var, self.u_constr = self._set_parameters()
        self._add_objective()

    # Function that gets the binary state variables of each subsystem from the model
    # It also changes the domain of the binary variables to real variables, since they are rounded on binaries later
    # The data structure in which the variables are stored is shown graphically in "data_structure.pdf"
    def _get_variables(self):
        variable_list = np.empty(shape=0)
        for i, node, entity in zip(range(len(self.entities)), self.nodes, self.entities):
            binaries_list = Variables(self.entities[0].op_horizon)
            if i != 0:
                for en in entity.get_all_entities():
                    for variable in en.model.component_objects(pyomo.Var):
                        if variable[0].domain is pmo.Binary:
                            binaries_list.append(variable)

            variable_list = np.append(variable_list, binaries_list)

        return variable_list

    # Function that gets the constraints of each subsystem from the model
    # It sorts them after equality and inequality constraints and writes them in the form Ax-b = 0 or Cx-d >= 0
    # The data structure in which the constraints are stored is shown graphically in "data_structure.pdf"
    def _get_constraints(self):
        # List that contains the dictionaries that store the specified constr_lists of each node
        constraint_list = np.empty(shape=0)
        for i, node, entity in zip(range(len(self.entities)), self.nodes, self.entities):
            node_dict = {}
            equality_constr_list = Constraints(self.entities[0].op_horizon)
            if i != 0:
                for en in entity.get_all_entities():
                    for constraint in en.model.component_objects(pyomo.Constraint):
                        # deactivate constraints in unconstrained mode, since they are considered by an augmented term
                        if self.x_update_mode == 'unconstrained':
                            constraint.deactivate()
                        for index in constraint:
                            # check if the constraint is an equality constraint and write it in the form Ax-b=0
                            if pyomo.value(constraint[index].lower) == pyomo.value(constraint[index].upper):
                                expr = constraint[index].body - constraint[index].lower
                                equality_constr_list.append(expr, index)

                            else:
                                constraint.activate()

            node_dict["equality_constr"] = equality_constr_list
            constraint_list = np.append(constraint_list, node_dict)

        return constraint_list

    # Function that sets the pyomo parameters for the dual variables for each subsystem
    # The parameters are stored with the same data structure as the constraints and variables
    def _set_parameters(self):
        # x_k to save the solution of x for next iteration
        x_k_param = np.empty(shape=0)
        # dual parameter for the binary variables
        u_var_param = np.empty(shape=0)
        # dual parameter for the constraints
        u_constr_param = np.empty(shape=0)
        for i, node, entity in zip(range(len(self.entities)), self.nodes, self.entities):
            node_dict = {}
            x_k_binaries_list = Variables(self.entities[0].op_horizon)
            u_binaries_list = Variables(self.entities[0].op_horizon)
            u_eq_list = Constraints(self.entities[0].op_horizon)
            u_ineq_list = Constraints(self.entities[0].op_horizon)
            v_k_list = Constraints(self.entities[0].op_horizon)
            if i != 0:
                length = self.variable_list[i].get_length()
                if length != 0:
                    node.model.bin_set = pyomo.RangeSet(0, length - 1)
                    node.model.x_bin = pyomo.Param(node.model.bin_set, entity.model.t, mutable=True, initialize=0)
                    node.model.u_bin = pyomo.Param(node.model.bin_set, entity.model.t, mutable=True, initialize=0)
                    x_k_binaries_list.append_param(node.model.x_bin, length)
                    u_binaries_list.append_param(node.model.u_bin, length)

                length = self.constraint_list[i]["equality_constr"].get_length()["time_indexed"]
                if length != 0:
                    node.model.eq_t_set = pyomo.RangeSet(0, length - 1)
                    node.model.u_eq_t = pyomo.Param(node.model.eq_t_set, entity.model.t, mutable=True, initialize=0)
                    u_eq_list.append_param(node.model.u_eq_t, length)

                length = self.constraint_list[i]["equality_constr"].get_length()["none_indexed"]
                if length != 0:
                    node.model.eq_n_set = pyomo.RangeSet(0, length - 1)
                    node.model.u_eq_n = pyomo.Param(node.model.eq_n_set, mutable=True, initialize=0)
                    u_eq_list.append_param(node.model.u_eq_n, length, None)

            x_k_param = np.append(x_k_param, x_k_binaries_list)
            u_var_param = np.append(u_var_param, u_binaries_list)
            node_dict = {}
            node_dict["equality_constr"] = u_eq_list
            u_constr_param = np.append(u_constr_param, node_dict)


        return x_k_param, u_var_param, u_constr_param

    def _add_objective(self):
        for i, node, entity in zip(range(len(self.entities)), self.nodes, self.entities):
            obj = node.model.beta * entity.get_objective()
            for j in range(entity.op_horizon):
                obj += node.model.rho / 2 * entity.model.p_el_vars[j] * entity.model.p_el_vars[j]
            # penalty term is expanded and constant is omitted
            if i == 0:
                # invert sign of p_el_schedule and p_el_vars (omitted for quadratic term)
                penalty = [(-node.model.last_p_el_schedules[t] - node.model.x_exch_[t] - node.model.u_exch[t])
                           for t in range(entity.op_horizon)]
                for j in range(entity.op_horizon):
                    obj += node.model.rho * penalty[j] * entity.model.p_el_vars[j]
            else:
                penalty = [(-node.model.last_p_el_schedules[t] + node.model.x_exch_[t] + node.model.u_exch[t])
                           for t in range(entity.op_horizon)]
                for j in range(entity.op_horizon):
                    obj += node.model.rho * penalty[j] * entity.model.p_el_vars[j]
                # Empirically the following term worsens the convergence of the algorithm in the unconstrained mode
                # although it belongs to the mathematical formulation of the algorithm. Therefore, it is only used in
                # constrained mode
                # Todo: Double check what happens to unconstrained variant
                """
                for t in range(entity.op_horizon):
                    a = self.variable_list[i].get_list(t)
                    b = self.x_k[i].get_list(t)
                    c = self.u_var[i].get_list(t)
                    expr = sum((x - x_k + u_k) ** 2 for x, x_k, u_k in zip(a, b, c))
                    obj += node.model.rho / 2 * expr
                """

                if self.x_update_mode == 'unconstrained':
                    # constraint entries of the augmented term
                    for t in range(entity.op_horizon + 1):
                        a = self.constraint_list[i]["equality_constr"].get_list(t)
                        b = self.u_constr[i]["equality_constr"].get_list(t)
                        expr = sum((c + p) ** 2 for c, p in zip(a, b))
                        obj += node.model.rho / 2 * expr

            node.model.o = pyomo.Objective(expr=obj)
        return

    # returns the average of the primal residual of subsystem i
    def primal_residual(self):
        primal_list = []
        for i, node, entity in zip(range(len(self.entities)), self.nodes, self.entities):
            r_i_t = np.empty(shape=0)
            for t in range(self.op_horizon + 1):
                r_i_t = np.append(r_i_t, self.constraint_list[i]["equality_constr"].get_list_values(t))
            primal_list.append(r_i_t)
        return primal_list

    # returns the temporal average of the dual residual of subsystem i
    def dual_residual(self, i, node):
        dual = 0
        for t in range(self.op_horizon):
            s_i_t = np.empty(shape=0)
            entry = self.variable_list[i].get_list_values(t) - self.x_k[i].get_list_values(t)
            s_i_t = np.append(s_i_t, entry)
            dual += np.linalg.norm(s_i_t)
        dual = node.model.rho.value * dual / self.op_horizon
        return dual

    # Implementation of the projection function Pi
    def _pi(self):
        for i, node, entity in zip(range(len(self.entities)), self.nodes, self.entities):
            for t in range(entity.op_horizon):
                x_old = self.variable_list[i].get_list_values(t)
                x_new = x_old + self.u_var[i].get_list_values(t)
                # round the binary variables on 0 or 1
                if self.mode == "integer":
                    for j in range(len(x_new)):
                        if abs(x_new[j]) >= 0.5:
                            x_new[j] = 1
                        else:
                            x_new[j] = 0
                self.variable_list[i].set_list_values(x_new, t)
        return

    def _presolve(self, full_update, beta, robustness, debug):
        # residuals for the exchange problem
        results, params = super()._presolve(full_update, beta, robustness, debug)
        results["r_norms"] = []
        results["s_norms"] = []
        for i, node, entity in zip(range(len(self.nodes)), self.nodes, self.entities):
            node.model.beta = self._get_beta(params, entity)
            if full_update:
                node.full_update(robustness)
            # residuals for all subsystems
            name = "r_i_norms_" + str(i)
            results[name] = []
            name = "s_i_norms_" + str(i)
            results[name] = []
            # average residuals
            results["r_sub_ave"] = []
            results["s_sub_ave"] = []
            # objective values
            results["obj_value"] = np.empty(shape=0)
        return results, params

    # returns the objective value
    def _get_objective(self):
        obj_value = 0
        for i, node, entity in zip(range(len(self.nodes)), self.nodes, self.entities):
            obj_value += pyomo.value(entity.get_objective())
        return obj_value

    # Function that checks if the stopping criteria is reached
    def _is_last_iteration(self, results, params, debug):
        if super(UnconstrainedLight, self)._is_last_iteration(results, params, debug):
            return True
        primal = 0
        dual = 0
        for i, node, entity in zip(range(len(self.nodes)), self.nodes, self.entities):
            name = "r_i_norms_" + str(i)
            primal += results[name][-1]
            name = "s_i_norms_" + str(i)
            dual += results[name][-1]
        # average of all dual updates
        primal = primal / (len(self.nodes) - 1)
        dual = dual / (len(self.nodes) - 1)
        results["r_sub_ave"].append(primal)
        results["s_sub_ave"].append(dual)

        print("r_exch", results["r_norms"][-1])
        print("s_exch", results["s_norms"][-1])
        print("r_sub", primal)
        print("s_sub", dual)


        # Reaching the stopping criteria
        if results["r_norms"][-1] <= self.eps_primal and results["s_norms"][-1] <= self.eps_dual\
                and primal <= self.eps_primal_i and dual <= self.eps_dual_i:
            print("reached")
            return True

        return

    def _iteration(self, results, params, debug):
        super(UnconstrainedLight, self)._iteration(results, params, debug)
        op_horizon = self.entities[0].op_horizon
        self.counter += 1
        print("Iteration", self.counter)

        # fill parameters if not already present
        if "p_el" not in params:
            params["p_el"] = np.zeros((len(self.entities), op_horizon))
        if "x_" not in params:
            params["x_"] = np.zeros(op_horizon)
        if "u" not in params:
            params["u"] = np.zeros(op_horizon)
        u = params["u"]

        # -----------------
        # 1) optimize all entities
        # -----------------
        to_solve_nodes = []
        variables = []
        for i, node, entity in zip(range(len(self.nodes)), self.nodes, self.entities):
            # for each subsystem, dual variables for the constraints have to be created
            dual = "u_bin_" + str(i)
            if dual not in params:
                params[dual] = np.zeros((op_horizon, self.variable_list[i].get_length()))
            # for each subsystem the solution of the descision variables have to be stored for the next iteration
            x_descision = "x_bin_" + str(i)
            if x_descision not in params:
                params[x_descision] = np.zeros((op_horizon, self.variable_list[i].get_length()))

            # Dual variables for the constraints
            if self.x_update_mode == 'unconstrained':
                dual = "u_eq_constr_" + str(i)
                if dual not in params:
                    params[dual] = self.u_constr[i]["equality_constr"].get_initial_list()
                dual = "u_ineq_constr_" + str(i)

            if not isinstance(
                    entity,
                    (CityDistrict, Building, Photovoltaic, WindEnergyConverter)
            ):
                continue
            # set all parameters
            for t in range(op_horizon):
                node.model.last_p_el_schedules[t] = params["p_el"][i][t]
                node.model.x_exch_[t] = params["x_"][t]
                node.model.u_exch[t] = params["u"][t]
                self.u_var[i].set_list_values(params["u_bin_" + str(i)][t, :], t)
                self.x_k[i].set_list_values(params["x_bin_" + str(i)][t, :], t)
                variables.append(self.variable_list[i].get_list(t))

            if self.x_update_mode == 'unconstrained':
                for t in range(op_horizon + 1):
                    self.u_constr[i]["equality_constr"].set_list_values(params["u_eq_constr_" + str(i)][t], t)

            # set objective of the node
            node.obj_update()
            to_solve_nodes.append(node)
            variables.append([entity.model.p_el_vars[t] for t in range(op_horizon)])
        self._solve_nodes(results, params, to_solve_nodes, variables=variables, debug=debug)
        # --------------------------
        # 2) incentive signal update
        # --------------------------

        # update all dual variables of subsystems
        for i, node, entity in zip(range(len(self.nodes)), self.nodes, self.entities):
            for t in range(op_horizon):
                x_bin = self.variable_list[i].get_list_values(t)
                x_k_bin = self.x_k[i].get_list_values(t)
                u_k_bin = self.u_var[i].get_list_values(t)
                params["u_bin_" + str(i)][t, :] = u_k_bin + x_bin - x_k_bin

            if self.x_update_mode == 'unconstrained':
                # constraint duals have the none index as an additional time index
                for t in range(op_horizon + 1):
                    eq_constr = self.constraint_list[i]["equality_constr"].get_list_values(t)
                    params["u_eq_constr_" + str(i)][t] = self.u_constr[i]["equality_constr"].get_list_values(t) \
                        + eq_constr

        primal_list = self.primal_residual()

        # Update all variables and round the binary variables
        self._pi()

        # Calculate all residual norms of subsystems. Norm the residuals by T
        for i, node, entity in zip(range(len(self.nodes)), self.nodes, self.entities):
            primal = np.linalg.norm(primal_list[i])
            primal = primal / self.op_horizon
            results["r_i_norms_" + str(i)].append(primal)
            results["s_i_norms_" + str(i)].append(self.dual_residual(i, node))

        # exchange dual update
        p_el_schedules = np.array([extract_pyomo_values(entity.model.p_el_vars, float) for entity in self.entities])
        x_ = (-p_el_schedules[0] + sum(p_el_schedules[1:])) / len(self.entities)
        u += x_

        # ------------------------------------------
        # 3) Calculate parameters for stopping criteria
        # ------------------------------------------
        results["r_norms"].append(np.linalg.norm(x_))

        s = np.zeros_like(p_el_schedules)
        for i, node, entity in zip(range(len(self.nodes)), self.nodes, self.entities):
            if i == 0:
                s[i] = - node.model.rho.value * (-p_el_schedules[0] + params["p_el"][0] + params["x_"] - x_)
            s[i] = - node.model.rho.value * (p_el_schedules[i] - params["p_el"][i] + params["x_"] - x_)
        results["s_norms"].append(np.linalg.norm(s.flatten()))

        # store the objective value
        obj = self._get_objective()
        results["obj_value"] = np.append(results["obj_value"], obj)

        # save parameters for another iteration
        params["p_el"] = p_el_schedules
        params["x_"] = x_
        params["u"] = u
        for i in range(len(self.nodes)):
            params["x_bin_" + str(i)] = self.variable_list[i].get_initial_data()

        results["schedule"] = p_el_schedules[0]
        return


class Variables:
    """Implementation of the data structure for the binary decision variables and their dual Variables.
       For each time step, a numpy array is created in which the time indexed pyomo variables or parameters are stored.
       The complete data structure is created in the class Exchange MIQP ADMM method _get_variables() and is
       shown graphically in 'data_structure_exchange_miqp_admm.pdf'.

    Parameters
    ----------
    op_horizon : int
        Number of simulation time steps
    obj : Pyomo Var or Pyomo Param
        Pyomo object is to append on a time indexed numpy array
    length: int
        Length of a numpy array containing pyomo Variables or Parameters
    time_steo: int
        Needed to call a Variables array of a specific time step
    new_values: numpy array
        Array of values that should replace the actual values of a Variables array of a specific time step
    """
    # In the constructor, an empty numpy array is created for the class Variable object for each time step
    def __init__(self, op_horizon):
        self.x = {}
        self.op_horizon = op_horizon
        for i in range(op_horizon):
            name = "t_" + str(i)
            self.x[name] = np.empty(shape=0)

    # Function to append a pyomo variable on the numpy array for each time step.
    # All variables are set to Reals since the Exchange MIQP ADMM algorithm rounds the binary variables
    def append(self, obj):
        for time_step in range(self.op_horizon):
            name = "t_" + str(time_step)
            obj[time_step].domain = pmo.Reals
            obj[time_step].value = 0
            self.x[name] = np.append(self.x[name], obj[time_step])
        return

    # Function to append a pyomo parameter on the numpy array for each time step
    def append_param(self, obj, length):
        for j in range(length):
            for time_step in range(self.op_horizon):
                name = "t_" + str(time_step)
                self.x[name] = np.append(self.x[name], obj[j, time_step])
        return

    # Function to get the length of a variable or parameter array (arrays have same length for all time step)
    def get_length(self):
        counter = 0
        for j in self.x["t_0"]:
            counter += 1
        return counter

    # Function to get a variable or parameter array for a certain time step (lists are implemented as numpy arrays)
    def get_list(self, time_step):
        return self.x["t_" + str(time_step)]

    # Function to get a list of the values of a variable or parameter array
    def get_list_values(self, time_step):
        data = np.empty(shape=0)
        for x in self.x["t_" + str(time_step)]:
            data = np.append(data, x.value)
        return data

    # Function to set the values of a variable or parameter array
    def set_list_values(self, new_values, time_step):
        if self.get_length() != np.size(new_values):
            raise Exception("Error: Arrays must have the same length!")
        for j, x in zip(range(self.get_length()), self.x["t_" + str(time_step)]):
            x.value = new_values[j]
        return

    # Function to get a matrix of the values of all variables or parameters of a class object for each time step
    def get_initial_data(self):
        columns = self.get_length()
        rows = self.op_horizon
        data = np.zeros((rows, columns))
        for t in range(self.op_horizon):
            data[t] = self.get_list_values(t)
        return data

    def remove_exchange_var(self):
        for time_step in range(self.op_horizon):
            self.x["t_" + str(time_step)] = np.delete(self.get_list(time_step), 0)
        return


# class creates a data structure for the pyomo constraints and parameters that belong to constraints
# Compared to variables, the constraints have an additional time index "NONE"
class Constraints:
    """Implementation of the data structure for the constraints and their dual Variables.
       For each time step a numpy array is created in which the time indexed pyomo expressions or Parameters are stored.
       The complete data structure is created in the class Exchange MIQP ADMM method _get_constraints() and is
       shown graphically in 'data_structure_exchange_miqp_admm.pdf'.

    Parameters
    ----------
    op_horizon : int
        Number of simulation time steps
    obj : Pyomo Expression or Pyomo Parameter
        Pyomo object is to append on a time indexed numpy array
    length: int
        Length of a numpy array containing pyomo Expressions or Parameters
    index: int or None
        Needed to call a Constraints array of a specific time step
    new_values: numpy array
        Array of values that should replace the actual values of a Constraints array of a specific time step
    """
    # Constructor: An empty numpy array is created for the Constraint object for each time step and the None index
    # From now on: time index means the time steps plus the None index (t_1, t_2, ... , t_n, None)
    def __init__(self, op_horizon):
        self.x = {}
        self.op_horizon = op_horizon
        for i in range(op_horizon + 1):
            name = "index_" + str(i)
            self.x[name] = np.empty(shape=0)

    # Function to append a pyomo constraint on the numpy array for a certain time index
    def append(self, obj, index):
        if index is None:
            name = "index_" + str(self.op_horizon)
        else:
            name = "index_" + str(index)
        self.x[name] = np.append(self.x[name], obj)

    # Function to append a pyomo parameter on the numpy array for a certain time index
    def append_param(self, obj, length, index=0):
        if index is None:
            name = "index_" + str(self.op_horizon)
            for j in range(length):
                self.x[name] = np.append(self.x[name], obj[j])
        else:
            for j in range(length):
                for t in range(self.op_horizon):
                    name = "index_" + str(t)
                    self.x[name] = np.append(self.x[name], obj[j, t])
        return

    # Function to get the constraint or parameter array for a certain time index
    def get_list(self, index):
        return self.x["index_" + str(index)]

    # Function to get a list of the values of a constraint or parameter array for a certain time index
    def get_list_values(self, index):
        data = np.empty(shape=0)
        for x in self.x["index_" + str(index)]:
            data = np.append(data, pyo_core.value(x))
        return data

    # Function to set the values of a constraint or parameter array for a certain time index
    def set_list_values(self, new_values, time_step):
        if time_step != self.op_horizon:
            for j, x in zip(range(self.get_length()["time_indexed"]), self.x["index_" + str(time_step)]):
                x.value = new_values[j]
        else:
            for j, x in zip(range(self.get_length()["none_indexed"]), self.x["index_" + str(time_step)]):
                x.value = new_values[j]
        return

    # Function returns a dictionary of the length of ay constraint or parameter for a time step and the None index
    # None index has a different length than the time step indices but those have the same length for all time steps
    def get_length(self):
        counter_1 = 0
        counter_2 = 0
        for x in self.x["index_0"]:
            counter_1 += 1
        for x in self.x["index_" + str(self.op_horizon)]:
            counter_2 += 1
        dict = {"time_indexed": counter_1, "none_indexed": counter_2}
        return dict

    # Function to get a matrix of the values of all constraints or parameters of a class object for each time index
    def get_initial_list(self):
        initial_list = []
        for t in range(self.op_horizon + 1):
            initial_list.append(self.get_list_values(t))
        return initial_list

    # Function to check if the inequality constraints are violated
    # If a constraint is not violated, its value is set to zero
    def violation(self, t):
        r = self.get_list_values(t)
        for j in range(len(r)):
            if r[j] > 0:
                r[j] = 0
        return r

    # Function to execute the update of the dual parameter v_k
    def v_k_update(self, u_k, t):
        v_k_plus_1 = self.get_list_values(t) + u_k
        for j in range(len(v_k_plus_1)):
            if v_k_plus_1[j] <= 0:
                v_k_plus_1[j] = 0
        return v_k_plus_1