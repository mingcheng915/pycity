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
import matplotlib.pyplot as plt

from pycity_scheduling.classes import *
from pycity_scheduling.util import extract_pyomo_values
from pycity_scheduling.algorithms.algorithm import IterationAlgorithm, DistributedAlgorithm, SolverNode
from pycity_scheduling.solvers import DEFAULT_SOLVER, DEFAULT_SOLVER_OPTIONS


class ExchangeMIQPADMMMPI(IterationAlgorithm, DistributedAlgorithm):
    """
    Implementation of the distributed ADMM algorithm.
    This class implements the distributed Exchange MIQP ADMM as described in [1].

    Parameters
    ----------
    city_district : CityDistrict
    mpi_interface : MPIInterface
        MPIInterface to use for solving the subproblems in parallel.
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
        Step size for the ADMM algorithm.
    max_iterations : int, optional
        Maximum number of ADMM iterations.
    robustness : tuple, optional
        Tuple of two floats. First entry defines how many time steps are
        protected from deviations. Second entry defines the magnitude of
        deviations which are considered.

    References
    ----------
    [1] "A simple effective heuristic for embedded mixed-integer quadratic programming"
    by Reza Takapoui, Nicholas Moehle, Stephen Boyd, and Alberto Bemporad
    Online: https://web.stanford.edu/~boyd/papers/pdf/miqp_admm.pdf (accessed on 2022/07/29)
    """
    def __init__(self, city_district, mpi_interface, solver=DEFAULT_SOLVER, solver_options=DEFAULT_SOLVER_OPTIONS,
                 mode="integer", x_update_mode='unconstrained', eps_primal=0.1, eps_dual=0.1, eps_primal_i=0.1,
                 eps_dual_i=0.1, rho=2, max_iterations=10000, robustness=None, fix=True):
        super(ExchangeMIQPADMMMPI, self).__init__(city_district, solver, solver_options, mode)

        self.mpi_interface = mpi_interface
        self.mode = mode
        self.x_update_mode = x_update_mode
        self.eps_primal = eps_primal
        self.eps_dual = eps_dual
        self.eps_primal_i = eps_primal_i
        self.eps_dual_i = eps_dual_i
        self.rho = rho
        self.max_iterations = max_iterations
        self.op_horizon = self.city_district.op_horizon
        self.fix = fix
        self.fixed_iter = 0
        self.counter = 0

        # Only consider entities of type CityDistrict, Building, Photovoltaic, WindEnergyConverter
        self._entities = [entity for entity in self.entities if
                          isinstance(entity, (CityDistrict, Building, Photovoltaic, WindEnergyConverter))]

        # Create a solver node for each entity
        self.nodes = [SolverNode(solver, solver_options, [entity], mode, robustness=robustness)
                      for entity in self._entities]

        # Determine which MPI processes is responsible for which node(s):
        if self.mpi_interface.get_size() > len(self._entities):
            mpi_process_range = np.array([i for i in range(len(self._entities))])
        elif self.mpi_interface.get_size() < len(self._entities):
            if self.mpi_interface.get_size() == 1:
                mpi_process_range = np.array([0 for i in range(len(self._entities))])
            else:
                a, b = divmod(len(self._entities) - 1, self.mpi_interface.get_size() - 1)
                mpi_process_range = np.repeat(np.array([i for i in range(1, self.mpi_interface.get_size())]), a)
                for i in range(b):
                    mpi_process_range = np.append(mpi_process_range, i + 1)
                mpi_process_range = np.concatenate([[0], mpi_process_range])
        else:
            mpi_process_range = np.array([i for i in range(len(self._entities))])
        self.mpi_process_range = np.sort(mpi_process_range)

        # Load data from the model
        self.constraint_list = self._get_constraints()
        self.variable_list = self._get_variables()

        # Create pyomo parameters for each entity
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            node.model.beta = pyomo.Param(mutable=True, initialize=1)
            node.model.x_exch_ = pyomo.Param(entity.model.t, mutable=True, initialize=0)
            node.model.u_exch = pyomo.Param(entity.model.t, mutable=True, initialize=0)
            node.model.last_p_el_schedules = pyomo.Param(entity.model.t, mutable=True, initialize=0)

        # Lists that contain the dual parameters of all subsystems
        self.x_k, self.u_var, self.u_constr, self.v_k = self._set_parameters()
        self._add_objective()

    # Function that gets the binary state variables of each subsystem from the model
    # It also changes the domain of the binary variables to real variables, since they are rounded on binaries later
    # The data structure in which the variables are stored is shown graphically in "data_structure.pdf"
    def _get_variables(self):
        variable_list = np.empty(shape=0)
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            binaries_list = Variables(self.op_horizon)
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
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            node_dict = {}
            equality_constr_list = Constraints(self.op_horizon)
            inequality_constr_list = Constraints(self.op_horizon)
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

                            # if the constraint is not an equality constraint it has to be an inequality constraint
                            # the next three checks are about to write that constraint in the form Cx-d >= 0
                            elif pyomo.value(constraint[index].upper) is None:
                                expr = constraint[index].body - constraint[index].lower
                                inequality_constr_list.append(expr, index)

                            elif pyomo.value(constraint[index].lower) is None:
                                expr = -constraint[index].body + constraint[index].upper
                                inequality_constr_list.append(expr, index)

                            else:
                                expr = -constraint[index].body + constraint[index].upper
                                inequality_constr_list.append(expr, index)
                                expr = constraint[index].body - constraint[index].lower
                                inequality_constr_list.append(expr, index)

            node_dict["equality_constr"] = equality_constr_list
            node_dict["inequality_constr"] = inequality_constr_list
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
        # dual parameter for the ADMM method for inequality constrained optimization
        v_k_param = np.empty(shape=0)
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            node_dict = {}
            x_k_binaries_list = Variables(self.op_horizon)
            u_binaries_list = Variables(self.op_horizon)
            u_eq_list = Constraints(self.op_horizon)
            u_ineq_list = Constraints(self.op_horizon)
            v_k_list = Constraints(self.op_horizon)
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

                length = self.constraint_list[i]["inequality_constr"].get_length()["time_indexed"]
                if length != 0:
                    node.model.ineq_t_set = pyomo.RangeSet(0, length - 1)
                    node.model.u_ineq_t = pyomo.Param(node.model.ineq_t_set, entity.model.t, mutable=True, initialize=0)
                    node.model.v_k_t = pyomo.Param(node.model.ineq_t_set, entity.model.t, mutable=True, initialize=0)
                    u_ineq_list.append_param(node.model.u_ineq_t, length)
                    v_k_list.append_param(node.model.v_k_t, length)

                length = self.constraint_list[i]["inequality_constr"].get_length()["none_indexed"]
                if length != 0:
                    node.model.ineq_n_set = pyomo.RangeSet(0, length - 1)
                    node.model.u_ineq_n = pyomo.Param(node.model.ineq_n_set, mutable=True, initialize=0)
                    node.model.v_k_n = pyomo.Param(node.model.ineq_n_set, mutable=True, initialize=0)
                    u_ineq_list.append_param(node.model.u_ineq_n, length, None)
                    v_k_list.append_param(node.model.v_k_n, length, None)

            x_k_param = np.append(x_k_param, x_k_binaries_list)
            u_var_param = np.append(u_var_param, u_binaries_list)
            node_dict = {}
            node_dict["equality_constr"] = u_eq_list
            node_dict["inequality_constr"] = u_ineq_list
            u_constr_param = np.append(u_constr_param, node_dict)
            v_k_param = np.append(v_k_param, v_k_list)

        return x_k_param, u_var_param, u_constr_param, v_k_param

    def _add_objective(self):
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            obj = node.model.beta * entity.get_objective()
            for j in range(self.op_horizon):
                obj += self.rho / 2 * entity.model.p_el_vars[j] * entity.model.p_el_vars[j]
            # penalty term is expanded and constant is omitted
            if i == 0:
                # invert sign of p_el_schedule and p_el_vars (omitted for quadratic term)
                penalty = [(-node.model.last_p_el_schedules[t] - node.model.x_exch_[t] - node.model.u_exch[t])
                           for t in range(self.op_horizon)]
                for j in range(self.op_horizon):
                    obj += self.rho * penalty[j] * entity.model.p_el_vars[j]
            else:
                penalty = [(-node.model.last_p_el_schedules[t] + node.model.x_exch_[t] + node.model.u_exch[t])
                           for t in range(self.op_horizon)]
                for j in range(self.op_horizon):
                    obj += self.rho * penalty[j] * entity.model.p_el_vars[j]
                # Empirically the following term worsens the convergence of the algorithm in the unconstrained mode
                # although it belongs to the mathematical formulation of the algorithm. Therefore, it is only used in
                # constrained mode
                # Todo: Double check what happens to unconstrained variant
                if self.x_update_mode == 'constrained':
                    for t in range(self.op_horizon):
                        a = self.variable_list[i].get_list(t)
                        b = self.x_k[i].get_list(t)
                        c = self.u_var[i].get_list(t)
                        expr = sum((x - x_k + u_k) ** 2 for x, x_k, u_k in zip(a, b, c))
                        obj += self.rho / 2 * expr

                if self.x_update_mode == 'unconstrained':
                    # constraint entries of the augmented term
                    for t in range(self.op_horizon + 1):
                        a = self.constraint_list[i]["equality_constr"].get_list(t)
                        b = self.u_constr[i]["equality_constr"].get_list(t)
                        expr = sum((c + p) ** 2 for c, p in zip(a, b))
                        obj += self.rho / 2 * expr
                        a = self.constraint_list[i]["inequality_constr"].get_list(t)
                        b = self.u_constr[i]["inequality_constr"].get_list(t)
                        c = self.v_k[i].get_list(t)
                        expr = sum((constraint + u_k - v_k) ** 2 for constraint, u_k, v_k in zip(a, b, c))
                        obj += self.rho / 2 * expr
            node.model.o = pyomo.Objective(expr=obj)
        return

    # returns the average of the primal residual of subsystem i
    def primal_residual(self):
        primal_list = []
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                r_i_t = np.empty(shape=0)
                for t in range(self.op_horizon + 1):
                    r_i_t = np.append(r_i_t, self.constraint_list[i]["equality_constr"].get_list_values(t))
                    r_i_t = np.append(r_i_t, self.constraint_list[i]["inequality_constr"].violation(t))
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
        dual = self.rho * dual / self.op_horizon
        return dual

    # Implementation of the projection function Pi
    def _pi(self):
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                for t in range(self.op_horizon):
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
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
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

    def _postsolve(self, results, params, debug):
        if self.mpi_interface.get_size() > 1:
            # Update all models across all MPI instances:
            pyomo_var_values = dict()
            asset_updates = np.empty(len(self._entities), dtype=np.object)
            for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
                if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                    for asset in entity.get_all_entities():
                        for v in asset.model.component_data_objects(ctype=pyomo.Var, descend_into=True):
                            pyomo_var_values[str(v)] = pyomo.value(v)
                    asset_updates[i] = pyomo_var_values

            if self.mpi_interface.get_rank() == 0:
                buffer = np.array([bytearray(10**7) for i in range(1, len(self._entities))])
                for i in range(1, len(self._entities)):
                    req = self.mpi_interface.get_comm().irecv(
                        buffer[i-1],
                        source=self.mpi_process_range[i],
                        tag=(int(results["iterations"][-1])+1) * len(self._entities) + i
                    )
                    asset_updates[i] = req.wait()
            else:
                for i in range(1, len(self._entities)):
                    if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                        req = self.mpi_interface.get_comm().isend(
                            asset_updates[i],
                            dest=0,
                            tag=(int(results["iterations"][-1])+1) * len(self._entities) + i
                        )
                        req.wait()
                asset_updates = np.empty(len(self._entities), dtype=np.object)

            """
            for i in range(len(self._entities)):
                if self.mpi_interface.get_rank() == 0:
                    asset_update = asset_updates[i]
                else:
                    asset_update = np.empty(1, dtype=np.object)
                asset_update = self.mpi_interface.get_comm().bcast(asset_update, root=0)
                if self.mpi_interface.get_rank() > 0:
                    asset_updates[i] = asset_update
            """

            if self.mpi_interface.get_rank() == 0:
                for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
                    for asset in entity.get_all_entities():
                        pyomo_var_values_map = pyomo.ComponentMap()
                        for v in asset.model.component_data_objects(ctype=pyomo.Var, descend_into=True):
                            if str(v) in asset_updates[i]:
                                pyomo_var_values_map[v] = pyomo.value(asset_updates[i][str(v)])
                        for var in pyomo_var_values_map:
                            var.set_value(pyomo_var_values_map[var])
                        asset.update_schedule()
        else:
            super()._postsolve(results, params, debug)
        return

    # returns the objective value
    def _get_objective(self):
        obj_value_list = np.zeros(len(self._entities), dtype=np.float64)
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                obj_value_list[i] = pyomo.value(entity.get_objective())
        if self.mpi_interface.mpi_size > 1:
            tmp = np.zeros(len(self._entities), dtype=np.float64)
            self.mpi_interface.get_comm().Allreduce(obj_value_list, tmp, self.mpi_interface.mpi.SUM)
            obj_value_list = tmp
        obj_value = np.sum(obj_value_list)
        return obj_value

    # Function that checks if the stopping criteria is reached
    def _is_last_iteration(self, results, params, debug):
        if super(ExchangeMIQPADMMMPI, self)._is_last_iteration(results, params, debug):
            return True
        primal = np.zeros(1, dtype=np.float64)
        dual = np.zeros(1, dtype=np.float64)
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            if self.mpi_interface.mpi_size > 1:
                if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                    primal[0] += results["r_i_norms_" + str(i)][-1]
                    dual[0] += results["s_i_norms_" + str(i)][-1]
            else:
                primal[0] += results["r_i_norms_" + str(i)][-1]
                dual[0] += results["s_i_norms_" + str(i)][-1]
        if self.mpi_interface.mpi_size > 1:
            total_primal = np.empty(1, dtype=np.float64)
            total_dual = np.empty(1, dtype=np.float64)
            self.mpi_interface.get_comm().Allreduce(primal, total_primal, self.mpi_interface.mpi.SUM)
            self.mpi_interface.get_comm().Allreduce(dual, total_dual, self.mpi_interface.mpi.SUM)
        else:
            total_primal = primal
            total_dual = dual

        # average of all dual updates
        primal = total_primal[0] / (len(self._entities) - 1)
        dual = total_dual[0] / (len(self._entities) - 1)
        results["r_sub_ave"].append(primal)
        results["s_sub_ave"].append(dual)

        # Reaching the stopping criteria
        if results["r_norms"][-1] <= self.eps_primal and results["s_norms"][-1] <= self.eps_dual\
                and primal <= self.eps_primal_i and dual <= self.eps_dual_i:
            return True

        return

    def debug(self, node):
        hp_c = 0
        eh_c = 0
        chp_c = 0
        bat_c = 0
        for en in node.get_entities():
            if isinstance(en, HeatPump) and hp_c == 0:
                en.model.p_th_heat_state_vars[0].pprint()
                hp_c += 1
            if isinstance(en, ElectricalHeater) and eh_c == 0:
                en.model.p_th_heat_state_vars[0].pprint()
                eh_c += 1
            if isinstance(en, CombinedHeatPower) and chp_c == 0:
                en.model.p_th_heat_state_vars[0].pprint()
                chp_c += 1
            if isinstance(en, Battery) and bat_c == 0:
                en.model.p_state_vars[0].pprint()
                bat_c += 1

    def _get_constraints_for_count(self):
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
                        equality_list.append([constraint, expr])

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

    def _penultimate_iteration(self, results):
        print("------------------Penultimate Iteration-------------------")
        print("Objective Value: ", results["obj_value"][-1])
        plt.figure(2)
        plt.plot(results["obj_value"])
        plt.ylabel("Objetive value")
        plt.title("Objetive Value")
        plt.savefig('penultimate_obj_value.png')

        equality_list, inequality_list = self._get_constraints_for_count()
        equality_counter = 0
        inequality_counter = 0
        # check if constraints are violated. Allow a numerical tolerance of 1e-3 which equals 1 Watt in the application
        for i in range(len(equality_list)):
            if abs(0-pyomo.value(equality_list[i][1])) > 1e-2:
                print("Equality violation: ", equality_list[i][0])
                print(equality_list[i][1], "   ", pyomo.value(equality_list[i][1]))
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

        return inequality_counter + equality_counter


    def _iteration(self, results, params, debug):
        super(ExchangeMIQPADMMMPI, self)._iteration(results, params, debug)
        self.counter += 1
        print("Iteration", self.counter)

        # fill parameters if not already present
        if "p_el" not in params:
            params["p_el"] = np.zeros((len(self._entities), self.op_horizon))
        if "x_" not in params:
            params["x_"] = np.zeros(self.op_horizon)
        if "u" not in params:
            params["u"] = np.zeros(self.op_horizon)
        u = params["u"]

        # -----------------
        # 1) optimize all entities
        # -----------------
        to_solve_nodes = []
        variables = []
        p_el_schedules = np.empty((len(self._entities), self.op_horizon))
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                # for each subsystem, dual variables for the constraints have to be created
                dual = "u_bin_" + str(i)
                if dual not in params:
                    params[dual] = np.zeros((self.op_horizon, self.variable_list[i].get_length()))
                # for each subsystem the solution of the descision variables have to be stored for the next iteration
                x_descision = "x_bin_" + str(i)
                if x_descision not in params:
                    params[x_descision] = np.zeros((self.op_horizon, self.variable_list[i].get_length()))

                # Dual variables for the constraints
                if self.x_update_mode == 'unconstrained':
                    dual = "v_k_" + str(i)
                    if dual not in params:
                        params[dual] = self.v_k[i].get_initial_list()
                    dual = "u_eq_constr_" + str(i)
                    if dual not in params:
                        params[dual] = self.u_constr[i]["equality_constr"].get_initial_list()
                    dual = "u_ineq_constr_" + str(i)
                    if dual not in params:
                        params[dual] = self.u_constr[i]["inequality_constr"].get_initial_list()

                # set all parameters
                for t in range(self.op_horizon):
                    node.model.last_p_el_schedules[t] = params["p_el"][i][t]
                    node.model.x_exch_[t] = params["x_"][t]
                    node.model.u_exch[t] = params["u"][t]
                    self.u_var[i].set_list_values(params["u_bin_" + str(i)][t, :], t)
                    self.x_k[i].set_list_values(params["x_bin_" + str(i)][t, :], t)
                    variables.append(self.variable_list[i].get_list(t))

                if self.x_update_mode == 'unconstrained':
                    for t in range(self.op_horizon + 1):
                        self.u_constr[i]["equality_constr"].set_list_values(params["u_eq_constr_" + str(i)][t], t)
                        self.u_constr[i]["inequality_constr"].set_list_values(params["u_ineq_constr_" + str(i)][t], t)
                        self.v_k[i].set_list_values(params["v_k_" + str(i)][t], t)

                # set objective of the node
                node.obj_update()
                to_solve_nodes.append(node)
                variables.append([entity.model.p_el_vars[t] for t in range(self.op_horizon)])
                self._solve_nodes(results, params, to_solve_nodes, variables=variables, debug=debug)

        # --------------------------
        # 2) incentive signal update
        # --------------------------
        # update all dual variables of subsystems
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                for t in range(self.op_horizon):
                    x_bin = self.variable_list[i].get_list_values(t)
                    x_k_bin = self.x_k[i].get_list_values(t)
                    u_k_bin = self.u_var[i].get_list_values(t)
                    params["u_bin_" + str(i)][t, :] = u_k_bin + x_bin - x_k_bin

                if self.x_update_mode == 'unconstrained':
                    # constraint duals have the none index as an additional time index
                    for t in range(self.op_horizon + 1):
                        eq_constr = self.constraint_list[i]["equality_constr"].get_list_values(t)
                        ineq_constr = self.constraint_list[i]["inequality_constr"].get_list_values(t)
                        u_ineq_k = self.u_constr[i]["inequality_constr"].get_list_values(t)
                        v_k_plus_1 = self.constraint_list[i]["inequality_constr"].v_k_update(u_ineq_k, t)
                        params["v_k_" + str(i)][t] = v_k_plus_1
                        params["u_eq_constr_" + str(i)][t] = self.u_constr[i]["equality_constr"].get_list_values(t)\
                            + eq_constr
                        params["u_ineq_constr_" + str(i)][t] = self.u_constr[i]["inequality_constr"].get_list_values(t)\
                            + ineq_constr - v_k_plus_1

        primal_list = self.primal_residual()

        # Update all variables and round the binary variables
        self._pi()

        if self.counter==self.max_iterations and self.fix==True:
            to_solve_nodes = []
            variables = []
            for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
                if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                    self.variable_list[i].fix_variables()
                    to_solve_nodes.append(node)
                    variables.append([entity.model.p_el_vars[t] for t in range(self.op_horizon)])
            self._solve_nodes(results, params, to_solve_nodes, variables=variables, debug=debug)
        # Calculate all residual norms of subsystems. Norm the residuals by T
        for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
            if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                primal = np.linalg.norm(primal_list[0])
                primal = primal / self.op_horizon
                results["r_i_norms_" + str(i)].append(primal)
                results["s_i_norms_" + str(i)].append(self.dual_residual(i, node))
        # exchange dual update
        if self.mpi_interface.get_rank() == 0:
            p_el_schedules[0] = np.array(extract_pyomo_values(self.city_district.model.p_el_vars, float),
                                         dtype=np.float64)
        for j in range(1, len(self._entities)):
            if self.mpi_interface.get_rank() == 0:
                if self.mpi_interface.get_size() > 1:
                    data = np.empty(self.op_horizon, dtype=np.float64)
                    req = self.mpi_interface.get_comm().Irecv(
                        data,
                        source=self.mpi_process_range[j],
                        tag=int(results["iterations"][-1]) * len(self._entities) + j
                    )
                    req.wait()
                    p_el_schedules[j] = np.array(data, dtype=np.float64)
                else:
                    p_el_schedules[j] = np.array(extract_pyomo_values(self._entities[j].model.p_el_vars, float),
                                                 dtype=np.float64)
            else:
                if self.mpi_interface.get_rank() == self.mpi_process_range[j]:
                    p_el_schedules[j] = np.array(extract_pyomo_values(self._entities[j].model.p_el_vars, float),
                                                 dtype=np.float64)
                    if self.mpi_interface.get_size() > 1:
                        req = self.mpi_interface.get_comm().Isend(
                            p_el_schedules[j],
                            dest=0,
                            tag=int(results["iterations"][-1]) * len(self._entities) + j
                        )
                        req.wait()
        if self.mpi_interface.get_rank() == 0:
            x_ = (-p_el_schedules[0] + sum(p_el_schedules[1:])) / len(self._entities)
            u += x_
        else:
            x_ = np.empty(self.op_horizon, dtype=np.float64)
            u = np.empty(self.op_horizon, dtype=np.float64)
            p_el_schedules = np.empty((len(self._entities), self.op_horizon), dtype=np.float64)
        if self.mpi_interface.get_size() > 1:
            self.mpi_interface.get_comm().Bcast(x_, root=0)
            self.mpi_interface.get_comm().Bcast(u, root=0)
            self.mpi_interface.get_comm().Bcast(p_el_schedules, root=0)
        # ------------------------------------------
        # 3) Calculate parameters for stopping criteria
        # ------------------------------------------
        if self.mpi_interface.get_rank() == 0:
            r_norm = np.array([np.linalg.norm(x_)], dtype=np.float64)
            s = np.zeros_like(p_el_schedules)
            for i, node, entity in zip(range(len(self._entities)), self.nodes, self._entities):
                if i == 0:
                    s[i] = - self.rho * (-p_el_schedules[0] + params["p_el"][0] + params["x_"] - x_)
                else:
                    s[i] = - self.rho * (p_el_schedules[i] - params["p_el"][i] + params["x_"] - x_)
            s_norm = np.array([np.linalg.norm(s.flatten())], dtype=np.float64)
        else:
            r_norm = np.empty(1, dtype=np.float64)
            s_norm = np.empty(1, dtype=np.float64)

        if self.mpi_interface.get_size() > 1:
            self.mpi_interface.get_comm().Bcast(r_norm, root=0)
            self.mpi_interface.get_comm().Bcast(s_norm, root=0)
        results["r_norms"].append(r_norm[0])
        results["s_norms"].append(s_norm[0])
        # store the objective value
        obj = self._get_objective()
        results["obj_value"] = np.append(results["obj_value"], obj)

        # save parameters for another iteration
        params["p_el"] = p_el_schedules
        params["x_"] = x_
        params["u"] = u

        for i in range(len(self._entities)):
            if self.mpi_interface.get_rank() == self.mpi_process_range[i]:
                params["x_bin_" + str(i)] = self.variable_list[i].get_initial_data()

        results["schedule"] = p_el_schedules[0]

        if self.mpi_interface.get_rank() == 0:
            if self.counter == self.max_iterations-1 or self.max_iterations== self.counter:
                self.fixed_iter = self._penultimate_iteration(results)
                print(self.fixed_iter)
        return


class Variables:
    """
    Implementation of the data structure for the binary decision variables and their dual Variables.
    For each time step, a numpy array is created in which the time indexed pyomo variables or parameters are stored.
    The complete data structure is created in the class Exchange MIQP ADMM method _get_variables() and is
    shown graphically in 'data_structure_exchange_miqp_admm.pdf'.

    Parameters
    ----------
    op_horizon : int
        Number of simulation time steps
    """
    # In the constructor, an empty numpy array is created for the class Variable object for each time step
    def __init__(self, op_horizon):
        self.x = {}
        self.op_horizon = op_horizon
        for i in range(self.op_horizon):
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

    def fix_variables(self):
        for time_step in range(self.op_horizon):
            for x in self.x["t_" + str(time_step)]:
                x.setlb(x.value)
                x.setub(x.value)
        return

    def print_variables(self):
        for time_step in range(self.op_horizon):
            for x in self.x["t_" + str(time_step)]:
                print(x)
                print(x.value)
        return


# class creates a data structure for the pyomo constraints and parameters that belong to constraints
# Compared to variables, the constraints have an additional time index "NONE"
class Constraints:
    """
    Implementation of the data structure for the constraints and their dual Variables.
    For each time step a numpy array is created in which the time indexed pyomo expressions or Parameters are stored.
    The complete data structure is created in the class Exchange MIQP ADMM method _get_constraints() and is
    shown graphically in 'data_structure_exchange_miqp_admm.pdf'.

    Parameters
    ----------
    op_horizon : int
        Number of simulation time steps
    """
    # Constructor: An empty numpy array is created for the Constraint object for each time step and the None index
    # From now on: time index means the time steps plus the None index (t_1, t_2, ... , t_n, None)
    def __init__(self, op_horizon):
        self.x = {}
        self.op_horizon = op_horizon
        for i in range(self.op_horizon + 1):
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
