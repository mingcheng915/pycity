import numpy as np
import gurobipy as gurobi

from pycity_scheduling.classes import *
from pycity_scheduling.exception import *
from pycity_scheduling.util import populate_models
from tempfile import TemporaryDirectory
import time


def get_obj_mod_lin(obj: gurobi.QuadExpr, cd_P_El_vars):
    horizon = len(cd_P_El_vars)
    lin_mod = np.zeros(horizon)
    if type(obj) == gurobi.QuadExpr:
        lin_expr = obj.getLinExpr()
    else:
        lin_expr = obj
    for i, v in enumerate(cd_P_El_vars):
        val = 0
        for j in range(lin_expr.size()):
            if v.VarName == lin_expr.getVar(j).VarName:
                val = lin_expr.getCoeff(j)
                break
        lin_mod[i] += val
    if type(obj) == gurobi.QuadExpr:
        for i, v in enumerate(cd_P_El_vars):
            val = 0
            for j in range(obj.size()):
                if v.VarName == obj.getVar1(j).VarName:
                    val = 2*obj.getCoeff(j)*obj.getVar1(j).X
                    break
            lin_mod[i] += val
    return lin_mod

def integer(city_district, models=None, rho=0.5, beta=1.0, max_iterations=10000, max_time=None, iteration_callback=None):
    """Perform Exchange ADMM on a city district.

    Do the scheduling of electrical energy in a city district using the
    Exchange ADMM algorithm, concerning different objectives for clients and
    the aggregator.

    Parameters
    ----------
    city_district : pycity_scheduling.classes.CityDistrict
    models : dict, optional
        Holds a `gurobi.Model` for each node and the aggregator.
    beta : float, optional #TODO use beta
        Tradeoff factor between system and customer objective. The customer
        objective is multiplied with beta.
    max_iterations : int, optional
        Maximum number of ADMM iterations.
    max_time : float, optional
        Maximum number of seconds to iterate.

    Returns
    -------
    int :
        Number of iterations."""
    assert 0 < rho <= 1
    op_horizon = city_district.op_horizon
    nodes = city_district.nodes
    n = 1 + len(nodes)

    iteration = 0
    P_El_Schedules = {}
    incentives = np.zeros(96)
    penalties = {}
    objs = {}
    last_choices = {}
    if models is None:
        models = populate_models(city_district, "admm")

    cd_choices = {}
    cd_power_constraints = []
    P_El_Schedules[0] = []
    total_P_el = [gurobi.LinExpr() for _ in range(op_horizon)]
    current_P_El_Schedule = {}
    current_P_El_Schedule[0] = np.zeros(op_horizon)
    for node_id, node in nodes.items():
        cd_choices[node_id] = []
        objs[node_id] = []
        last_choices[node_id] = 0
        P_El_Schedules[node_id] = []
        current_P_El_Schedule[node_id] = np.zeros(96)
        penalties[node_id] = []
        node['entity'].update_model(models[node_id])

    city_district.update_model(models[0])
    cd_mod_obj = city_district.get_objective()

    # ----------------
    # Start scheduling
    # ----------------
    # do optimization iterations until stopping criteria are met
    from pycity_scheduling.util.mpi_optimize_nodes import mpi_context
    with TemporaryDirectory() as dirname:
        with mpi_context() as MPI_Workers:
            start_tick = time.monotonic()
            while iteration < max_iterations:#TODO
                iteration += 1
                if iteration > max_iterations:
                    raise PyCitySchedulingMaxIteration(
                        "Exceeded iteration limit of {0} iterations"
                        .format(max_iterations)
                    )
                if max_time is not None and start_tick + max_time <= time.monotonic():
                    raise PyCitySchedulingMaxIteration(
                        "Exceeded time limit of {0} seconds\n"
                        .format(time.monotonic() - start_tick)
                    )

                # -----------------
                # 1) optimize nodes
                # -----------------
                print("optimize nodes")
                mpi_models = {}
                for node_id, node in nodes.items():
                    entity = node['entity']
                    if not isinstance(
                            entity,
                            (Building, Photovoltaic, WindEnergyConverter)
                    ):
                        continue

                    obj = entity.get_objective(beta)
                    # penalty term is expanded and constant is omitted
                    if not iteration == 1:
                        mod_obj = gurobi.QuadExpr()
                        mod_obj += obj
                        mod_obj.addTerms(
                            incentives,
                            entity.P_El_vars
                        )
                    else:
                        mod_obj = obj
                    model = models[node_id]
                    model.setObjective(mod_obj)
                    objs[node_id].append(mod_obj)

                    mpi_models[node_id] = model

                # to make recorded time not dependent on worker count time is stopped
                mpi_start = time.monotonic()
                MPI_Workers.calculate(mpi_models)
                mpi_end = time.monotonic()
                start_tick += (mpi_end - mpi_start) - max([model.Params.TimeLimit for model in mpi_models.values()])

                for node_id, node in nodes.items():
                    entity = node['entity']
                    if not isinstance(
                            entity,
                            (Building, Photovoltaic, WindEnergyConverter)
                    ):
                        continue

                    obj = entity.get_objective(beta)
                    model.write("%s%d_%i.sol".format(dirname, node_id, iteration-1))

                    try:
                        P_El_Schedules[node_id].append(np.zeros(op_horizon))
                        np.copyto(
                            P_El_Schedules[node_id][-1],
                            [var.x for var in entity.P_El_vars]
                        )
                        if iteration == 1:
                            penalties[node_id].append(obj.getValue())
                        else:
                            penalties[node_id].append(obj.getValue())
                    except gurobi.GurobiError:
                        print("Model Status: %i" % model.status)
                        if model.status == 4:  # and settings.DEBUG:
                            model.computeIIS()
                            model.write("infeasible.ilp")
                        raise PyCitySchedulingGurobiException(
                            "{0}: Could not read from variables at iteration {1}."
                            .format(str(entity), iteration)
                        )

                # ----------------------
                # 2) optimize aggregator
                # ----------------------
                model = models[0]

                for constraint in cd_power_constraints:
                    model.remove(constraint)
                cd_power_constraints = []

                for node_id, node in nodes.items():
                    choice = model.addVar(vtype=gurobi.GRB.BINARY, name="%s_choice_for=%s_%s" % (city_district._long_ID, node['entity']._long_ID, str(iteration)))
                    cd_choices[node_id].append(choice)
                    for i, p in enumerate(total_P_el):
                        p += P_El_Schedules[node_id][-1][i]*choice
                    cd_mod_obj += penalties[node_id][-1]*choice
                    cd_power_constraints.append(
                        model.addConstr(gurobi.quicksum(cd_choices[node_id]) == 1)
                    )
                for i in range(op_horizon):
                    cd_power_constraints.append(model.addConstr(city_district.P_El_vars[i] == total_P_el[i]))
                model.update()

                print("start gurobi for agg")
                model.setObjective(cd_mod_obj)
                model.optimize()
                print("gurobi finished")
                print(str(cd_mod_obj.getValue()))
                try:
                    for node_id, choices in cd_choices.items():
                        for i, choice in enumerate(choices):
                            if choice.x > 0.9:
                                current_P_El_Schedule[node_id] = P_El_Schedules[node_id][i]
                                break
                    np.copyto(
                        current_P_El_Schedule[0],
                        [var.x for var in city_district.P_El_vars]
                    )
                except gurobi.GurobiError:
                    print("Model Status: %i" % model.status)
                    if model.status == 4:  # and settings.DEBUG:
                        model.computeIIS()
                        model.write("infeasible.ilp")
                    raise PyCitySchedulingGurobiException(
                        "{0}: Could not read from variables."
                            .format(str(city_district))
                    )
                print("incentive update")
                incentives = (1-rho)*incentives+rho*get_obj_mod_lin(city_district.get_objective(), city_district.P_El_vars)
                if iteration == 1:
                    incentives /= rho
                # TODO: Think about stopping criteria

                print("doing callback")
                for node_id in nodes.keys():
                    c = None
                    for i, choice in enumerate(cd_choices[node_id]):
                        if choice.x > 0.9:
                            c = i
                            break
                    model = models[node_id]
                    model.setObjective(objs[node_id][c])
                    model.read("%s%d_%i.sol".format(dirname, node_id, c))
                    oldtime = model.Params.TimeLimit
                    model.Params.TimeLimit = 0.5
                    model.optimize()
                    model.Params.TimeLimit = oldtime
                if iteration_callback is not None:
                    iteration_callback(city_district, models, time=time.monotonic() - start_tick)

    city_district.update_schedule()
    for entity in city_district.get_lower_entities():
        entity.update_schedule()

    return iteration
