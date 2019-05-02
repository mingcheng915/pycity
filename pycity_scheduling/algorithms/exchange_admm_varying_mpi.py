import numpy as np
import gurobipy as gurobi
import time

from pycity_scheduling.classes import *
from pycity_scheduling.exception import *
from pycity_scheduling.util import populate_models


def exchange_admm_varying_mpi(city_district, models=None, beta=1.0, eps_primal=0.1,
                  eps_dual=1.0, rho=2.0, my=10, tau_incr=2.0, tau_decr=2.0, max_iterations=10000, max_time=None, iteration_callback=None):
    from pycity_scheduling.util.mpi_optimize_nodes import mpi_context
    """Perform Exchange ADMM on a city district.

    Do the scheduling of electrical energy in a city district using the
    Exchange ADMM algorithm, concerning different objectives for clients and
    the aggregator.

    Parameters
    ----------
    city_district : pycity_scheduling.classes.CityDistrict
    models : dict, optional
        Holds a `gurobi.Model` for each node and the aggregator.
    beta : float, optional
        Tradeoff factor between system and customer objective. The customer
        objective is multiplied with beta.
    eps_primal : float, optional
        Primal stopping criterion for the ADMM algorithm.
    eps_dual : float, optional
        Dual stopping criterion for the ADMM algorithm.
    rho : float, optional
        Stepsize for the ADMM algorithm.
    my : float, optional
        Threshold in size difference between ||s||² and ||r||² to trigger rho update
    tau_incr : float, optional
        Value to increase rho by if ||r||² > my ||s||²
    tau_decr : float, optional
        Value to decrease rho by if ||s||² > my ||r||²
    max_iterations : int, optional
        Maximum number of ADMM iterations.
    max_time : float, optional
        Maximum number of seconds to iterate.

    Returns
    -------
    int :
        Number of iterations.
    numpy.ndarray of float :
        Norms of r for all iterations.
    numpy.ndarray of float :
        Norms of s for all iterations.
    numpy.ndarray of float :
        Scaled price vector after last iteration.

    Notes
    -----
     - uses the Exchange ADMM algorithm described in [1]
     - not yet suitable for exchanging thermal power
     - uses varying penalty parameter as described in [2]

    References
    ----------
    .. [1] "Alternating Direction Method of Multipliers for Decentralized
       Electric Vehicle Charging Control" by Jose Rivera, Philipp Wolfrum,
       Sandra Hirche, Christoph Goebel, and Hans-Arno Jacobsen
       Online: https://mediatum.ub.tum.de/doc/1187583/1187583.pdf
    .. [2] "Distributed Optimization and Statistical Learning via the
       Alternating Direction Method of Multipliers" by Stephen Boyd,
       Neal Parikh, Eric Chu, Borja Peleato and Jonathan Eckstein
       Online: https://stanford.edu/class/ee367/reading/admm_distr_stats.pdf
    """

    op_horizon = city_district.op_horizon
    nodes = city_district.nodes
    n = 1 + len(nodes)

    iteration = 0
    x_ = np.zeros(op_horizon)
    u = np.zeros(op_horizon)
    old_x_ = np.zeros(op_horizon)
    old_P_El_Schedule = {}
    current_P_El_Schedule = {}
    s = np.zeros(n * op_horizon)
    r_norms = [gurobi.GRB.INFINITY]
    s_norms = [gurobi.GRB.INFINITY]

    if models is None:
        models = populate_models(city_district, "admm")
    old_P_El_Schedule[0] = np.zeros(op_horizon)
    current_P_El_Schedule[0] = np.zeros(op_horizon)
    for node_id, node in nodes.items():
        old_P_El_Schedule[node_id] = np.zeros(op_horizon)
        current_P_El_Schedule[node_id] = np.zeros(op_horizon)
        node['entity'].update_model(models[node_id])

    city_district.update_model(models[0])

    # ----------------
    # Start scheduling
    # ----------------

    # do optimization iterations until stopping criteria are met
    with mpi_context() as MPI_Workers:
        start_tick = time.monotonic()
        while r_norms[-1] > eps_primal or s_norms[-1] > eps_dual:
            iteration += 1
            if iteration > max_iterations:
                raise PyCitySchedulingMaxIteration(
                    "Exceeded iteration limit of {0} iterations\n"
                    "Norms were ||r|| =  {1}, ||s|| = {2}"
                    .format(max_iterations, r_norms[-1], s_norms[-1])
                )
            if max_time is not None and start_tick + max_time <= time.monotonic():
                raise PyCitySchedulingMaxIteration(
                    "Exceeded time limit of {0} seconds\n"
                    "Norms were ||r|| =  {1}, ||s|| = {2}"
                    .format(time.monotonic()-start_tick, r_norms[-1], s_norms[-1])
                )

            np.copyto(old_x_, x_)
            #vary rho
            if iteration > 1:
                if r_norms[-1] > my * s_norms[-1]:
                    rho *= tau_incr
                    u /= tau_incr
                elif s_norms[-1] > my * r_norms[-1]:
                    rho /= tau_decr
                    u *= tau_decr
            # -----------------
            # 1) optimize nodes
            # -----------------
            mpi_models = {}
            for node_id, node in nodes.items():
                entity = node['entity']
                if not isinstance(
                        entity,
                        (Building, Photovoltaic, WindEnergyConverter)
                ):
                    continue
                np.copyto(
                    old_P_El_Schedule[node_id],
                    current_P_El_Schedule[node_id]
                )

                obj = entity.get_objective(beta)
                # penalty term is expanded and constant is omitted
                obj.addTerms(
                    [rho / 2] * op_horizon,
                    entity.P_El_vars,
                    entity.P_El_vars
                )
                obj.addTerms(
                    (rho * (- pv + xv + uv) for pv, xv, uv in
                     zip(current_P_El_Schedule[node_id], x_, u)),
                    entity.P_El_vars
                )

                model = models[node_id]
                model.setObjective(obj)
                model.update()
                mpi_models[node_id]=model

            MPI_Workers.calculate(mpi_models)

            for node_id, node in nodes.items():
                entity = node['entity']
                if not isinstance(
                        entity,
                        (Building, Photovoltaic, WindEnergyConverter)
                ):
                    continue
                model = models[node_id]
                try:
                    np.copyto(
                        current_P_El_Schedule[node_id],
                        [var.x for var in entity.P_El_vars]
                    )
                except gurobi.GurobiError:
                    print("Model Status: %i" % model.status)
                    if model.status == 4:  # and settings.DEBUG:
                        model.computeIIS()
                        model.write("infeasible.ilp")
                    if model.status == 12:  # and settings.DEBUG:
                        print("Last P_El_Schedule:")
                        print(current_P_El_Schedule[node_id])
                        print("Last x_:")
                        print(x_)
                    raise PyCitySchedulingGurobiException(
                        "{0}: Could not read from variables at iteration {1}."
                        .format(str(entity), iteration)
                    )

            # ----------------------
            # 2) optimize aggregator
            # ----------------------
            model = models[0]
            np.copyto(
                old_P_El_Schedule[0],
                current_P_El_Schedule[0]
            )
            obj = city_district.get_objective()
            # penalty term is expanded and constant is omitted
            # invert sign of P_El_Schedule and P_El_vars (omitted for quadratic
            # term)
            obj.addTerms(
                [rho / 2] * op_horizon,
                city_district.P_El_vars,
                city_district.P_El_vars
            )
            obj.addTerms(
                (rho * (- pv - xv - uv) for pv, xv, uv in
                 zip(current_P_El_Schedule[0], x_, u)),
                city_district.P_El_vars
            )
            model.setObjective(obj)
            model.optimize()

            try:
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

            # --------------------------
            # 3) incentive signal update
            # --------------------------
            np.copyto(x_, -current_P_El_Schedule[0])
            for node_id in nodes.keys():
                x_ += current_P_El_Schedule[node_id]
            x_ /= n
            u += x_

            # ------------------------------------------
            # Calculate parameters for stopping criteria
            # ------------------------------------------
            # TODO: Think about stopping criteria
            # From an interpretational perspective it would make sense to remove
            # the `n *` for the r norm and introduce a `1/n` factor for the s norm
            r_norms.append(n * np.linalg.norm(x_))
            np.copyto(
                s[0:op_horizon],
                - rho * (
                    -current_P_El_Schedule[0] + old_P_El_Schedule[0] + old_x_ - x_)
            )
            i1 = op_horizon
            for node_id, node in nodes.items():
                np.copyto(
                    s[i1:i1+op_horizon],
                    - rho * (
                        current_P_El_Schedule[node_id] - old_P_El_Schedule[node_id]
                        + old_x_ - x_
                    )
                )
                i1 += op_horizon
            s_norms.append(np.linalg.norm(s))
            if iteration_callback is not None:
                iteration_callback(city_district, models, r_norm=r_norms[-1], s_norm=s_norms[-1], rho=rho, time=time.monotonic() - start_tick)

        city_district.update_schedule()
        for entity in city_district.get_lower_entities():
            entity.update_schedule()

        # if settings.BENCHMARK:
        #     print("\nNumber of ADMM iterations: {0}".format(iteration))
        #     print("r: {0}, s: {1}".format(r_norms[-1], s_norms[-1]))

        return iteration, r_norms, s_norms, u
