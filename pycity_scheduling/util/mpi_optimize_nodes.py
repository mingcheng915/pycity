from mpi4py import MPI
from pycity_scheduling.classes import *
from pycity_scheduling.exception import *
import numpy as np
import gurobipy as gurobi
import sys
from typing import Iterable

class Node(dict):
    def __init__(self, worker: int, node):
        super(Node, self).__init__(node)
        self.__worker_id__ = worker
        self.__idata__ = None
    def do_iteration(self,quad_obj, lin_obj, timeLimit=60):
        self.__i_data__ = {'node_id': node.name, 'timeLimit': timeLimit, 'obj_mod':[quad_obj, lin_obj]}

class MPI_Nodes(dict):
    def __init__(self, comm, nodes):
        super(MPI_Nodes, self).__init__()
        self.__comm__ = comm
        self.__size__ = comm.Get_remote_size()
        self.__workers__ = range(1,self.__size__)
        for node_i, node in enumerate(nodes):
            self[node["name"]] = Node(node_i % len(self.__workers__), node)
        for worker in self.__workers__:
            comm.send([node for node in self.values() if node.__worker_id__ == worker], dest=worker, tag=10)

    def calculate(self):
        #send work out
        for worker_id in range(self.__size__):
            comm.send([node.__i_data__ for node in self.values() if
                       node.__i_data__ is not None and node.__worker_id__ == worker_id]
                      , dest=worker_id, tag=11)
        #reset i_data
        for node in self.values():
            node.__i_data__ = None
        #collect work
        for worker_id in range(self.__size__):
            data = comm.recv(source=worker_id, tag=12)
            for name, p in data.items():
                self[name]['P_El_Schedule'] = p
        return


class mpi_context:
    def __init__(self,nodes,max_procs=10):
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['/home/schwarz/sue/Git/pyCity_scheduling/pycity_scheduling/util/mpi_optimize_nodes.py'], maxprocs=max_procs)
        self.size = self.comm.Get_remote_size()
        self.obj = MPI_Nodes(self.comm, nodes)
        return self.obj
    def __enter__(self):
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i in range(self.size):
            self.comm.send(None, dest=i, tag=11)
        self.comm.Disconnect()





if __name__ == '__main__':

    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()

    nodes = comm.recv(source=0, tag=10)
    for node in nodes:
        m = gurobi.Model("single node")
        m.setParam("LogFile", "")
        m.setParam("Threads", 1)
        node["model"] = m
    i_data = comm.recv(source=0, tag=11)
    while i_data is not None:
        output = {}
        for node_data in i_data:
            if node_data is None:
                continue
            node_id = node_data["node_id"]
            node = nodes[node_id]
            entity = node['entity']
            obj_mod = node_data['obj_mod']
            m = node["model"]
            m.setParam("TimeLimit", node_data['timeLimit'])
            if not isinstance(
                    entity,
                    (Building, Photovoltaic, WindEnergyConverter)
            ):
                continue
            obj = entity.get_objective()
            # penalty term is expanded and constant is omitted
            obj.addTerms(
                obj_mod[0],
                entity.P_El_vars,
                entity.P_El_vars
            )
            obj.addTerms(
                obj_mod[1],
                entity.P_El_vars
            )

            m.setObjective(obj)
            m.optimize()

            try:
                output[node.name] = [var.x for var in entity.P_El_vars]
            except gurobi.GurobiError:
                print("Model Status: %i" % m.status)
                raise PyCitySchedulingGurobiException(
                    "{0}: Could not read from variables."
                        .format(str(node))
                )
        comm.send(output, dest=0, tag=12)
        i_data = comm.recv(source=0, tag=11)
    comm.Disconnect()
