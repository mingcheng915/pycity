from mpi4py import MPI
from pycity_scheduling.classes import *
from pycity_scheduling.exception import *
import numpy as np
import gurobipy as gurobi
import sys
from typing import Iterable

class Node(dict):
    def __init__(self, worker: int, node):
        super(Node, self).__init__(self)
        self.update(node)
        self.__worker_id__ = worker
        self.__idata__ = None
    def do_iteration(self,quad_obj, lin_obj):
        self.__i_data__ = {'node_id': self['name'], 'obj_mod':[quad_obj, lin_obj]}

class MPI_Nodes(dict):
    def __init__(self, comm, nodes):
        super(MPI_Nodes, self).__init__()
        self.__comm__ = comm
        self.__size__ = comm.Get_remote_size()
        self.__workers__ = range(self.__size__)
        for node_i, node in enumerate(nodes):
            self[node["name"]] = Node(node_i % len(self.__workers__), node)
        for worker in self.__workers__:
            comm.send(dict((node_id, node) for node_id, node in self.items() if node.__worker_id__ == worker), dest=worker, tag=10)

    def calculate(self):
        #send work out
        for worker_id in range(self.__size__):
            tosend = [node.__i_data__ if node.__i_data__ is not None and node.__worker_id__ == worker_id else None for node in self.values()]
            self.__comm__.send(list(tosend) , dest=worker_id, tag=11)
        #reset i_data
        for node in self.values():
            node.__i_data__ = None
        #collect work
        for worker_id in range(self.__size__):
            data = self.__comm__.recv(source=worker_id, tag=12)
            for name, p in data.items():
                self[name]['vars'] = p
        return


class mpi_context:
    def __init__(self,nodes,max_procs=20):
        procs = min(max_procs, len(nodes))
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['/home/uerlich/ba/pycity_scheduling/pycity_scheduling/util/mpi_optimize_nodes.py'], maxprocs=procs)
        self.size = self.comm.Get_remote_size()
        self.obj = MPI_Nodes(self.comm, nodes)
    def __enter__(self):
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i in self.obj.__workers__:
            self.comm.send(None, dest=i, tag=11)
        self.comm.Disconnect()





if __name__ == '__main__':
    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()
    nodes = comm.recv(source=0, tag=10)
    for node_id, node in nodes.items():
        m = gurobi.Model(str(node_id) + " Scheduling Model")
        m.setParam("OutputFlag", False)
        m.setParam("LogFile", "")
        m.setParam("Threads", 1)
        m.setParam("MIPGap", 20)
        node['entity'].populate_model(m)
        node['entity'].update_model(m)
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

            print("mpi:"+ str(node_id) +" " + str(m.getObjective().getValue()))
            try:
                output[node_id] = dict((var.VarName, var.X) for var in m.getVars())
            except gurobi.GurobiError:
                print("Model Status: %i" % m.status)
                raise PyCitySchedulingGurobiException(
                    "{0}: Could not read from variables."
                        .format(str(node))
                )
        comm.send(output, dest=0, tag=12)
        i_data = comm.recv(source=0, tag=11)
    print("exiting")
    comm.Disconnect()
