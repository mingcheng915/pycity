from mpi4py import MPI
from pycity_scheduling.classes import *
from pycity_scheduling.exception import *
import numpy as np
import gurobipy as gurobi
import sys
from .populate_models import populate_models

class Node(dict):
    def __init__(self, worker: int, node):
        super(Node, self).__init__(self)
        self.update(node)
        self.__worker_id__ = worker
        self.__idata__ = None
    def do_iteration(self):
        quad = gurobi.QuadExpr()
        quad += self['model'].getObj()
        lin = quad.getLinExpr()
        self.__i_data__ = {'node_id': self['name'], 'obj': {
            'quad': [{'var1': quad.getVar1(i).varname, 'var2': quad.getVar2(i).varname, 'coeff': quad.getCoeff(i)}
                     for i in range(quad.size())],
            'lin': [{'var': lin.getVar(i).varname, 'coeff': lin.getCoeff(i)}
                    for i in range(lin.size())],
            'const': lin.getConstant()
        }}

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
            self.__comm__.send(list(tosend), dest=worker_id, tag=11)
        #reset i_data
        for node in self.values():
            node.__i_data__ = None
        #collect work
        for worker_id in range(self.__size__):
            data = self.__comm__.recv(source=worker_id, tag=12)
            for node_id, node_data in data.items():
                bounds = []
                for varname, var in node_data['vars']:
                    model_var = self[node_id].__model__.getVarByName(varname)
                    bounds.append({'ub': model_var.ub, 'lb': model_var.lb, 'var': model_var})
                    model_var.ub = var
                    model_var.lb = var
                self[node_id].__model__.optimize()
                for b in bounds:
                    b['var'].ub = b['ub']
                    b['var'].lb = b['lb']
                assert self[node_id].__model__.getObj().getValue() == node_data['obj']
        return


class mpi_context:
    def __init__(self, city_district, max_procs=20):
        nodes = city_district.nodes.values()
        procs = min(max_procs, len(nodes))
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['/home/uerlich/ba/pycity_scheduling/pycity_scheduling/util/mpi_optimize_nodes.py'], maxprocs=procs)
        self.size = self.comm.Get_remote_size()
        self.obj = MPI_Nodes(self.comm, nodes)
        self.models = populate_models(city_district, "admm")#TODO
        for node_id, node in nodes.items():
            node['entity'].update_model(self.models[node_id])
        city_district.update_model(self.models[0])

    def __enter__(self):
        return self.obj, self.models

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
            obj = node_data['obj']
            m = node["model"]
            if not isinstance(
                    entity,
                    (Building, Photovoltaic, WindEnergyConverter)
            ):
                continue
            total_obj = gurobi.QuadExpr()
            for entry in total_obj['quad']:
                total_obj.add(m.getVarByName(entry['var1']) * m.getVarByName(entry['var2']), entry['coeff'])
            for entry in total_obj['lin']:
                total_obj.add(m.getVarByName(entry['var']), entry['coeff'])
            total_obj += obj['const']
            m.setObjective(total_obj)
            m.optimize()

            print("mpi:"+ str(node_id) +" " + str(m.getObjective().getValue()))
            try:
                output[node_id] = {'vars': dict((var.VarName, var.X) for var in m.getVars()),
                                   'obj': total_obj.getValue()}
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
