from mpi4py import MPI
from pycity_scheduling.classes import *
from pycity_scheduling.exception import *
import gurobipy as gurobi
import sys
import tempfile

def unpack_model(data: dict, model: gurobi.Model):
    with tempfile.TemporaryDirectory() as dir:
        with open(dir.name + "/tmp.hnt", 'w+') as f:
            f.write(data['hnt'])
        model.read(dir.name + "/tmp.hnt")
        if 'sol' in data:
            with open(dir.name + "/tmp.sol", 'w+') as f:
                f.write(data['sol'])
            model.read(dir.name + "/tmp.sol")
        else:
            with open(dir.name + "/tmp.mps", 'w+') as f:
                f.write(data['mps'])
            model.read(dir.name + "/tmp.mps")
            with open(dir.name + "/tmp.mst", 'w+') as f:
                f.write(data['mst'])
            model.read(dir.name + "/tmp.mst")
            with open(dir.name + "/tmp.prm", 'w+') as f:
                f.write(data['prm'])
            model.read(dir.name + "/tmp.prm")
        model.update()

def pack_model(model_id, model, solution=False) -> dict:
    tosend = {}
    tosend['id'] = model_id
    model.update()
    with tempfile.TemporaryDirectory() as dir:
        model.write(dir.name + "/tmp.hnt")
        with open(dir.name + "/tmp.hnt") as f:
            tosend['hnt'] = f.read()
        if solution is True:
            model.write(dir.name + "/tmp.sol")
            with open(dir.name + "/tmp.sol") as f:
                tosend['sol'] = f.read()
        else:
            model.write(dir.name + "/tmp.mps")
            with open(dir.name + "/tmp.mps") as f:
                tosend['mps'] = f.read()
            model.write(dir.name + "/tmp.mst")
            with open(dir.name + "/tmp.mst") as f:
                tosend['mst'] = f.read()
            model.write(dir.name + "/tmp.prm")
            with open(dir.name + "/tmp.prm") as f:
                tosend['prm'] = f.read()

class MPI_Models():
    def __init__(self, comm):
        self.__comm__ = comm
        self.__size__ = comm.Get_remote_size()
        self.__workers__ = list(range(self.__size__))

    def calculate(self, models):
        #send work out
        free = self.__workers__
        for model_id, model in self.items():
            if len(free) == 0:#wait to receive results from a worker
                info = MPI.Status()
                recv = self.__comm__.recv(tag=12, source=MPI.ANY_SOURCE, status=info)
                unpack_model(recv, self[recv['id']])
                free.append(info.Get_source())
            tosend = pack_model(model_id, model, False)
            self.__comm__.send(tosend, dest=free.pop(1), tag=11)

        #collect remaining work
        for worker in self.__workers__:
            if worker not in free:
                recv = self.__comm__.recv(tag=12, source=MPI.ANY_SOURCE, status=info)
                unpack_model(recv, self[recv['id']])

class mpi_context:
    def __init__(self, procs=20):
        assert procs >= 1
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['/home/uerlich/ba/pycity_scheduling/pycity_scheduling/util/mpi_optimize_nodes.py'], maxprocs=procs)
        self.modules = MPI_Models(self.comm)

    def __enter__(self):
        return self.modules

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i in self.modules.__workers__:
            self.comm.send(None, dest=i, tag=11)
        self.comm.Disconnect()


if __name__ == '__main__':
    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()
    i_data = comm.recv(source=0, tag=11)
    while i_data is not None:
        model = gurobi.Model()
        unpack_model(i_data, model)
        model.optimize()
        output = pack_model(i_data['id'], model, True)
        comm.send(output, dest=0, tag=12)
        i_data = comm.recv(source=0, tag=11)
    print("exiting")
    comm.Disconnect()
