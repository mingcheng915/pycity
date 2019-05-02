from mpi4py import MPI
import gurobipy as gurobi
import sys
import tempfile

def unpack_model(data: dict) -> gurobi.Model:
    with tempfile.TemporaryDirectory() as dirname:
        with open(dirname + "/tmp.mps", 'w+') as f:
            f.write(data['mps'])
        model = gurobi.read(dirname + "/tmp.mps")
        model.setParam('OutputFlag', False)
        with open(dirname + "/tmp.hnt", 'w+') as f:
            f.write(data['hnt'])
        model.read(dirname + "/tmp.hnt")
        if 'mst' in data:
            with open(dirname + "/tmp.mst", 'w+') as f:
                f.write(data['mst'])
            model.read(dirname + "/tmp.mst")
        with open(dirname + "/tmp.prm", 'w+') as f:
            f.write(data['prm'])
        model.read(dirname + "/tmp.prm")
        model.update()
        return model

def unpack_results(data: dict, model: gurobi.Model) -> []:
    with tempfile.TemporaryDirectory() as dirname:
        assert 'sol' in data
        with open(dirname + "/tmp.hnt", 'w+') as f:
            f.write(data['hnt'])
        model.read(dirname + "/tmp.hnt")
        with open(dirname + "/tmp.sol", 'w+') as f:
            f.write(data['sol'])
        model.read(dirname + "/tmp.sol")
        oldtime = model.Params.TimeLimit
        model.Params.TimeLimit = 0.5
        model.optimize()
        model.Params.TimeLimit = oldtime
        return data.get("all_sols", [])


def pack_model(model_id, model, solution=False) -> dict:
    tosend = {}
    tosend['id'] = model_id
    model.update()
    with tempfile.TemporaryDirectory() as dirname:
        model.write(dirname + "/tmp.hnt")
        with open(dirname + "/tmp.hnt") as f:
            tosend['hnt'] = f.read()
        if solution is True:
            tosend['vars'] = {v.varname: v.x for v in model.getVars()}
            model.write(dirname + "/tmp.sol")
            with open(dirname + "/tmp.sol") as f:
                tosend['sol'] = f.read()
            tosend['all_sols'] = []
            if model.solCount is not None and model.solCount>1:
                for i in range(model.solCount):
                    model.setParam("SolutionNumber", i)
                    sol_i = {v.VarName: v.xn for v in model.getVars()}
                    sol_i["obj"] = model.PoolObjVal
                    tosend['all_sols'].append(sol_i)
            else:
                sol_0 = {v.VarName: v.X for v in model.getVars()}
                sol_0["obj"] = model.ObjVal
                tosend['all_sols'].append(sol_0)
        else:
            model.write(dirname + "/tmp.mps")
            with open(dirname + "/tmp.mps") as f:
                tosend['mps'] = f.read()
            if not any([var.start == gurobi.GRB.UNDEFINED for var in model.getVars()]):
                model.write(dirname + "/tmp.mst")
                with open(dirname + "/tmp.mst") as f:
                    tosend['mst'] = f.read()
            model.write(dirname + "/tmp.prm")
            with open(dirname + "/tmp.prm") as f:
                tosend['prm'] = f.read()
    return tosend

class MPI_Models():
    def __init__(self, comm):
        self.__comm__ = comm
        self.__size__ = comm.Get_remote_size()
        self.__workers__ = list(range(self.__size__))

    def calculate(self, models):
        """
        Parameters
        ----------
        models : dict
            Dict of gurobi models, that are to be optimized
        """
        #send work out
        all_solutions = {model_id: [] for model_id in models.keys()}
        free = self.__workers__.copy()
        for model_id, model in models.items():
            if len(free) == 0:#wait to receive results from a worker
                info = MPI.Status()
                recv = self.__comm__.recv(tag=12, source=MPI.ANY_SOURCE, status=info)
                all_solutions[recv['id']] = unpack_results(recv, models[recv['id']])
                free.append(info.Get_source())
            tosend = pack_model(model_id, model, False)
            assert tosend is not None
            self.__comm__.send(tosend, dest=free.pop(0), tag=11)

        #collect remaining work
        for worker in self.__workers__:
            if worker not in free:
                recv = self.__comm__.recv(tag=12, source=MPI.ANY_SOURCE)
                all_solutions[recv['id']] = unpack_results(recv, models[recv['id']])
        print("finished calculation")
        return all_solutions

class mpi_context:
    """This Class provides support for distributing gurobi models to mpi workers."""
    def __init__(self, procs=None):
        """

        Parameters
        ----------
        procs : int
            Maximum Number of workers to spawn. Default to MPI universe size minus current world size.
        """
        import warnings
        kwargs = {}
        if procs is not None:
            kwargs["maxprocs"]=procs
        else:
            assert MPI.UNIVERSE_SIZE != MPI.KEYVAL_INVALID
            kwargs["maxprocs"] = MPI.COMM_WORLD.Get_attr(MPI.UNIVERSE_SIZE)
            kwargs["maxprocs"] -= MPI.COMM_WORLD.Get_size()
            assert type(kwargs["maxprocs"]) == int
            if kwargs["maxprocs"]>100:
                warnings.warn("MPI Context received huge proc parameter: {}". format(kwargs["maxprocs"]))
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['/home/uerlich/ba/pycity_scheduling/pycity_scheduling/util/mpi_optimize_nodes.py'], **kwargs)
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
        model = unpack_model(i_data)
        model.optimize()
        output = pack_model(i_data['id'], model, True)
        comm.send(output, dest=0, tag=12)
        i_data = comm.recv(source=0, tag=11)
    comm.Disconnect()
