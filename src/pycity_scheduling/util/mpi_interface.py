"""
The pycity_scheduling framework


Copyright (C) 2021,
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

import os
import sys
import numpy as np


class MPIInterface:
    def __init__(self):
        self.mpi = None
        self.mpi_comm = None
        self.mpi_rank = 0
        self.mpi_size = 1
        try:
            from mpi4py import MPI
            self.mpi = MPI
            self.mpi_comm = MPI.COMM_WORLD
            self.mpi_rank = self.mpi_comm.Get_rank()
            self.mpi_size = self.mpi_comm.Get_size()
        except:
            pass

    def get_rank(self):
        return self.mpi_rank

    def get_size(self):
        return self.mpi_size

    def get_comm(self):
        return self.mpi_comm

    def get_mpi_process_range(self, n):
        # Determine which MPI processes is responsible for which node(s):
        if self.get_size() > n:
            mpi_process_range = np.array([i for i in range(n)])
        elif self.get_size() < n:
            if self.get_size() == 1:
                mpi_process_range = np.array([0 for i in range(n)])
            else:
                a, b = divmod(n - 1, self.get_size() - 1)
                mpi_process_range = np.repeat(np.array([i for i in range(1, self.get_size())]), a)
                for i in range(b):
                    mpi_process_range = np.append(mpi_process_range, i + 1)
                mpi_process_range = np.concatenate([[0], mpi_process_range])
        else:
            mpi_process_range = np.array([i for i in range(n)])
        mpi_process_range = np.sort(mpi_process_range)
        return mpi_process_range

    def disable_multiple_printing(self):
        """
        Turn off printing for all MPI processes with MPI rank other than 0 and always flush prints for rank 0.
        """
        sys.stdout = UnbufferedPrint(sys.stdout)
        sys.stderr = UnbufferedPrint(sys.stderr)
        if self.mpi is not None and self.mpi_size > 1:
            if self.mpi_rank > 0:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')


class UnbufferedPrint(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, data):
        self.stream.writelines(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)
