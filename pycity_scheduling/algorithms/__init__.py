from .stand_alone_optimization_algorithm import stand_alone_optimization
from .local_optimization_algorithm import local_optimization
from .exchange_admm_algorithm import exchange_admm
from .central_optimization_algortihm import central_optimization
from .dual_decomposition_algortihm import dual_decomposition
from .exchange_admm_algorithm_mpi import exchange_admm_mpi
from .exchange_admm_r_and_f_mpi import exchange_admm_r_and_f_mpi

algorithms = {
    "stand-alone": stand_alone_optimization,
    "local": local_optimization,
    "exchange-admm": exchange_admm,
    "exchange-admm-mpi": exchange_admm_mpi,
    "central": central_optimization,
    "dual-decomposition": dual_decomposition,
    "exchange_admm_r_and_f_mpi": exchange_admm_r_and_f_mpi
}

__all__ = [
    "stand_alone_optimization",
    "local_optimization",
    "exchange_admm",
    "exchange_admm_mpi",
    "central_optimization",
    "dual_decomposition",
    "algorithms",
    "exchange_admm_r_and_f_mpi"
]
