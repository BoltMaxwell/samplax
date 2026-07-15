"""samplax: a curated SG-MCMC sampler library in pure JAX.

Kernels are vendored from verified ports of the original papers' code:

- SGLD / SGHMC          csgmcmc-jax + SGHMC-jax   (Chen et al. 2014; Zhang et al. 2020)
- pSGLD preconditioning Li et al. 2016 (matching ift-sde's psgld)
- AMAGOLD               amagold-jax               (Zhang, Cooper, De Sa 2020)
- low-precision SGLD    low-precision-sgld-jax    (Zhang, Wilson, De Sa 2022)
- cyclical schedules    csgmcmc-jax               (Zhang et al. 2020)

Core protocol: ``kernel = samplax.sgld(...)``; ``state = kernel.init(key, x)``;
``state = kernel.step(key, state, grad, step_size, temperature)`` — see
:mod:`samplax.base`.
"""

from . import schedules
from .base import Kernel, gaussian_like
from .gibbs import gibbs_precision
from .kernels.amagold import amagold, amagold_minibatch
from .kernels.hmc import hmc
from .kernels.sghmc import SGHMCState, sghmc
from .kernels.sgld import SGLDState, sgld
from .preconditioners import Preconditioner, identity, rmsprop
from .schedules import ScheduleState, constant, cyclical, exponential, polynomial
from .transforms import quant, vc
from .transforms.lp_sgld import LPKernel, lp_sgld

__version__ = "0.1.1"

__all__ = [
    "Kernel", "gaussian_like",
    "sgld", "SGLDState", "sghmc", "SGHMCState",
    "hmc", "amagold", "amagold_minibatch",
    "lp_sgld", "LPKernel", "quant", "vc",
    "Preconditioner", "identity", "rmsprop",
    "schedules", "ScheduleState", "constant", "cyclical", "exponential", "polynomial",
    "gibbs_precision",
]
