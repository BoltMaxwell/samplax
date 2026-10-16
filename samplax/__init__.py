"""samplax: a curated SG-MCMC sampler library in pure JAX.

Vendored kernels, equivalence-tested against verified JAX ports of the paper
authors' code:

- SGLD / SGHMC           csgmcmc-jax + SGHMC-jax     (Chen et al. 2014; Zhang et al. 2020)
- AMAGOLD                amagold-jax                 (Zhang, Cooper, De Sa 2020)
- low-precision SGLD     low-precision-sgld-jax      (Zhang, Wilson, De Sa 2022)
- cyclical schedules     csgmcmc-jax                 (Zhang et al. 2020)
- HMC reference          SGHMC-jax                   (Chen et al. 2014)

Vendored constructions, checked against targets (no equivalence test):

- pSGLD preconditioning  Li et al. 2016 (RMSprop; same construction as a private ift-sde sampler)
- Gibbs Gamma hyperpriors  SGHMC-jax (Chen et al. 2014)

Original kernels, implemented here and verified against closed-form targets:

- pCN-within-Gibbs      Cotter et al. 2013 (pCN: whitened latent + M-H parameter block)

Core protocol: ``kernel = samplax.sgld(...)``; ``state = kernel.init(key, x)``;
``state = kernel.step(key, state, grad, step_size, temperature)`` — see
:mod:`samplax.base`.
"""

from . import schedules
from .base import Kernel, gaussian_like
from .gibbs import gibbs_precision
from .kernels.amagold import (
    AmagoldTrackedState, amagold, amagold_minibatch, amagold_tracked,
)
from .kernels.hmc import hmc
from .kernels.pcn_gibbs import pcn_gibbs
from .kernels.sghmc import SGHMCState, sghmc
from .kernels.sgld import SGLDState, sgld
from .preconditioners import Preconditioner, identity, rmsprop
from .schedules import ScheduleState, constant, cyclical, exponential, polynomial
from .transforms import quant, vc
from .transforms.lp_sgld import LPKernel, lp_sgld

__version__ = "0.8.0"

__all__ = [
    "Kernel", "gaussian_like",
    "sgld", "SGLDState", "sghmc", "SGHMCState",
    "hmc", "pcn_gibbs", "amagold", "amagold_minibatch",
    "amagold_tracked", "AmagoldTrackedState",
    "lp_sgld", "LPKernel", "quant", "vc",
    "Preconditioner", "identity", "rmsprop",
    "schedules", "ScheduleState", "constant", "cyclical", "exponential", "polynomial",
    "gibbs_precision",
]
