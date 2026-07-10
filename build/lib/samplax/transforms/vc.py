"""Variance-corrected (VC) quantization -- the paper's core contribution.

A naive low-precision SGLD step adds Gaussian noise and then rounds, which distorts
the per-step noise variance. VC quantization instead produces a *quantized* sample
whose discrete distribution has exactly the target Langevin variance ``var``, so
low-precision SGLD keeps the correct stationary distribution.

Fixed-point path (``Q_vc``), ported from ``gaussian/gaussian.py:34-75`` and
``bnn/optim.py``. Two regimes for step ``D = 2**-fl`` and ``var_fix = D**2/4``
(the variance stochastic rounding already contributes at the half-step):

* ``var > var_fix`` : add the deficit as Gaussian noise, nearest-round, then add a
  sign-correlated discrete ``{+D,-D,0}`` correction (``_sample_mu``).
* ``var <= var_fix``: stochastically round (which itself has variance ``var_s``),
  then top up to ``var`` with discrete ``{+D,-D,0}`` noise (``_sample``).

All branches are computed and selected with ``jnp.where`` so ``var`` may be traced
(e.g. a cyclical learning rate).

Block-FP path (``Q_vc_block``): identical math, but the step ``D`` is per-block
(shared exponent per row, ``dim=0``) and recomputed each call from the tensor, giving
each output channel a grid matched to its own magnitude. The discrete samplers below
already work with an array-valued ``D`` via broadcasting, so only ``D`` and the base
quantizer change.
"""

import jax
import jax.numpy as jnp

from .quant import _EBIT, block_quantize, fixed_point_quantize


def _sample(key, var, D):
    """Discrete noise in {+D, -D, 0} with variance ``var`` (P(+D)=P(-D)=var/(2 D^2))."""
    p1 = var / (2 * D**2)
    u = jax.random.uniform(key, jnp.shape(var))
    return jnp.where(u < p1, D, jnp.where(u < 2 * p1, -D, 0.0))


def _sample_mu(key, mu, var_fix, D):
    """Mean-correcting discrete noise for the residual regime (``mu = |residual|``)."""
    p1 = (var_fix + mu**2 + mu * D) / (2 * D**2)
    p2 = (var_fix + mu**2 - mu * D) / (2 * D**2)
    u = jax.random.uniform(key, jnp.shape(mu))
    return jnp.where(u < p1, D, jnp.where(u < p1 + p2, -D, 0.0))


def Q_vc(key, mu, var, wl, fl):
    """Variance-corrected fixed-point quantization of ``mu`` targeting variance ``var``."""
    D = 2.0 ** (-fl)
    var_fix = D**2 / 4.0
    kg, ksmu, kqmu, ks = jax.random.split(key, 4)

    # regime A: var > var_fix
    x = mu + jnp.sqrt(jnp.maximum(var - var_fix, 0.0)) * jax.random.normal(kg, jnp.shape(mu))
    quant_x = fixed_point_quantize(x, wl, fl, "nearest")
    res_a = x - quant_x
    theta_a = quant_x + jnp.sign(res_a) * _sample_mu(ksmu, jnp.abs(res_a), var_fix, D)

    # regime B: var <= var_fix
    quant_mu = fixed_point_quantize(mu, wl, fl, "stochastic", kqmu)
    res_b = mu - quant_mu
    p1 = jnp.abs(res_b) / D
    var_s = (1.0 - p1) * res_b**2 + p1 * (-res_b + jnp.sign(res_b) * D) ** 2
    theta_b = quant_mu + _sample(ks, jnp.maximum(var - var_s, 0.0), D)

    theta = jnp.where(var > var_fix, theta_a, theta_b)
    t_max = 2.0 ** (wl - fl - 1) - D
    return jnp.clip(theta, -(2.0 ** (wl - fl - 1)), t_max)


def _block_D_FL(x, wl):
    """Per-row (dim 0) block step ``D`` and fractional length ``FL`` for a ``wl``-bit
    block-FP grid. The shared exponent is ``floor(log2(row max))``; ``FL = wl-2-exp``
    and ``D = 2**-FL`` match ``quant.block_quantize``'s grid, broadcast over the row."""
    me = jnp.max(jnp.abs(x.reshape(x.shape[0], -1)), axis=1)
    exp = jnp.clip(jnp.floor(jnp.log2(jnp.where(me == 0, 1.0, me))),
                   -(2 ** (_EBIT - 1)), 2 ** (_EBIT - 1) - 1)
    shape = [x.shape[0]] + [1] * (x.ndim - 1)
    FL = (wl - 2 - exp).reshape(shape)
    return 2.0 ** (-FL), FL


def Q_vc_block(key, mu, var, wl):
    """Block-FP variance-corrected quantization (port of ``bnn/optim.py:fp_Q_vc``).

    Same two regimes as :func:`Q_vc`, but the step ``D`` (hence ``var_fix``) is
    per-block and adaptive. Regime A recomputes the grid from the noised ``x``;
    regime B uses the grid from ``mu``.
    """
    D_mu, FL_mu = _block_D_FL(mu, wl)
    var_fix = D_mu**2 / 4.0
    kg, ksmu, kqmu, ks = jax.random.split(key, 4)

    # regime A: var > var_fix (grid recomputed from the noised x)
    x = mu + jnp.sqrt(jnp.maximum(var - var_fix, 0.0)) * jax.random.normal(kg, jnp.shape(mu))
    quant_x = block_quantize(x, wl, "nearest", dim=0)
    D_x, FL_x = _block_D_FL(x, wl)
    res_a = x - quant_x
    theta_a = quant_x + jnp.sign(res_a) * _sample_mu(ksmu, jnp.abs(res_a), var_fix, D_x)

    # regime B: var <= var_fix (grid from mu)
    quant_mu = block_quantize(mu, wl, "stochastic", kqmu, dim=0)
    res_b = mu - quant_mu
    p1 = jnp.abs(res_b) / D_mu
    var_s = (1.0 - p1) * res_b**2 + p1 * (-res_b + jnp.sign(res_b) * D_mu) ** 2
    theta_b = quant_mu + _sample(ks, jnp.maximum(var - var_s, 0.0), D_mu)

    theta = jnp.where(var > var_fix, theta_a, theta_b)
    FL = jnp.where(var > var_fix, FL_x, FL_mu)  # per-row clamp uses the matching grid
    t_max = 2.0 ** (wl - FL - 1) - 2.0 ** (-FL)
    return jnp.clip(theta, -(2.0 ** (wl - FL - 1)), t_max)
