"""Low-precision SGLD (Zhang, Wilson, De Sa, ICML 2022).

Provenance: vendored from low-precision-sgld-jax ``optim_lp.make_lp_update``
(quantizers cross-checked bit-for-bit against QPyTorch; VC verified on the
Gaussian toy and CIFAR).

Interface note: unlike the core kernels, this sampler keeps the original's
conventions because quantization grids are scale-sensitive: ``grad`` is the
DESCENT gradient of the *mean* loss (as produced by standard training code),
``weight_decay`` is applied inside the update (after gradient quantization,
exactly as the original), ``datasize`` converts the mean-loss scale to the
posterior scale for the injected noise (var = 2 lr T / datasize), and ``lr``
is the mean-loss learning rate.

Variants: ``sgldlp_f`` (full-precision accumulator, quantized forward pass),
``naive`` (low-precision accumulator, biased), ``vc`` (low-precision
accumulator with variance-corrected quantization — the paper's method).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .quant import block_quantize, fixed_point_quantize
from .vc import Q_vc, Q_vc_block


class LPSGLDState(NamedTuple):
    position: jax.Array  # the accumulator (low-precision for -L variants)


class LPKernel(NamedTuple):
    init: Callable
    step: Callable
    forward_quant: Callable  # forward_quant(key, position) -> weights for fwd pass


def _tree_map_keyed(fn, tree, key):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef, [fn(x, k) for x, k in zip(leaves, keys)])


def _tree_gaussian(key, tree, std):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    noise = [std * jax.random.normal(k, x.shape, x.dtype)
             for x, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, noise)


def lp_sgld(variant, wl, fl, *, datasize, weight_decay=0.0, number="fixed"):
    """Build a low-precision SGLD :class:`LPKernel`.

    ``step(key, state, grad, lr, temperature=1.0)`` with ``grad`` the descent
    mean-loss gradient (see module docstring).
    """
    if number == "fixed":
        def quant(x, k):
            return fixed_point_quantize(x, wl, fl, "stochastic", k)

        def vc_quant(k, x, var):
            return Q_vc(k, x, var, wl, fl)
    elif number == "block":
        def quant(x, k):
            return block_quantize(x, wl, "stochastic", k, dim=0)

        def vc_quant(k, x, var):
            return Q_vc_block(k, x, var, wl)
    else:
        raise ValueError(f"unknown number format {number!r}")

    def init(key, position):
        if variant in ("naive", "vc"):
            position = _tree_map_keyed(quant, position, key)
        return LPSGLDState(position)

    def forward_quant(key, state):
        if variant == "sgldlp_f":
            return _tree_map_keyed(quant, state.position, key)
        return state.position

    def step(key, state, grad, lr, temperature=1.0):
        kg, kn, kw = jax.random.split(key, 3)
        params = state.position
        qgrads = _tree_map_keyed(quant, grad, kg)
        var = 2.0 * lr * temperature / datasize

        def gd(p, g):
            return p - lr * (g + weight_decay * p)

        if variant in ("sgldlp_f", "naive"):
            stepped = jax.tree_util.tree_map(gd, params, qgrads)
            gn = _tree_gaussian(kn, params, jnp.sqrt(var))
            stepped = jax.tree_util.tree_map(lambda p, n: p + n, stepped, gn)
            out = stepped if variant == "sgldlp_f" else _tree_map_keyed(quant, stepped, kw)
            return LPSGLDState(out)
        if variant == "vc":
            mu = jax.tree_util.tree_map(gd, params, qgrads)
            return LPSGLDState(
                _tree_map_keyed(lambda x, k: vc_quant(k, x, var), mu, kw))
        raise ValueError(f"unknown variant {variant!r}")

    return LPKernel(init, step, forward_quant)
