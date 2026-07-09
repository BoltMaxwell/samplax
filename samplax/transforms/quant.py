"""Low-precision arithmetic simulation in pure JAX (a drop-in for the slice of
QPyTorch this project uses).

Implements the three number formats the paper needs, each with nearest and
stochastic rounding:

* ``fixed_point_quantize(x, wl, fl, rounding, key)`` -- fixed point, ``wl`` total
  bits, ``fl`` fractional bits. Step ``2**-fl``; representable range
  ``[-2**(wl-fl-1), 2**(wl-fl-1) - 2**-fl]``.
* ``block_quantize(x, wl, rounding, key, dim)`` -- block floating point: a shared
  exponent (from the block max) with ``wl``-bit signed mantissa. ``dim=None`` shares
  one exponent across the whole tensor; ``dim=0`` shares per row.
* ``float_quantize(x, exp, man, rounding, key)`` -- low-precision float with ``exp``
  exponent / ``man`` mantissa bits.

Stochastic rounding takes an explicit ``jax.random`` key (round ``x`` up with
probability equal to its fractional distance, so ``E[q(x)] = x``). All functions are
jittable with ``wl/fl/exp/man`` as static Python ints, and expose a straight-through
gradient (identity) via :func:`straight_through`.

Semantics follow ``models/quantizer.py`` (block) and QPyTorch's fixed/float formats;
the cluster cross-check test asserts agreement with the real ``qtorch``.
"""

from typing import NamedTuple, Optional

from functools import partial

import jax
import jax.numpy as jnp

_EBIT = 8  # exponent bits for the block-FP shared exponent (QPyTorch default)


# --- number-format specs (names mirror qtorch for familiarity) ------------------
class FixedPoint(NamedTuple):
    wl: int
    fl: int


class BlockFloatingPoint(NamedTuple):
    wl: int
    dim: Optional[int] = None


class FloatingPoint(NamedTuple):
    exp: int
    man: int


def straight_through(x, xq):
    """Return ``xq`` on the forward pass, identity gradient on the backward pass."""
    return x + jax.lax.stop_gradient(xq - x)


def _round_int(t, rounding, key):
    """Round to integers: nearest, or stochastic (floor(t + U[0,1)))."""
    if rounding == "nearest":
        return jnp.round(t)
    if rounding == "stochastic":
        if key is None:
            raise ValueError("stochastic rounding requires a random key")
        return jnp.floor(t + jax.random.uniform(key, t.shape, dtype=t.dtype))
    raise ValueError(f"invalid rounding {rounding!r}")


def fixed_point_quantize(x, wl, fl, rounding="stochastic", key=None):
    sigma = 2.0 ** (-fl)
    t_min = -(2.0 ** (wl - fl - 1))
    t_max = 2.0 ** (wl - fl - 1) - sigma
    q = _round_int(x / sigma, rounding, key) * sigma
    return jnp.clip(q, t_min, t_max)


def block_quantize(x, wl, rounding="stochastic", key=None, dim=None):
    if dim is None:
        max_entry = jnp.max(jnp.abs(x))
        max_exp = jnp.floor(jnp.log2(jnp.where(max_entry == 0, 1.0, max_entry)))
    else:
        # shared exponent per slice along `dim` (paper uses dim=0)
        moved = jnp.moveaxis(x, dim, 0)
        flat = moved.reshape(moved.shape[0], -1)
        me = jnp.max(jnp.abs(flat), axis=1)
        max_exp = jnp.floor(jnp.log2(jnp.where(me == 0, 1.0, me)))
        shape = [1] * x.ndim
        shape[dim] = x.shape[dim]
        max_exp = max_exp.reshape(shape)
    max_exp = jnp.clip(max_exp, -(2 ** (_EBIT - 1)), 2 ** (_EBIT - 1) - 1)

    scale = 2.0 ** (-max_exp + (wl - 2))
    i = _round_int(x * scale, rounding, key)
    i = jnp.clip(i, -(2.0 ** (wl - 1)), 2.0 ** (wl - 1) - 1)
    q = i * 2.0 ** (max_exp - (wl - 2))
    return jnp.where(jnp.max(jnp.abs(x)) == 0, x, q)


def float_quantize(x, exp, man, rounding="nearest", key=None):
    if rounding == "nearest":
        return jax.lax.reduce_precision(x, exponent_bits=exp, mantissa_bits=man)
    if rounding == "stochastic":
        if key is None:
            raise ValueError("stochastic rounding requires a random key")
        # Round-to-nearest neighbours at the target mantissa precision, then pick the
        # upper neighbour with probability set by the fractional distance. The ULP is
        # 2**(floor(log2|x|) - man).
        ax = jnp.abs(x)
        e = jnp.floor(jnp.log2(jnp.where(ax == 0, 1.0, ax)))
        ulp = 2.0 ** (e - man)
        lower = jnp.floor(x / ulp) * ulp
        frac = jnp.where(ulp == 0, 0.0, (x - lower) / ulp)
        up = jax.random.uniform(key, x.shape, dtype=x.dtype) < frac
        q = jnp.where(up, lower + ulp, lower)
        return jnp.where(ax == 0, x, q)
    raise ValueError(f"invalid rounding {rounding!r}")


def make_quantizer(number, rounding="stochastic"):
    """Return ``quant(x, key=None) -> x_q`` for a number-format spec -- mirrors
    ``qtorch.quant.quantizer(forward_number=..., forward_rounding=...)``."""
    if isinstance(number, FixedPoint):
        return lambda x, key=None: fixed_point_quantize(x, number.wl, number.fl, rounding, key)
    if isinstance(number, BlockFloatingPoint):
        return lambda x, key=None: block_quantize(x, number.wl, rounding, key, number.dim)
    if isinstance(number, FloatingPoint):
        return lambda x, key=None: float_quantize(x, number.exp, number.man, rounding, key)
    raise TypeError(f"unknown number format {number!r}")


# --- fully low-precision layers: activation (forward) + error (backward) quant ---
@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def lp_quant(x, key, wl_act, wl_err, rounding):
    """Block-quantize an activation on the forward pass to ``wl_act`` bits, and the
    gradient (error) on the backward pass to ``wl_err`` bits (port of
    ``models/quantizer.py:BlockRounding``). ``wl=-1`` means full precision on that pass.
    Whole-tensor shared exponent (per example under vmap). ``key`` seeds the stochastic
    rounding; forward and backward use independent split keys."""
    k_act, _ = jax.random.split(key)
    return block_quantize(x, wl_act, rounding, k_act, dim=None) if wl_act != -1 else x


def _lp_quant_fwd(x, key, wl_act, wl_err, rounding):
    k_act, k_err = jax.random.split(key)
    y = block_quantize(x, wl_act, rounding, k_act, dim=None) if wl_act != -1 else x
    return y, k_err  # stash the backward key


def _lp_quant_bwd(wl_act, wl_err, rounding, k_err, g):
    gx = block_quantize(g, wl_err, rounding, k_err, dim=None) if wl_err != -1 else g
    return (gx, None)  # cotangents for (x, key); key gets none


lp_quant.defvjp(_lp_quant_fwd, _lp_quant_bwd)
