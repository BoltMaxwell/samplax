"""Bit-equivalence of samplax kernels against the source repos they were
vendored from. These tests import the sibling repos directly and are skipped
when a sibling is not present (CI without the four repos still runs
test_correctness.py).

Run:  JAX_PLATFORMS=cpu python tests/test_vendored_equivalence.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import samplax

DOCS = os.path.expanduser("~/Documents")
SIBLINGS = {
    "csgmcmc": os.path.join(DOCS, "csgmcmc-jax"),
    "lpsgld": os.path.join(DOCS, "low-precision-sgld-jax"),
    "amagold": os.path.join(DOCS, "amagold-jax"),
    "sghmc": os.path.join(DOCS, "SGHMC-jax"),
}
for path in SIBLINGS.values():
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _have(name):
    return os.path.isdir(SIBLINGS[name])


def _tree_close(a, b, **kw):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), **kw)


POSITION = {"a": jnp.arange(6.0).reshape(2, 3) / 7.0, "b": jnp.ones(4) * 0.3}
GRAD = jax.tree_util.tree_map(lambda t: 0.1 * t - 0.05, POSITION)


def test_sgld_matches_csgmcmc():
    if not _have("csgmcmc"):
        print("  (skipped)"); return
    from csgmcmc_jax.samplers import jax_backend as ref

    key = jax.random.key(0)
    ref_step = ref.make_sgld()
    out_ref = ref_step(key, POSITION, GRAD, 1e-3, 0.7)
    kernel = samplax.sgld()
    out = kernel.step(key, kernel.init(key, POSITION), GRAD, 1e-3, 0.7)
    _tree_close(out.position, out_ref, rtol=1e-7, atol=0)


def test_sghmc_matches_csgmcmc():
    if not _have("csgmcmc"):
        print("  (skipped)"); return
    from csgmcmc_jax.samplers import jax_backend as ref

    key = jax.random.key(1)
    mom = jax.tree_util.tree_map(lambda t: 0.01 * t, POSITION)
    ref_step = ref.make_sghmc(alpha=0.9)
    pos_ref, mom_ref = ref_step(key, POSITION, mom, GRAD, 1e-3, 0.7)
    kernel = samplax.sghmc(alpha=0.9)
    state = samplax.SGHMCState(POSITION, mom, ())
    out = kernel.step(key, state, GRAD, 1e-3, 0.7)
    _tree_close(out.position, pos_ref, rtol=1e-7, atol=0)
    _tree_close(out.momentum, mom_ref, rtol=1e-7, atol=0)


def test_sghmc_vhat_matches_sghmc_jax():
    if not _have("sghmc"):
        print("  (skipped)"); return
    from sghmc_jax.samplers import jax_backend as ref

    key = jax.random.key(2)
    eta, alpha, v_hat = 0.01, 0.05, 1.0
    init_ref, step_ref = ref.make_sghmc(eta, alpha, v_hat)
    mom = init_ref(key, POSITION)
    # sghmc-jax uses the DESCENT gradient of U; samplax the ascent gradient
    grad_u = jax.tree_util.tree_map(jnp.negative, GRAD)
    pos_ref, mom_ref = step_ref(key, POSITION, mom, grad_u)
    kernel = samplax.sghmc(alpha=alpha, v_hat=v_hat)
    out = kernel.step(key, samplax.SGHMCState(POSITION, mom, ()), GRAD, eta, 1.0)
    _tree_close(out.position, pos_ref, rtol=1e-6, atol=1e-9)
    _tree_close(out.momentum, mom_ref, rtol=1e-6, atol=1e-9)


def test_cyclical_schedule_matches_csgmcmc():
    if not _have("csgmcmc"):
        print("  (skipped)"); return
    from csgmcmc_jax.samplers.cyclical import build_schedule as ref_build

    ref = ref_build(1000, 4, 0.5, 0.25)
    ours = samplax.cyclical(1000, 4, 0.5, 0.25)
    for t in (0, 62, 63, 249, 250, 999):
        a, b = ours(t), ref(t)
        np.testing.assert_allclose(float(a.step_size), float(b.step_size), rtol=1e-7)
        assert bool(a.do_sample) == bool(b.do_sample), t


def test_lp_sgld_matches_lpsgld():
    if not _have("lpsgld"):
        print("  (skipped)"); return
    from lpsgld_jax.optim_lp import make_lp_update

    key = jax.random.key(3)
    for variant in ("sgldlp_f", "naive", "vc"):
        fq_ref, up_ref = make_lp_update(variant, 8, 6, weight_decay=5e-4,
                                        temperature=1.0, datasize=100)
        # lpsgld's grads are descent mean-loss gradients
        out_ref = up_ref(key, POSITION, GRAD, 0.1)
        kernel = samplax.lp_sgld(variant, 8, 6, datasize=100, weight_decay=5e-4)
        out = kernel.step(key, samplax.transforms.lp_sgld.LPSGLDState(POSITION),
                          GRAD, 0.1, 1.0)
        _tree_close(out.position, out_ref, rtol=1e-7, atol=0)
        fq = kernel.forward_quant(key, samplax.transforms.lp_sgld.LPSGLDState(POSITION))
        _tree_close(fq, fq_ref(key, POSITION), rtol=1e-7, atol=0)


def test_amagold_matches_amagold_jax():
    if not _have("amagold"):
        print("  (skipped)"); return
    from amagold_jax.samplers import amagold_kernel as ref_kernel

    u = lambda x: 0.5 * x**2
    grad = lambda key, x: x + 0.1 * jax.random.normal(key)
    ref_step = ref_kernel(u, grad, dt=0.25, nstep=10, C=0.5, mh=True)
    our_step = samplax.amagold(u, grad, dt=0.25, nstep=10, C=0.5, mh=True)
    x = jnp.asarray(0.3)
    for seed in range(5):
        key = jax.random.key(seed)
        xr, ar = ref_step(key, x)
        xo, ao = our_step(key, x)
        np.testing.assert_allclose(float(xo), float(xr), rtol=1e-7)
        assert bool(ao) == bool(ar)


def test_hmc_matches_sghmc_jax():
    if not _have("sghmc"):
        print("  (skipped)"); return
    from sghmc_jax.samplers import jax_backend as ref

    u = lambda x: 0.5 * jnp.sum(x**2)
    grad = lambda key, x: x
    ref_step = ref.hmc_kernel(u, grad, dt=0.1, nstep=10, m=1.0, mh=True)
    our_step = samplax.hmc(u, grad, dt=0.1, nstep=10, m=1.0, mh=True)
    x = jnp.asarray(0.5)
    for seed in range(5):
        key = jax.random.key(seed)
        np.testing.assert_allclose(float(our_step(key, x)), float(ref_step(key, x)),
                                   rtol=1e-7)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
