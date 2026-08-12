"""Adapters for the ift-sde repo's two sampler seams.

Family-B seam (``methods/nsvi``, ``methods/npsgld`` engines):
:func:`run_sgmcmc` has the same keyword-only signature as ``run_nsvi`` /
``run_npsgld`` — ``(rng_key, *, d_w, d_theta, d_x0, log_likelihood_fn,
energy_fn, config)`` — and returns a result object with ``.samples``
(``z/w/theta/x0_samples``) and ``.history``, so it drops into the runner
dispatch tables unchanged. It samples the *relaxed joint posterior*

    log p(z) = log_likelihood(w, theta, x0) - energy(w, theta, x0)
               + log N(theta; 0, theta_prior_std) + log N(x0; 0, x0_prior_std)

with any samplax kernel, schedule, and preconditioner. ``log_likelihood_fn``
follows ift-sde's 2026-07-14 three-argument contract, ``log_likelihood_fn(w,
theta, x0)`` — ``x0`` must actually reach the likelihood (the relaxation
target couples them through ``energy_fn`` only when the likelihood is itself
``x0``-independent, which is no longer assumed here).

NOTE: unlike NPSGLD it does **not** estimate the NIFF prior partition
gradient by default; for calibration targets where log Z(theta, x0) varies,
supply a :class:`Correction` (e.g. ift-sde's auxiliary-chain estimator) and
its gradient is added to the drift — that is the intended mix-and-match:
their nesting, samplax's kernels. A ``Correction`` is a stateful
``(init, step)`` pair rather than a bare gradient callback because the
motivating use case is a persistent-PCD-style nested aux chain: the aux
chain's own position/preconditioner state must persist and warm-start
across outer steps, and a stateless hook cannot express that persistence
under ``lax.scan`` (there is nowhere to carry it). Each outer chain gets its
own correction state, threaded through the scan carry alongside the kernel
state and returned in ``result.final_state["correction"]``. For state
estimation (theta fixed) the partition term is constant and the default
(correction-free) target is exact.

Sampler configuration (:class:`SGMCMCConfig`) mirrors ift-sde's engines:
``init_mean`` (default zeros) offsets the chain-init Gaussian, ``schedule``
adds ``"exponential"`` (geometric interpolation to ``step_size_final``,
matching ift-sde's NPSGLD decay) alongside ``"constant"``/``"cyclical"``/
``"polynomial"``. Post-burn-in chunk-ends are kept as samples only when the
schedule marks that chunk's final step as a sampling step (``do_sample``);
for ``"cyclical"`` this means chunk-ends that fall in an exploration phase
(temperature zeroed) are dropped, so a cyclical run keeps strictly fewer
than ``(iterations - burn_in) // thinning`` samples -- the other schedules
are always ``do_sample=True`` and keep the full count. Gradients/positions
are sanitized exactly as
``methods/npsgld/npsgld.py`` does: gradients are NaN-to-zero'd and clipped
elementwise to ``grad_clip`` before the kernel step; the post-step position
is reverted to its pre-step value wherever it turned non-finite and then
clipped elementwise to ``state_clip`` (kernel-state fields other than
``position`` — momentum, preconditioner accumulators — are left untouched).

AMAGOLD M-H kernel (``kernel="amagold"``) accepts `amagold_dt`, `amagold_nstep`,
and `amagold_C` config; M-H acceptance data flow differs from SGLD/SGHMC:
``history["accept_rate"]`` is appended every chunk (not gated by ``trace_every``),
so its length differs from ``history["step"]``/``history["log_posterior"]``; also,
``history["accept_rate"]`` is pooled over chains+time per chunk (scalar), while
``final_state["accept_rate"]`` is per-chain (shape ``(chains,)``). Requires
``correction=None`` (M-H test needs the true energy) and ``dt`` (leapfrog
stepsize, required).

Nesting-I seam (``experiments/calibration/param_loop.py``):
:func:`make_field_sampler` builds the inner-field-sampler boundary callable
``f(field_state, param_state, key) -> field_chain`` from a samplax kernel and
a per-(field, param) gradient callback.

x64: ift-sde enables ``jax_enable_x64``; these adapters follow the dtype of
the inputs, so they run in float64 there and float32 elsewhere.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ..kernels.amagold import amagold_tracked
from ..kernels.pcn_gibbs import pcn_gibbs
from ..kernels.sghmc import sghmc
from ..kernels.sgld import sgld
from ..preconditioners import identity, rmsprop
from ..schedules import constant, cyclical, exponential, polynomial


class Correction(NamedTuple):
    """Stateful nested-chain correction, added to the outer chain's drift.

    ``init(key, z0) -> cstate`` builds one chain's correction state from
    that chain's initial position ``z0`` (shape ``(d_z,)``, single chain —
    :func:`run_sgmcmc` vmaps it over chains).

    ``step(key, z, cstate) -> (grad_z, cstate)`` advances the correction one
    outer step at the current position ``z`` (``(d_z,)``) and returns the
    additive gradient contribution (``(d_z,)``) plus the updated state.
    """

    init: Callable
    step: Callable


@dataclass(frozen=True)
class SGMCMCConfig:
    kernel: str = "sghmc"            # "sgld" | "sghmc"
    alpha: float = 0.1               # sghmc friction
    preconditioner: str = "identity"  # "identity" | "rmsprop"
    rmsprop_beta: float = 0.99        # rmsprop EMA decay (used when preconditioner="rmsprop")
    rmsprop_eps: float = 1e-5         # rmsprop damping   (ift-sde NPSGLDConfig.delta analog)
    iterations: int = 20_000
    chains: int = 4
    burn_in: int = 5_000
    thinning: int = 10
    step_size: float = 2.0e-4
    step_size_final: Optional[float] = None
    schedule: str = "constant"       # "constant" | "cyclical" | "polynomial" | "exponential"
    num_cycles: int = 4              # cyclical
    exploration_ratio: float = 0.25  # cyclical
    poly_b: float = 1.0              # polynomial: step_size * (b + t)^-gamma
    poly_gamma: float = 0.55
    temperature: float = 1.0
    theta_prior_std: float = 1.0
    x0_prior_std: float = 1.0
    init_mean: Optional[tuple] = None
    init_std: float = 0.5
    grad_clip: float = 1.0e3
    state_clip: float = 1.0e6
    trace_every: int = 1_000
    amagold_dt: Optional[float] = None  # AMAGOLD leapfrog step (kernel="amagold"); w-block step when block dts are set
    amagold_nstep: int = 5               # AMAGOLD inner leapfrog steps per outer step
    amagold_C: float = 1.0               # AMAGOLD friction
    # Per-block leapfrog steps (diagonal mass matrix). When set, the kernel dt
    # becomes the (d_z,) vector [amagold_dt]*d_w + [amagold_dt_theta]*d_theta
    # + [amagold_dt_x0]*d_x0. This is the fix for stiff-w/mobile-theta targets
    # (whitened Duffing: one scalar dt froze theta — ift-sde
    # sampler_experimentation v5). amagold_dt_x0 defaults to amagold_dt_theta.
    amagold_dt_theta: Optional[float] = None
    amagold_dt_x0: Optional[float] = None
    # Scalar thermostat step for the friction/noise discretization (see the
    # kernel docstring; any positive scalar is exact). Default: amagold_dt.
    amagold_dt_friction: Optional[float] = None
    # pCN-within-Gibbs (kernel="pcn-gibbs"): gradient-free blocked M-H for
    # whitened targets (energy must be the isotropic w-prior ||w||^2/2 — probed
    # at setup). pcn_mh_scales (length d_theta + d_x0) overrides the scalar
    # pcn_mh_scale for per-coordinate proposal widths on the (theta, x0) block.
    pcn_beta: float = 0.1            # pCN proposal mixing in (0, 1]
    pcn_n_pcn: int = 5               # w-updates per sweep
    pcn_n_mh: int = 5                # (theta, x0)-updates per sweep
    pcn_mh_scale: float = 0.02       # scalar RW proposal scale for (theta, x0)
    pcn_mh_scales: Optional[tuple] = None  # per-coordinate override
    # Ridge-scale (interweaving) move: index of the log-scale coordinate in
    # the (theta, x0) block that multiplies the whitened field (e.g. log sigma
    # for OU raw index 2, log sigma_v for the Duffing noncentered variant
    # index 3). None disables. See kernels.pcn_gibbs for the exactness
    # argument; it is what traverses the sigma-w amplitude ridge.
    pcn_scale_index: Optional[int] = None
    pcn_scale_step: float = 0.2
    pcn_n_scale: int = 2


@dataclass
class SGMCMCResult:
    samples: dict
    history: dict
    final_state: dict
    config: SGMCMCConfig


def _split_z(z, d_w, d_theta):
    return z[..., :d_w], z[..., d_w:d_w + d_theta], z[..., d_w + d_theta:]


def _normal_logpdf(x, std):
    return jnp.sum(-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(std) - 0.5 * (x / std) ** 2)


def _sanitize_grad(g, clip):
    return jnp.clip(jnp.nan_to_num(g, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)


def _sanitize_position(candidate, old, clip):
    return jnp.clip(jnp.where(jnp.isfinite(candidate), candidate, old), -clip, clip)


def _assemble_samples(kept, cfg, d_w, d_theta, d_x0, d_z):
    z_samples = np.stack(kept, axis=1) if kept else np.zeros((cfg.chains, 0, d_z))
    w_s, th_s, x0_s = (z_samples[..., :d_w],
                       z_samples[..., d_w:d_w + d_theta],
                       z_samples[..., d_w + d_theta:])
    return z_samples, w_s, th_s, x0_s


def run_sgmcmc(rng_key, *, d_w, d_theta, d_x0, log_likelihood_fn, energy_fn,
               config: Optional[SGMCMCConfig] = None,
               log_prior_fn: Optional[Callable] = None,
               correction: Optional[Correction] = None) -> SGMCMCResult:
    cfg = config or SGMCMCConfig()
    if cfg.iterations % cfg.thinning != 0:
        raise ValueError(
            f"iterations ({cfg.iterations}) must be divisible by thinning "
            f"({cfg.thinning}): the chunked scan would silently drop the "
            f"remainder steps (and an exponential schedule would never reach "
            f"step_size_final)")
    d_z = d_w + d_theta + d_x0

    if (cfg.kernel not in ("amagold", "pcn-gibbs") and cfg.schedule == "exponential"
            and cfg.step_size_final is None):
        raise ValueError("schedule='exponential' requires step_size_final to be set")

    if cfg.init_mean is None:
        init_mean = jnp.zeros((d_z,))
    else:
        init_mean = jnp.asarray(cfg.init_mean).reshape((-1,))
        if init_mean.shape[0] != d_z:
            raise ValueError(
                f"init_mean has length {init_mean.shape[0]}, expected {d_z}")

    def log_posterior(z):
        w, theta, x0 = _split_z(z, d_w, d_theta)
        lp = log_likelihood_fn(w, theta, x0) - energy_fn(w, theta, x0)
        if log_prior_fn is not None:
            return lp + log_prior_fn(z)
        return (lp + _normal_logpdf(theta, jnp.asarray(cfg.theta_prior_std))
                + _normal_logpdf(x0, jnp.asarray(cfg.x0_prior_std)))

    grad_fn = jax.grad(log_posterior)

    if cfg.kernel == "amagold":
        return _run_amagold(rng_key, cfg, d_w, d_theta, d_x0, d_z,
                            init_mean, log_posterior, grad_fn, correction)

    if cfg.kernel == "pcn-gibbs":
        return _run_pcn_gibbs(rng_key, cfg, d_w, d_theta, d_x0, d_z, init_mean,
                              log_likelihood_fn, energy_fn, log_prior_fn,
                              correction)

    if cfg.preconditioner == "identity":
        precond = identity()
    elif cfg.preconditioner == "rmsprop":
        precond = rmsprop(beta=cfg.rmsprop_beta, eps=cfg.rmsprop_eps)
    else:
        raise ValueError(f"unknown preconditioner {cfg.preconditioner!r}")
    kernel = (sgld(preconditioner=precond) if cfg.kernel == "sgld"
              else sghmc(alpha=cfg.alpha, preconditioner=precond))
    schedule = {
        "constant": lambda: constant(cfg.step_size),
        "cyclical": lambda: cyclical(cfg.iterations, cfg.num_cycles,
                                     cfg.step_size, cfg.exploration_ratio),
        "polynomial": lambda: polynomial(cfg.step_size, cfg.poly_b, cfg.poly_gamma),
        "exponential": lambda: exponential(cfg.step_size, cfg.step_size_final,
                                           cfg.iterations),
    }[cfg.schedule]()

    key_init, key_cinit, key_run = jax.random.split(jnp.asarray(rng_key), 3)
    z0 = init_mean[None, :] + cfg.init_std * jax.random.normal(key_init, (cfg.chains, d_z))
    states = jax.vmap(lambda z: kernel.init(key_init, z))(z0)
    if correction is not None:
        cinit_keys = jax.random.split(key_cinit, cfg.chains)
        cstates = jax.vmap(correction.init)(cinit_keys, z0)
    else:
        cstates = ()

    def one_step(carry, inp):
        states, cstates, t = carry
        keys = inp
        sched = schedule(t)

        def chain_step(key, state, cstate):
            pos_old = state.position
            g = grad_fn(pos_old)
            if correction is not None:
                k_corr, k_step = jax.random.split(key)
                g_corr, cstate = correction.step(k_corr, pos_old, cstate)
                g = g + g_corr
            else:
                k_step = key
            g = _sanitize_grad(g, cfg.grad_clip)
            temp = cfg.temperature * jnp.where(sched.do_sample, 1.0, 0.0)
            state = kernel.step(k_step, state, g, sched.step_size, temp)
            # Sanitize the position only: momentum / preconditioner accumulator
            # fields are left untouched, matching npsgld's _sanitize_state
            # (which only ever guards the sampled position vector).
            position = _sanitize_position(state.position, pos_old, cfg.state_clip)
            state = state._replace(position=position)
            return state, cstate

        states, cstates = jax.vmap(chain_step)(keys, states, cstates)
        lp = jax.vmap(lambda s: log_posterior(s.position))(states)
        return (states, cstates, t + 1), lp

    @jax.jit
    def run_chunk(states, cstates, t, keys):
        return jax.lax.scan(one_step, (states, cstates, t), keys)

    n_chunks = cfg.iterations // cfg.thinning
    kept, trace_t, trace_lp = [], [], []
    t = jnp.asarray(0)
    for c in range(n_chunks):
        key_run, sub = jax.random.split(key_run)
        keys = jax.random.split(sub, (cfg.thinning, cfg.chains))
        (states, cstates, t), lps = run_chunk(states, cstates, t, keys)
        step_now = (c + 1) * cfg.thinning
        # A cyclical schedule zeroes the temperature during its exploration
        # phase, so chunk-ends that land there are optimization iterates, not
        # posterior draws -- only keep a chunk when the schedule marks its
        # final step (0-indexed step_now - 1) as a sampling step. Constant /
        # exponential / polynomial schedules are always do_sample=True, so
        # this is a no-op for them.
        do_sample_final = bool(schedule(step_now - 1).do_sample)
        if step_now > cfg.burn_in and do_sample_final:
            kept.append(np.asarray(states.position))
        if step_now % cfg.trace_every == 0 or c == n_chunks - 1:
            trace_t.append(step_now)
            trace_lp.append(np.asarray(lps[-1]).tolist())

    z_samples, w_s, th_s, x0_s = _assemble_samples(kept, cfg, d_w, d_theta, d_x0, d_z)
    return SGMCMCResult(
        samples={"z_samples": z_samples, "w_samples": w_s,
                 "theta_samples": th_s, "x0_samples": x0_s},
        history={"step": trace_t, "log_posterior": trace_lp},
        final_state={"z": np.asarray(states.position),
                     "correction": jax.tree_util.tree_map(np.asarray, cstates)},
        config=cfg,
    )


def _run_amagold(rng_key, cfg, d_w, d_theta, d_x0, d_z, init_mean,
                  log_posterior, grad_fn, correction):
    """AMAGOLD driver: a separate scan/keep-mask path, not a per-step Kernel.

    AMAGOLD owns its own leapfrog loop and an amortized M-H test, so it has
    no schedule, no temperature, and (currently) no nested-Z correction --
    the M-H test needs the *true* energy, which a Correction's additive
    gradient contribution does not provide. Every post-burn-in chunk end is
    kept (there is no cyclical-style do_sample gate to consult).
    """
    if cfg.amagold_dt is None:
        raise ValueError("amagold requires amagold_dt")
    if correction is not None:
        raise ValueError(
            "AMAGOLD's M-H correction needs the true energy; it does not "
            "compose with a nested-Z correction -- use kernel='sgld'/"
            "'sghmc' for correction, or an energy that is "
            "(theta,x0)-independent")
    if cfg.preconditioner not in ("identity",):
        # Loud by doctrine: this flag was silently inert for a month and a
        # "preconditioned AMAGOLD" that never existed made it into results
        # (ift-sde sampler_experimentation v5). Per-block dts are the
        # supported anisotropy mechanism for this kernel.
        raise ValueError(
            f"amagold has no preconditioner (got {cfg.preconditioner!r}); "
            "use preconditioner='identity' and express anisotropy via "
            "amagold_dt_theta / amagold_dt_x0 (diagonal mass matrix)")
    if cfg.schedule != "constant":
        warnings.warn(
            "amagold ignores cfg.schedule: step size is fixed by "
            "amagold_dt and there is no temperature schedule under the "
            "M-H correction")

    def u_fn(z):
        return -log_posterior(z)

    def grad_u(key, z):
        return _sanitize_grad(-grad_fn(z), cfg.grad_clip)

    if cfg.amagold_dt_theta is not None or cfg.amagold_dt_x0 is not None:
        dt_theta = cfg.amagold_dt_theta if cfg.amagold_dt_theta is not None else cfg.amagold_dt
        dt_x0 = cfg.amagold_dt_x0 if cfg.amagold_dt_x0 is not None else dt_theta
        dt = jnp.concatenate([
            jnp.full((d_w,), cfg.amagold_dt, dtype=jnp.float64),
            jnp.full((d_theta,), dt_theta, dtype=jnp.float64),
            jnp.full((d_x0,), dt_x0, dtype=jnp.float64),
        ])
    else:
        dt = cfg.amagold_dt

    # Tracked form: the energy at the current position rides in the state, so
    # an outer step costs ONE log_posterior evaluation instead of three (the
    # kernel's own two, plus the trace pass that used to re-evaluate it here).
    # `lp` below is read straight off the state -- state.energy is exactly
    # u_fn(state.position), so -state.energy is log_posterior(position) with no
    # approximation. Safe because u_fn is fixed at build time and deterministic,
    # which the M-H test already requires (hence the correction=None guard above).
    kernel_init, kernel_step = amagold_tracked(
        u_fn, grad_u, dt=dt, nstep=cfg.amagold_nstep, C=cfg.amagold_C,
        dt_friction=cfg.amagold_dt_friction)

    key_init, key_run = jax.random.split(jnp.asarray(rng_key), 2)
    positions = init_mean[None, :] + cfg.init_std * jax.random.normal(
        key_init, (cfg.chains, d_z))
    states = jax.jit(jax.vmap(kernel_init))(positions)

    def one_step(carry, keys):
        states, accept_sum = carry
        states, accepted = jax.vmap(kernel_step)(keys, states)
        accept_sum = accept_sum + accepted.astype(accept_sum.dtype)
        return (states, accept_sum), (-states.energy, accepted)

    @jax.jit
    def run_chunk(states, accept_sum, keys):
        return jax.lax.scan(one_step, (states, accept_sum), keys)

    n_chunks = cfg.iterations // cfg.thinning
    kept, trace_t, trace_lp, accept_history = [], [], [], []
    accept_sum = jnp.zeros((cfg.chains,), dtype=positions.dtype)
    for c in range(n_chunks):
        key_run, sub = jax.random.split(key_run)
        keys = jax.random.split(sub, (cfg.thinning, cfg.chains))
        (states, accept_sum), (lps, accepted) = run_chunk(
            states, accept_sum, keys)
        step_now = (c + 1) * cfg.thinning
        # AMAGOLD always samples (no cyclical-style exploration phase): keep
        # every post-burn-in chunk end.
        if step_now > cfg.burn_in:
            kept.append(np.asarray(states.position))
        accept_history.append(float(np.asarray(accepted).mean()))
        if step_now % cfg.trace_every == 0 or c == n_chunks - 1:
            trace_t.append(step_now)
            trace_lp.append(np.asarray(lps[-1]).tolist())

    z_samples, w_s, th_s, x0_s = _assemble_samples(kept, cfg, d_w, d_theta, d_x0, d_z)
    accept_rate = np.asarray(accept_sum) / cfg.iterations
    return SGMCMCResult(
        samples={"z_samples": z_samples, "w_samples": w_s,
                 "theta_samples": th_s, "x0_samples": x0_s},
        history={"step": trace_t, "log_posterior": trace_lp,
                 "accept_rate": accept_history},
        final_state={"z": np.asarray(states.position), "accept_rate": accept_rate,
                     "correction": ()},
        config=cfg,
    )


def _run_pcn_gibbs(rng_key, cfg, d_w, d_theta, d_x0, d_z, init_mean,
                   log_likelihood_fn, energy_fn, log_prior_fn, correction):
    """pCN-within-Gibbs driver: gradient-free blocked M-H (kernels.pcn_gibbs).

    Valid ONLY for whitened targets whose energy is the isotropic w-prior
    ``||w||^2/2`` (+ a constant): the pCN move absorbs that prior into its
    proposal, so the sweep targets loglik + theta/x0 priors. The requirement
    is PROBED at setup (three random points) and violated energies raise.
    Like AMAGOLD: no schedule, no preconditioner, no nested correction.
    """
    if correction is not None:
        raise ValueError(
            "pcn-gibbs does not compose with a nested-Z correction; the "
            "whitened (constant-Z) targets it exists for do not need one")
    if cfg.preconditioner not in ("identity",):
        raise ValueError(
            f"pcn-gibbs has no preconditioner (got {cfg.preconditioner!r}); "
            "use preconditioner='identity' and set pcn_mh_scales for "
            "per-coordinate (theta, x0) proposal widths")
    if cfg.schedule != "constant":
        warnings.warn("pcn-gibbs ignores cfg.schedule: pcn_beta and "
                      "pcn_mh_scale(s) are fixed; there is no step schedule")
    if log_prior_fn is not None:
        raise ValueError(
            "pcn-gibbs builds its own target from theta_prior_std/x0_prior_std; "
            "a custom log_prior_fn is not supported (it could depend on w, "
            "which would break the pCN prior cancellation)")

    # Probe: energy must equal ||w||^2/2 + const (theta/x0-independent too).
    probe_key = jax.random.PRNGKey(0)
    consts = []
    for i in range(3):
        k1, k2, k3, probe_key = jax.random.split(jax.random.fold_in(probe_key, i), 4)
        w_p = jax.random.normal(k1, (max(d_w, 1),))[:d_w]
        th_p = jax.random.normal(k2, (max(d_theta, 1),))[:d_theta]
        x0_p = jax.random.normal(k3, (max(d_x0, 1),))[:d_x0]
        e = energy_fn(w_p, th_p, x0_p)
        consts.append(float(e - 0.5 * jnp.sum(w_p**2)))
    if max(consts) - min(consts) > 1e-6 * max(1.0, abs(consts[0])):
        raise ValueError(
            "pcn-gibbs requires energy_fn(w, theta, x0) == ||w||^2/2 + const "
            f"(whitened target); probe found varying residuals {consts}. Use "
            "kernel='sgld'/'sghmc' for non-whitened energies")

    def f(z):
        w, theta, x0 = _split_z(z, d_w, d_theta)
        return (log_likelihood_fn(w, theta, x0)
                + _normal_logpdf(theta, jnp.asarray(cfg.theta_prior_std))
                + _normal_logpdf(x0, jnp.asarray(cfg.x0_prior_std)))

    d_rest = d_theta + d_x0
    if cfg.pcn_mh_scales is not None:
        mh_scales = jnp.asarray(cfg.pcn_mh_scales, dtype=jnp.float64)
        if mh_scales.shape[0] != d_rest:
            raise ValueError(
                f"pcn_mh_scales has length {mh_scales.shape[0]}, expected {d_rest}")
    else:
        mh_scales = jnp.full((d_rest,), cfg.pcn_mh_scale, dtype=jnp.float64)

    kernel_step = pcn_gibbs(f, d_w=d_w, beta=cfg.pcn_beta, mh_scales=mh_scales,
                            n_pcn=cfg.pcn_n_pcn, n_mh=cfg.pcn_n_mh,
                            scale_index=cfg.pcn_scale_index,
                            scale_step=cfg.pcn_scale_step,
                            n_scale=cfg.pcn_n_scale)

    key_init, key_run = jax.random.split(jnp.asarray(rng_key), 2)
    positions = init_mean[None, :] + cfg.init_std * jax.random.normal(
        key_init, (cfg.chains, d_z))

    def one_step(carry, keys):
        positions, aw_sum, ar_sum, as_sum = carry
        new_positions, (aw, ar, a_s) = jax.vmap(kernel_step)(keys, positions)
        lp = jax.vmap(f)(new_positions)
        return (new_positions, aw_sum + aw, ar_sum + ar, as_sum + a_s), (lp, aw, ar, a_s)

    @jax.jit
    def run_chunk(positions, aw_sum, ar_sum, as_sum, keys):
        return jax.lax.scan(one_step, (positions, aw_sum, ar_sum, as_sum), keys)

    n_chunks = cfg.iterations // cfg.thinning
    kept, trace_t, trace_lp = [], [], []
    accept_w_hist, accept_r_hist, accept_s_hist = [], [], []
    aw_sum = jnp.zeros((cfg.chains,), dtype=positions.dtype)
    ar_sum = jnp.zeros((cfg.chains,), dtype=positions.dtype)
    as_sum = jnp.zeros((cfg.chains,), dtype=positions.dtype)
    for c in range(n_chunks):
        key_run, sub = jax.random.split(key_run)
        keys = jax.random.split(sub, (cfg.thinning, cfg.chains))
        (positions, aw_sum, ar_sum, as_sum), (lps, aws, ars, ass) = run_chunk(
            positions, aw_sum, ar_sum, as_sum, keys)
        step_now = (c + 1) * cfg.thinning
        if step_now > cfg.burn_in:
            kept.append(np.asarray(positions))
        accept_w_hist.append(float(np.asarray(aws).mean()))
        accept_r_hist.append(float(np.asarray(ars).mean()))
        accept_s_hist.append(float(np.asarray(ass).mean()))
        if step_now % cfg.trace_every == 0 or c == n_chunks - 1:
            trace_t.append(step_now)
            trace_lp.append(np.asarray(lps[-1]).tolist())

    z_samples, w_s, th_s, x0_s = _assemble_samples(kept, cfg, d_w, d_theta, d_x0, d_z)
    return SGMCMCResult(
        samples={"z_samples": z_samples, "w_samples": w_s,
                 "theta_samples": th_s, "x0_samples": x0_s},
        history={"step": trace_t, "log_posterior": trace_lp,
                 "accept_w": accept_w_hist, "accept_theta": accept_r_hist,
                 "accept_scale": accept_s_hist},
        final_state={"z": np.asarray(positions),
                     "accept_rate_w": np.asarray(aw_sum) / cfg.iterations,
                     "accept_rate_theta": np.asarray(ar_sum) / cfg.iterations,
                     "accept_rate_scale": np.asarray(as_sum) / cfg.iterations,
                     "correction": ()},
        config=cfg,
    )


def make_field_sampler(kernel, schedule, *, num_steps, keep_every=1, burn=0,
                       temperature=1.0, grad_clip=None):
    """Build the param_loop.py inner-sampler boundary callable.

    Given ``grad_fn(field, param_state, key) -> ascent-gradient pytree``,
    returns ``f(field_state, param_state, key) -> chain`` where ``chain`` has
    a leading sample axis, as ``sgld_parameter_sampler`` expects. Usage::

        inner = make_field_sampler(samplax.sgld(), samplax.constant(1e-4),
                                   num_steps=..., keep_every=..., burn=...)
        posterior_field_sampler_fn = lambda f, p, k: inner(grad_fn, f, p, k)
    """

    def run(grad_fn, field_state, param_state, key):
        state = kernel.init(key, field_state)

        def body(carry, inp):
            state, t = carry
            subkey = inp
            k_grad, k_step = jax.random.split(subkey)
            g = grad_fn(state.position, param_state, k_grad)
            if grad_clip is not None:
                flat = ravel_pytree(g)[0]
                norm = jnp.sqrt(jnp.sum(flat * flat))
                factor = jnp.where(norm > grad_clip, grad_clip / norm, 1.0)
                g = jax.tree_util.tree_map(lambda x: factor * x, g)
            sched = schedule(t)
            temp = temperature * jnp.where(sched.do_sample, 1.0, 0.0)
            state = kernel.step(k_step, state, g, sched.step_size, temp)
            return (state, t + 1), state.position

        (_, _), chain = jax.lax.scan(body, (state, jnp.asarray(0)),
                                     jax.random.split(key, num_steps))
        return jax.tree_util.tree_map(lambda x: x[burn::keep_every], chain)

    return run
