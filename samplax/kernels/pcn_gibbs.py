"""pCN-within-Gibbs: blocked, gradient-free M-H for whitened latent-field models.

Target family: pi(z) ∝ exp(f(z)) · N(w; 0, I) with z = [w | rest], where
``rest`` is a small parameter block (theta, x0) whose own prior is INCLUDED in
``f``, and w is a high-dimensional whitened latent with an exact standard-normal
prior. This is precisely the structure of the whitened IFT targets
(``samplax.integrations.ift_sde``: log_likelihood + theta/x0 priors in f;
energy = ||w||^2/2 is the w prior).

Two alternating exact conditional M-H moves per sweep:

- **w | rest — preconditioned Crank-Nicolson.** Proposal
  ``w' = sqrt(1 - beta^2) w + beta xi``, xi ~ N(0, I), is reversible with
  respect to the N(0, I) prior, so the prior CANCELS in the acceptance ratio:
  ``a = exp(f(z') - f(z))``. Acceptance is therefore dimension-robust in d_w
  (the classic Cotter-Roberts-Stuart-White property) — no leapfrog, no
  stability ceiling from the stiff w block.
- **rest | w — Gaussian random-walk M-H** with per-coordinate ``mh_scales``,
  acceptance ``exp(f(z') - f(z))`` (f carries the rest-prior).

Both moves target their exact conditionals, so the sweep composition is
pi-invariant for ANY beta in (0, 1] and any scales — tuning affects mixing
only, never correctness. No gradients are evaluated anywhere.

Why this exists (ift-sde sampler_experimentation v5/v9): joint-trajectory
kernels (AMAGOLD, scalar or mass-matrix) fail on these targets because within
a trajectory theta moves while w cannot follow the shifting w-conditional —
every proposal climbs energy along the stiff cross-block direction. Blocking
removes the cross-block coherence requirement entirely: w moves at fixed
(theta, x0), (theta, x0) moves at fixed w.
"""

import jax
import jax.numpy as jnp


def pcn_gibbs(log_target_no_wprior, *, d_w, beta, mh_scales, n_pcn=1, n_mh=1,
              scale_index=None, scale_step=0.1, n_scale=1):
    """Build one Gibbs sweep. Returns ``step(key, z) -> (z_new, (aw, ar, as_))``
    with the accepted fractions of the sweep's n_pcn w-updates, n_mh
    rest-updates, and (when enabled) n_scale ridge-scale moves.

    ``log_target_no_wprior(z) -> scalar`` must EXCLUDE the N(0, I) prior on
    ``z[:d_w]`` (it is absorbed by the pCN proposal) and INCLUDE everything
    else (likelihood + rest-block priors). ``mh_scales`` has shape (d_rest,);
    a zero-length rest block skips the rest updates entirely.

    **Ridge-scale move** (``scale_index`` = index INTO THE REST BLOCK of a
    log-scale coordinate, e.g. log sigma): the interweaving-style proposal

        eps ~ N(0, scale_step^2);  w' = w e^{-eps};  rest[i]' = rest[i] + eps

    with log-acceptance ``f(z') - f(z) - (e^{-2 eps} - 1) ||w||^2 / 2
    - d_w eps`` (w-prior ratio + the |det| = e^{-d_w eps} Jacobian of the
    deterministic map). It is a valid M-H move for ANY target; its power is on
    whitened amplitude ridges, where the likelihood depends on w only through
    ``exp(rest[i]) * (linear map of w)`` and is therefore EXACTLY invariant
    along the move — it traverses the sigma-w ridge freely no matter how
    sharp the likelihood is transverse to it, which is precisely where plain
    conditional updates throttle (accept -> 0 as the conditionals tighten).
    Requires the FULL w block to be scaled by that sigma (true for the
    whitened OU and velocity-noise Duffing targets).
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"pcn beta must be in (0, 1], got {beta}")
    if n_pcn < 1 or n_mh < 1:
        raise ValueError("n_pcn and n_mh must be >= 1")
    mh_scales = jnp.asarray(mh_scales)
    has_rest = mh_scales.shape[0] > 0
    if scale_index is not None:
        if not has_rest or not (0 <= scale_index < mh_scales.shape[0]):
            raise ValueError(f"scale_index {scale_index} outside the rest block")
        if n_scale < 1:
            raise ValueError("n_scale must be >= 1 when scale_index is set")
    root = jnp.sqrt(1.0 - beta * beta)

    def step(key, z):
        key_w, key_r, key_s = jax.random.split(key, 3)
        w0, rest0 = z[:d_w], z[d_w:]
        f0 = log_target_no_wprior(z)

        def w_update(carry, k):
            w, f = carry
            k_prop, k_u = jax.random.split(k)
            w_prop = root * w + beta * jax.random.normal(k_prop, jnp.shape(w))
            f_prop = log_target_no_wprior(jnp.concatenate([w_prop, rest0]))
            acc = jnp.log(jax.random.uniform(k_u)) < (f_prop - f)
            return (jnp.where(acc, w_prop, w), jnp.where(acc, f_prop, f)), acc

        (w, f), acc_w = jax.lax.scan(
            w_update, (w0, f0), jax.random.split(key_w, n_pcn))

        if has_rest:
            def rest_update(carry, k):
                rest, f = carry
                k_prop, k_u = jax.random.split(k)
                rest_prop = rest + mh_scales * jax.random.normal(k_prop, jnp.shape(rest))
                f_prop = log_target_no_wprior(jnp.concatenate([w, rest_prop]))
                acc = jnp.log(jax.random.uniform(k_u)) < (f_prop - f)
                return (jnp.where(acc, rest_prop, rest), jnp.where(acc, f_prop, f)), acc

            (rest, f), acc_r = jax.lax.scan(
                rest_update, (rest0, f), jax.random.split(key_r, n_mh))
            acc_r_frac = jnp.mean(acc_r.astype(jnp.float64))
        else:
            rest = rest0
            acc_r_frac = jnp.asarray(1.0, dtype=jnp.float64)

        if scale_index is not None:
            def scale_update(carry, k):
                w_c, rest_c, f_c = carry
                k_eps, k_u = jax.random.split(k)
                eps = scale_step * jax.random.normal(k_eps, ())
                w_prop = w_c * jnp.exp(-eps)
                rest_prop = rest_c.at[scale_index].add(eps)
                f_prop = log_target_no_wprior(jnp.concatenate([w_prop, rest_prop]))
                log_alpha = (f_prop - f_c
                             - 0.5 * (jnp.exp(-2.0 * eps) - 1.0) * jnp.sum(w_c**2)
                             - d_w * eps)
                acc = jnp.log(jax.random.uniform(k_u)) < log_alpha
                w_c = jnp.where(acc, w_prop, w_c)
                rest_c = jnp.where(acc, rest_prop, rest_c)
                f_c = jnp.where(acc, f_prop, f_c)
                return (w_c, rest_c, f_c), acc

            (w, rest, _), acc_s = jax.lax.scan(
                scale_update, (w, rest, f), jax.random.split(key_s, n_scale))
            acc_s_frac = jnp.mean(acc_s.astype(jnp.float64))
        else:
            acc_s_frac = jnp.asarray(1.0, dtype=jnp.float64)

        z_new = jnp.concatenate([w, rest])
        return z_new, (jnp.mean(acc_w.astype(jnp.float64)), acc_r_frac, acc_s_frac)

    return step
