"""Gibbs resampling of Gamma precision hyperpriors, as used by the ML-SGHMC
bayesnn/mf experiments (vendored from SGHMC-jax).

For a parameter group w with prior N(0, 1/lambda) and hyperprior
Gamma(alpha0, beta0) on lambda (rate parameterization):

    lambda ~ Gamma(alpha0 + n/2, beta0 + sum(w^2)/2)

``gibbs_precision`` resamples one lambda per pytree leaf (the original's
"gibbs-sep"); pass ``joint=True`` to pool all leaves ("gibbs-joint").
"""

import jax
import jax.numpy as jnp


def gibbs_precision(key, params, alpha0=1.0, beta0=1.0, joint=False):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    if joint:
        alpha = alpha0 + 0.5 * sum(w.size for w in leaves)
        beta = beta0 + 0.5 * sum(jnp.sum(w * w) for w in leaves)
        lam = jax.random.gamma(keys[0], alpha) / beta
        return jax.tree_util.tree_unflatten(treedef, [lam] * len(leaves))
    lams = []
    for k, w in zip(keys, leaves):
        alpha = alpha0 + 0.5 * w.size
        beta = beta0 + 0.5 * jnp.sum(w * w)
        lams.append(jax.random.gamma(k, alpha) / beta)
    return jax.tree_util.tree_unflatten(treedef, lams)
