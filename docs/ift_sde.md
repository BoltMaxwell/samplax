# Using samplax from ift-sde

ift-sde has two sampler seams; `samplax.integrations.ift_sde` implements both.

## Family B: the joint-posterior engines

ift-sde's `run_nsvi` / `run_npsgld` engines share one signature, and runners
dispatch on a method string. `run_sgmcmc` has the same signature:

```python
from samplax.integrations.ift_sde import run_sgmcmc, SGMCMCConfig

res = run_sgmcmc(
    key, d_w=variant.d_w, d_theta=variant.d_theta, d_x0=variant.d_x0,
    log_likelihood_fn=variant.log_likelihood_fn,
    energy_fn=variant.energy_fn,
    config=SGMCMCConfig(kernel="sghmc", preconditioner="rmsprop",
                        schedule="cyclical", iterations=20_000),
)
res.samples["theta_samples"]   # (chains, kept, d_theta)
```

Add it to a runner's dispatch table next to `nsvi`/`npsgld` and it is
selectable by method string.

**Target caveat:** the default target is the relaxed joint posterior
`log_lik - energy + Gaussian priors`. NPSGLD additionally estimates the NIFF
prior partition gradient with an auxiliary chain; for calibration targets
where `log Z(theta, x0)` varies, pass that estimator as
`correction_grad_fn(z, key) -> grad_z` — samplax supplies the kernel and the
loop, ift-sde supplies the nesting. For state estimation (theta fixed) the
partition term is constant and the default target is already exact.

## Nesting I: inner field samplers for `param_loop`

`sgld_parameter_sampler` consumes inner chains through
`f(field_state, param_state, key) -> field_chain`:

```python
import samplax
from samplax.integrations.ift_sde import make_field_sampler

inner = make_field_sampler(
    samplax.sgld(preconditioner=samplax.rmsprop()),
    samplax.constant(1e-4),
    num_steps=3000, keep_every=10, burn=500)

posterior_field_sampler_fn = lambda f, p, k: inner(posterior_grad_fn, f, p, k)
prior_field_sampler_fn     = lambda f, p, k: inner(prior_grad_fn, f, p, k)
```

where the grad callbacks close over your Hamiltonian exactly as
`calibrate.py` builds them today.

## Practicalities

- ift-sde enables `jax_enable_x64`; samplax follows the input dtype, so
  everything runs float64 there.
- Install: add `git+https://github.com/BoltMaxwell/samplax` to
  `requirements.txt` (the same pattern as the `dax` dependency).
- ift-sde avoids `lax.scan` in some outer loops due to GPU-stack quirks;
  samplax kernels are single steps, so they work in either loop style.
  (`run_sgmcmc` scans in thinning-sized chunks; if the scan issue bites,
  set `thinning=1` to fall back to per-step dispatch.)
