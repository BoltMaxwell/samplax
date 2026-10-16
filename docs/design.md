# Design and provenance

## Why another SG-MCMC library

[jax-sgmc](https://github.com/tummfm/jax-sgmc) and
[sgmcmcjax](https://github.com/jeremiecoullon/SGMCMCJax) already exist and are
good. samplax differs deliberately:

1. **Provenance over coverage.** Every kernel traces to a verified port of the
   paper authors' own code — SGHMC-jax, csgmcmc-jax, low-precision-sgld-jax,
   amagold-jax — each of which was compared quantitatively against the
   original implementation (matlab/Octave, numpy, C++, PyTorch) before its
   originals were retired. The vendored copies here are bit-equivalence-tested
   against those ports (`tests/test_vendored_equivalence.py`). A sampler
   whose lineage we can't verify doesn't go in.
2. **Not a framework.** No data loaders, potential modules, solver aliases,
   or I/O collectors. You bring gradients (or log-density callbacks) and a
   pytree. Kernels never own the training loop, so they embed in outer loops
   — the requirement that drove the design (see the ift-sde integration).
3. **Cross-cut composition axes.** Schedules (constant / polynomial /
   cyclical), preconditioners (identity / RMSprop-pSGLD), precision
   (quantizers + variance-corrected quantization), and M-H correction
   (AMAGOLD) are orthogonal where the math allows it, and their couplings
   are documented where it doesn't (VC quantization is SGLD-family by
   construction; AMAGOLD needs energy evaluations).
4. **Pure JAX.** No blackjax dependency. The exact parameter mappings between
   these kernels and blackjax's `sgmcmc.diffusions`, and the stability
   analysis of blackjax's qp splitting (variance inflation ~ alpha/(alpha -
   eta), marginal stability at eta = alpha), live in the SGHMC-jax
   verification notes if cross-checking is ever needed.

## Conventions

- **Ascent gradients.** `grad` is the gradient of the log-density
  (log-posterior). Ports from descent-convention code negate at the boundary;
  the equivalence tests pin this down.
- **Step-time step size and temperature.** Anything a schedule might touch is
  a `step()` argument, not a factory argument.
- **Flat noise draws.** Gaussian noise is one flat draw over the raveled
  pytree (`samplax.gaussian_like`), so trajectories are invariant to how
  parameters are grouped into containers.
- **Dtype follows the position.** Works under `jax_enable_x64` (ift-sde) and
  in float32.
- **State NamedTuples.** First field is always `position`.

## Known sharp edges (inherited knowledge)

- SGHMC's momentum refresh must precede the position update (the original
  "pq" ordering). The reversed ordering (used by blackjax's diffusion) only
  contracts for `step-size-scaled friction > effective step`, and inflates
  the stationary variance even when stable.
- The `v_hat` gradient-noise correction (Chen et al. 2014) requires
  `0.5 * v_hat * step_size < alpha` (SGHMC) or `< 1` (SGLD).
- AMAGOLD's M-H energy is `datasize x` a mean-scale energy difference: in
  float32 the exponent carries O(1) rounding noise, so late-run acceptance
  statistics are implementation-sensitive (the sampled distribution is not).
- SGHMC with the AMAGOLD BNN hyperparameters (lr 5e-4, alpha 1e-5) diverges
  around epoch 510 on MNIST — reproduced within one epoch of the original
  PyTorch. Divergence behavior is part of the method, not a bug.
- Low-precision quantization grids are scale-sensitive: `lp_sgld` keeps the
  original mean-loss-gradient convention rather than the library's sum-scale
  ascent convention. Don't "fix" this without re-deriving the grids.

## Roadmap

- Fisher preconditioners (diag/dense) to match ift-sde's full preconditioner
  menu.
- Validation of `run_sgmcmc` against ift-sde's `validate_mission.py` harness.
- Candidates for curation: SGNHT, reSGLD — same rule: only via a verified
  port of the original code.
