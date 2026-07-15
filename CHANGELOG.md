# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] — 2026-07-15

### Fixed

- **cyclical keep-mask**: `ift_sde.run_sgmcmc` post-burn-in chunk-ends were kept as
  samples regardless of the schedule's `do_sample`, so a cyclical schedule's
  exploration-phase chunk-ends (temperature zeroed -- optimization iterates, not
  draws) were being kept as posterior samples. Now a chunk is only kept when the
  schedule marks its final step (`(c+1)*thinning - 1`) `do_sample=True`;
  constant/exponential/polynomial schedules are always `do_sample=True` so their
  kept-sample count is unchanged. A cyclical run now keeps strictly fewer samples
  than `(iterations - burn_in) // thinning` -- documented in the module docstring.
- **1 new test** (43 total; corrects the 0.2.0 entry's stale "40 total").

### Added (0.2.0 catch-up, previously undocumented)

- **`SGMCMCConfig.rmsprop_beta` / `rmsprop_eps`**: rmsprop preconditioner EMA decay
  and damping are now configurable (previously fixed at the library defaults),
  needed by the ift-sde wrapper to match its NPSGLD sibling's tuning
  (`alpha_initial` / `delta`).
- **`iterations % thinning` guard**: `run_sgmcmc` now raises `ValueError` when
  `iterations` isn't divisible by `thinning`, instead of silently dropping the
  chunked scan's remainder steps (and an exponential schedule never reaching
  `step_size_final`).

## [0.2.0] — 2026-07-15

### Added

- **exponential schedule**: constant × exp(−ζ t) decay, joins constant, polynomial, and cyclical.
- **adapter refresh**: ift-sde integration now uses 3-arg contract (init_mean, schedule, npsgld-parity sanitization), with stateful `Correction` protocol replacing correction_grad_fn.
- **nested log-Z correction**: persistent-PCD ∇log Z correction with optional re-warm policy (Tieleman 2008 framing) and state-level aux sanitization.
- **14 new tests** (40 total).

### Changed

- **breaking**: `ift_sde.run_sgmcmc` API now requires `correction` (stateful `Correction` protocol) instead of `correction_grad_fn` callback.
