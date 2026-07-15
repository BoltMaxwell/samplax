# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] — 2026-07-15

### Added

- **exponential schedule**: constant × exp(−ζ t) decay, joins constant, polynomial, and cyclical.
- **adapter refresh**: ift-sde integration now uses 3-arg contract (init_mean, schedule, npsgld-parity sanitization), with stateful `Correction` protocol replacing correction_grad_fn.
- **nested log-Z correction**: persistent-PCD ∇log Z correction with optional re-warm policy (Tieleman 2008 framing) and state-level aux sanitization.
- **14 new tests** (40 total).

### Changed

- **breaking**: `ift_sde.run_sgmcmc` API now requires `correction` (stateful `Correction` protocol) instead of `correction_grad_fn` callback.
