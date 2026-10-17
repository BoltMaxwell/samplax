# samplax

A curated stochastic-gradient MCMC library in pure JAX. SGLD, SGHMC, cyclical
schedules, AMAGOLD, low-precision SGLD and the HMC reference are vendored from verified
JAX ports of the paper authors' code and equivalence-tested against them. pSGLD
preconditioning and the Gibbs hyperpriors are vendored constructions checked against
targets, and pCN-within-Gibbs is original and verified against closed-form targets.

```{toctree}
:maxdepth: 2

quickstart
design
api
```

samplax is released under the Apache License 2.0. `LICENSE` and `NOTICE` in the
repository carry the terms and the credits for each vendored kernel's upstream
code and paper; `CITATION.cff` has the full citation metadata for the paper
samplax was written for and for every method paper behind the kernels.
