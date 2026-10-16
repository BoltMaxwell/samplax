# samplax

A curated stochastic-gradient MCMC library in pure JAX. SGLD, SGHMC, cyclical
schedules, AMAGOLD, low-precision SGLD and the HMC reference are vendored from verified
JAX ports of the paper authors' code and bit-equivalence-tested against them. pSGLD
preconditioning and the Gibbs hyperpriors are vendored constructions checked against
targets, and pCN-within-Gibbs is original and verified against closed-form targets.

```{toctree}
:maxdepth: 2

quickstart
design
api
```
