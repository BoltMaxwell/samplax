# samplax

A curated stochastic-gradient MCMC library in pure JAX. Kernels come in two
classes: vendored kernels are taken from verified JAX ports of the original
paper authors' code and bit-equivalence-tested against those ports; original
kernels are implemented here from the paper and verified against closed-form
targets.

```{toctree}
:maxdepth: 2

quickstart
design
api
```
