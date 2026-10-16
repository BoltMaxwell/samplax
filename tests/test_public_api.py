import samplax


def test_every_kernel_module_is_exported():
    assert samplax.pcn_gibbs is samplax.kernels.pcn_gibbs.pcn_gibbs
    assert "pcn_gibbs" in samplax.__all__


def test_all_names_resolve():
    for name in samplax.__all__:
        assert getattr(samplax, name) is not None, name


def test_agents_md_names_every_public_module():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "AGENTS.md").read_text()
    for module in ("kernels", "schedules", "preconditioners", "transforms", "gibbs", "base"):
        assert module in text, module


import jax
import pytest


@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        (dict(step_size=0.0, step_size_final=0.1, num_training_steps=10), r"step_size must be > 0 .*got"),
        (dict(step_size=0.1, step_size_final=-1.0, num_training_steps=10), r"step_size_final must be > 0 .*got"),
        (dict(step_size=0.1, step_size_final=0.01, num_training_steps=0), r"num_training_steps must be >= 1 .*got"),
    ],
)
def test_schedules_errors_name_valid_inputs(kwargs, pattern):
    from samplax.schedules import exponential

    with pytest.raises(ValueError, match=pattern):
        exponential(**kwargs)


@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        (dict(beta=0.0), r"beta must be in \(0, 1\].*got"),
        (dict(beta=0.5, n_pcn=0), r"must be >= 1 .*got n_pcn=0, n_mh=1"),
        (dict(beta=0.5, n_mh=0), r"must be >= 1 .*got n_pcn=1, n_mh=0"),
        (dict(beta=0.5, scale_index=3), r"scale_index must be in \[0, 1\).*got 3"),
        (dict(beta=0.5, scale_index=0, n_scale=0), r"n_scale must be >= 1 .*got n_scale=0"),
    ],
)
def test_pcn_gibbs_errors_name_valid_inputs(kwargs, pattern):
    from samplax import pcn_gibbs

    with pytest.raises(ValueError, match=pattern):
        pcn_gibbs(lambda z: 0.0, d_w=2, mh_scales=[0.1], **kwargs)


@pytest.mark.parametrize(
    "args, kwargs, pattern",
    [
        (("xyz", 8, 4), {}, r"expected one of 'naive', 'sgldlp_f', 'vc'"),
        (("naive", 8, 4), dict(number="xyz"), r"expected one of 'fixed', 'block'"),
    ],
)
def test_lp_sgld_errors_name_valid_inputs(args, kwargs, pattern):
    from samplax.transforms.lp_sgld import lp_sgld

    # Both are rejected at construction, not on the first step.
    with pytest.raises(ValueError, match=pattern):
        lp_sgld(*args, datasize=10, **kwargs)


def test_quant_errors_name_valid_inputs():
    import jax.numpy as jnp
    from samplax.transforms import quant

    x = jnp.ones(3)
    with pytest.raises(ValueError, match=r"key=jax\.random\.PRNGKey"):
        quant._round_int(x, "stochastic", None)
    with pytest.raises(ValueError, match=r"key=jax\.random\.PRNGKey"):
        quant.float_quantize(x, 5, 2, "stochastic", None)
    with pytest.raises(ValueError, match="expected one of 'nearest', 'stochastic'"):
        quant._round_int(x, "xyz", None)
    with pytest.raises(ValueError, match="expected one of 'nearest', 'stochastic'"):
        quant.float_quantize(x, 5, 2, "xyz")
    with pytest.raises(TypeError, match="expected an instance of FixedPoint"):
        quant.make_quantizer("fixed")
