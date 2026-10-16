import samplax


def test_every_kernel_module_is_exported():
    assert samplax.pcn_gibbs is samplax.kernels.pcn_gibbs.pcn_gibbs
    assert "pcn_gibbs" in samplax.__all__


def test_all_names_resolve():
    for name in samplax.__all__:
        assert getattr(samplax, name) is not None, name
