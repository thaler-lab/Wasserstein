import platform

import pytest

import wasserstein

@pytest.mark.config
def test_has_openmp_options():
    assert hasattr(wasserstein, 'with_openmp')
    assert hasattr(wasserstein, 'without_openmp')
    assert hasattr(wasserstein, 'openmp')

@pytest.mark.config
def test_switch_openmp_support():
    assert wasserstein.openmp()

    if platform.system() == 'Darwin':
        wasserstein.without_openmp()
        assert not wasserstein.openmp()

    wasserstein.with_openmp()
    assert wasserstein.openmp()
