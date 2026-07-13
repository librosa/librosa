#!/usr/bin/env python
# Test configuration and customization

import pytest
import numpy as np
import matplotlib
from packaging.version import parse as parse_version


# An RNG seed for all tests to use if they need randomness
@pytest.fixture(scope="session")
def rng_factory():
    def _factory():
        # Use a fixed seed for reproducibility
        return np.random.default_rng(seed=440)

    return _factory


@pytest.fixture
def rng(rng_factory):
    """Return a random number generator for use in tests."""
    return rng_factory()


@pytest.fixture(scope="module")
def rng_mod(rng_factory):
    """A module-scoped RNG fixture"""
    return rng_factory()


def pytest_addoption(parser):
    parser.addoption(
        "--librosa-isolation",
        action="store_true",
        default=False,
        help="Skip tests that require network access",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--librosa-isolation"):
        # User wants to run network tests, so do nothing.
        return
    skip_isolation = pytest.mark.skip(reason="testing in isolation mode")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_isolation)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: mark tests that require network access."
    )
    # Bypass image comparison tests on older matplotlib versions
    # due to changes in font rendering
    if parse_version(matplotlib.__version__) < parse_version("3.11"):
        # pytest-mpl relies on config.getoption("--mpl") 
        # which reads directly from config.option.mpl
        config.option.mpl = False
