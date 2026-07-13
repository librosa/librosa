#!/usr/bin/env python
# Test configuration and customization

import pytest


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
