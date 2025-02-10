#!/usr/bin/env python
# Test configuration and customization

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--librosa-isolation",
        action="store_true",
        default=False,
        help="Skip tests that require network access"
    )
    parser.addoption(
        "--librosa-optional",
        action="store_false",
        default=True,
        help="Run tests that use optional dependencies"
    )


def pytest_collection_modifyitems(config, items):
    print(config.getoption("--librosa-isolation"))
    if not config.getoption("--librosa-isolation"):
        # User wants to run network tests, so do nothing.
        return
    skip_isolation = pytest.mark.skip(reason="testing in isolation mode")
    for item in items:
        if "noisolation" in item.keywords:
            item.add_marker(skip_isolation)
