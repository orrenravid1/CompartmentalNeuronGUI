from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-jaxley",
        action="store_true",
        default=False,
        help="run Jaxley backend tests and benchmarks",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "jaxley: marks tests that exercise the optional Jaxley backend or slower subprocess checks",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-jaxley"):
        return

    skip_jaxley = pytest.mark.skip(reason="need --run-jaxley to run Jaxley backend tests")
    for item in items:
        if "jaxley" in item.keywords:
            item.add_marker(skip_jaxley)
