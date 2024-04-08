#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Version info"""

import sys
import importlib

short_version = "0.10"
version = "0.10.2"


def __get_mod_version(modname):
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
        try:
            return mod.__version__
        except AttributeError:
            return "installed, no version number available"

    except ImportError:
        return None


def show_versions() -> None:
    """Return the version information for all librosa dependencies."""
    core_deps = [
        "audioread",
        "numpy",
        "scipy",
        "sklearn",
        "joblib",
        "decorator",
        "numba",
        "soundfile",
        "pooch",
        "soxr",
        "typing_extensions",
        "lazy_loader",
        "msgpack",
    ]

    extra_deps = [
        "numpydoc",
        "sphinx",
        "sphinx_rtd_theme",
        "matplotlib",
        "sphinx_multiversion",
        "sphinx_gallery",
        "mir_eval",
        "ipython",
        "sphinxcontrib.rsvgconverter",
        "pytest",
        "pytest_mpl",
        "pytest_cov",
        "samplerate",
        "resampy",
        "presets",
        "packaging",
    ]

    print("INSTALLED VERSIONS")
    print("------------------")
    print(f"python: {sys.version}\n")
    print(f"librosa: {version}\n")
    for dep in core_deps:
        print("{}: {}".format(dep, __get_mod_version(dep)))
    print("")
    for dep in extra_deps:
        print("{}: {}".format(dep, __get_mod_version(dep)))
