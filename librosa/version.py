#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Version info"""

import importlib
import sys

short_version = "1.0.0"
version = "1.0.0"


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
        "numpy",
        "scipy",
        "sklearn",
        "joblib",
        "decorator",
        "numba",
        "soundfile",
        "pooch",
        "soxr",
        "lazy_loader",
        "msgpack",
    ]

    extra_deps = [
        "numpydoc",
        "sphinx",
        "pydata-sphinx-theme",
        "matplotlib",
        "sphinx_gallery",
        "sphinx-design",
        "sphinxcontrib-googleanalytics",
        "sphinx-copybutton",
        "umap-learn",
        "pandas",
        "myst-parser",
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
        "scipy-stubs",
        "types-decorator",
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
