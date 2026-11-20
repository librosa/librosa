#!/usr/bin/env python
"""Version info"""

import importlib
import importlib.metadata
import sys

short_version = "1.1.0dev"
version = "1.1.0dev"


def __get_mod_version(modname: str) -> str | None:
    try:
        dist_name = "scikit-learn" if modname == "sklearn" else modname
        return importlib.metadata.version(dist_name)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        pass

    try:
        name = "IPython" if modname.lower() == "ipython" else modname.replace("-", "_")
        if name in sys.modules:
            mod = sys.modules[name]
        else:
            mod = importlib.import_module(name)
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
        "scikit-learn",
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
        "sphinx-gallery",
        "sphinx-design",
        "sphinxcontrib-googleanalytics",
        "sphinx-copybutton",
        "sphinxcontrib-mermaid",
        "sphinxcontrib-svg2pdfconverter",
        "umap-learn",
        "pandas",
        "myst-parser",
        "mir-eval",
        "ipython",
        "sounddevice",
        "pytest",
        "pytest-mpl",
        "pytest-cov",
        "samplerate",
        "resampy",
        "presets",
        "scipy-stubs",
    ]

    print("INSTALLED VERSIONS")
    print("------------------")
    print(f"python: {sys.version}\n")
    print(f"librosa: {version}\n")
    for dep in core_deps:
        print(f"{dep}: {__get_mod_version(dep)}")
    print()
    for dep in extra_deps:
        print(f"{dep}: {__get_mod_version(dep)}")
