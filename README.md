[![librosa logo](docs/img/librosa_logo_text.svg)](https://librosa.org/)

# librosa

`librosa` is a Python library for audio and music signal processing. It provides the foundational algorithms and tools required for building music information retrieval (MIR) systems.

[![PyPI](https://img.shields.io/pypi/v/librosa.svg)](https://pypi.python.org/pypi/librosa)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/librosa/badges/version.svg)](https://anaconda.org/conda-forge/librosa)
[![License](https://img.shields.io/pypi/l/librosa.svg)](https://github.com/librosa/librosa/blob/main/LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

[![CI](https://github.com/librosa/librosa/actions/workflows/ci.yml/badge.svg)](https://github.com/librosa/librosa/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/librosa/librosa/branch/main/graph/badge.svg?token=ULWnUHaIJC)](https://codecov.io/gh/librosa/librosa)
[![Docs](https://github.com/librosa/librosa/actions/workflows/docs.yml/badge.svg)](https://librosa.org/doc/latest/index.html)
[![Scientific Python Ecosystem Coordination](https://img.shields.io/badge/SPEC-0,1,7-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/)

## Documentation

Full documentation is available at:

* https://librosa.org/doc/

If you're new to librosa, we recommend starting with the
[tutorials](https://librosa.org/doc/latest/auto_tutorials).

If you're looking for API details, head directly to the [API reference](https://librosa.org/doc/latest/api).

To see what's new, check out the [change log](https://librosa.org/doc/latest/changelog.html).

## Installation


### Using PyPI

The latest stable release is available on PyPI, and you can install it by the command
```
pip install librosa
```

### Using conda

Anaconda users can install using `conda-forge`:
```
conda install -c conda-forge librosa
```

For optional dependencies and advanced setup, see the guide in the documentation.


---

## Citing

There are two ways to cite librosa in scholarly work, depending on whether you are citing it for its use in your work
or for its design and motivation.

We additionally encourage you to cite the original publications describing the methods implemented in librosa, in addition to citing librosa itself.  References can often be found within function documentation.

### Citing for use

If you are citing for use in your own work, please cite the version you used by retrieving the appropriate DOI and citation information from Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

This ensures that all contributing authors are credited for the work you are building on.

From librosa version 0.10.2 or later, you can also use `librosa.cite()` to get the DOI link for any version of librosa:

```python
import librosa
librosa.cite()
```

And from 1.0 onward, you can directly retrieve the BibTeX entry for the version you are using:

```python
print(librosa.cite(bib=True))
```

### Citing for design 

If you wish to cite librosa for its design, motivation, etc., please cite the paper
published at SciPy 2015:

    McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto.
    "librosa: Audio and music signal analysis in python."
    In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

As a rule of thumb:

- Use a versioned librosa citation to document the software implementation used in your experiments.
- Cite the original methodological papers for the algorithms that are central to your work.
- Cite the SciPy paper when discussing librosa itself, its design, or its contribution as a software library.
