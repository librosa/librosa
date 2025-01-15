<h1><a href="https://librosa.org/"><img src="docs/img/librosa_logo_text.svg" alt="librosa logo"></a></h1>

# librosa

A python package for music and audio analysis.  

[![PyPI](https://img.shields.io/pypi/v/librosa.svg)](https://pypi.python.org/pypi/librosa)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/librosa/badges/version.svg)](https://anaconda.org/conda-forge/librosa)
[![License](https://img.shields.io/pypi/l/librosa.svg)](https://github.com/librosa/librosa/blob/main/LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

[![CI](https://github.com/librosa/librosa/actions/workflows/ci.yml/badge.svg)](https://github.com/librosa/librosa/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/librosa/librosa/branch/main/graph/badge.svg?token=ULWnUHaIJC)](https://codecov.io/gh/librosa/librosa)
[![Docs](https://github.com/librosa/librosa/actions/workflows/docs.yml/badge.svg)](https://librosa.org/doc/latest/index.html)

# Table of Contents

- [librosa](#librosa)
- [Table of Contents](#table-of-contents)
  - [Documentation](#documentation)
  - [Installation](#installation)
    - [Using PyPI](#using-pypi)
    - [Using Anaconda](#using-anaconda)
    - [Building from source](#building-from-source)
    - [Hints for the Installation](#hints-for-the-installation)
    - [`soundfile`](#soundfile)
    - [`audioread` and MP3 support](#audioread-and-mp3-support)
  - [Discussion](#discussion)
  - [Citing](#citing)

---

## Documentation

See <https://librosa.org/doc/> for a complete reference manual and introductory tutorials.

The [advanced example gallery](https://librosa.org/doc/latest/advanced.html) should give you a quick sense of the kinds
of things that librosa can do.

---

[Back To Top ↥](#librosa)

## Installation

### Using PyPI

The latest stable release is available on PyPI, and you can install it by saying

```sh
python -m pip install librosa
```

### Using Anaconda

Anaconda users can install using ```conda-forge```:

```sh
conda install -c conda-forge librosa
```

### Building from source

To build librosa from source, say

```sh
pip install setuptools
python setup.py build
```

Then, to install librosa, say

```sh
python setup.py install
```

If all went well, you should be able to execute the following commands from a python console:

```pycon
import librosa
librosa.show_versions()
```

This should print out a description of your software environment, along with the installed versions of other packages used by librosa.

📝 OS X users should follow the installation guide given below.

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```sh
unzip librosa.zip
python -m pip install -e librosa
```

or

```sh
git clone https://github.com/librosa/librosa.git
python -m pip install -e librosa
```

By calling `pip list` you should see `librosa` now as an installed package:

```sh
librosa (0.x.x, /path/to/librosa)
```

---

[Back To Top ↥](#librosa)

### Hints for the Installation

`librosa` uses `soundfile` and `audioread` to load audio files.

📝 Note that older releases of `soundfile` (prior to 0.11) do not support MP3, which will cause librosa to fall back on the `audioread` library.

### `soundfile`

If you're using `conda` to install librosa, then audio encoding dependencies will be handled automatically.

If you're using `pip` on a Linux environment, you may need to install `libsndfile`
manually.  Please refer to the [SoundFile installation documentation](https://python-soundfile.readthedocs.io/#installation) for details.

### `audioread` and MP3 support

To fuel `audioread` with more audio-decoding power (e.g., for reading MP3 files),
you may need to install either *ffmpeg* or *GStreamer*.

📝*Note that on some platforms, `audioread` needs at least one of the programs to work properly.*

If you are using Anaconda, install *ffmpeg* by calling

```sh
conda install -c conda-forge ffmpeg
```

If you are not using Anaconda, here are some common commands for different operating systems:

- ####  Linux (`apt-get`)

```bash
apt-get install ffmpeg
```

or

```bash
apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly
```

- #### Linux (`yum`)

```bash
yum install ffmpeg
```

or

```bash
yum install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly
```

- #### Mac

```sh
brew install ffmpeg
```

or

```sh
brew install gstreamer
```

- #### Windows

download ffmpeg binaries from this [website](https://www.gyan.dev/ffmpeg/builds/) or gstreamer binaries from this [website](https://gstreamer.freedesktop.org/)

For GStreamer, you also need to install the Python bindings with

```sh
python -m pip install pygobject
```

---

[Back To Top ↥](#librosa)

## Discussion

Please direct non-development questions and discussion topics to our web forum at
<https://groups.google.com/forum/#!forum/librosa>

---

[Back To Top ↥](#librosa)

## Citing

If you want to cite librosa in a scholarly work, there are two ways to do it.

- If you are using the library for your work, for the sake of reproducibility, please cite
  the version you used as indexed at Zenodo:

    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

  From librosa version 0.10.2 or later, you can also use `librosa.cite()`
  to get the DOI link for any version of librosa.

- If you wish to cite librosa for its design, motivation, etc., please cite the paper
  published at SciPy 2015:

    McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

---

[Back To Top ↥](#librosa)
