librosa
=======
A python package for music and audio analysis.  

[![PyPI](https://img.shields.io/pypi/v/librosa.svg)](https://pypi.python.org/pypi/librosa)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/librosa/badges/version.svg)](https://anaconda.org/conda-forge/librosa)
[![License](https://img.shields.io/pypi/l/librosa.svg)](https://github.com/librosa/librosa/blob/master/LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

[![Build Status](https://travis-ci.org/librosa/librosa.png?branch=master)](http://travis-ci.org/librosa/librosa?branch=master)
[![Build status](https://ci.appveyor.com/api/projects/status/8i1hhr8yj78195xf?svg=true)](https://ci.appveyor.com/project/bmcfee/librosa)
[![Coverage Status](https://coveralls.io/repos/librosa/librosa/badge.svg?branch=master)](https://coveralls.io/r/librosa/librosa?branch=master)
[![Dependency Status](https://dependencyci.com/github/librosa/librosa/badge)](https://dependencyci.com/github/librosa/librosa)


Documentation
-------------
See http://librosa.github.io/librosa/ for a complete reference manual and introductory tutorials.


Demonstration notebooks
-----------------------
What does librosa do?  Here are some quick demonstrations:

* [Introduction notebook](http://nbviewer.ipython.org/github/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb): a brief introduction to some commonly used features.
* [Decomposition and IPython integration](http://nbviewer.ipython.org/github/librosa/librosa/blob/master/examples/LibROSA%20audio%20effects%20and%20playback.ipynb): an intermediate demonstration, illustrating how to process and play back sound


Installation
------------

The latest stable release is available on PyPI, and you can install it by saying
```
pip install librosa
```

Anaconda users can install using ``conda-forge``:
```
conda install -c conda-forge librosa
```

To build librosa from source, say `python setup.py build`.
Then, to install librosa, say `python setup.py install`.
If all went well, you should be able to execute the demo scripts under `examples/`
(OS X users should follow the installation guide given below).

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```
unzip librosa.zip
pip install -e librosa
```
or
```
git clone https://github.com/librosa/librosa.git
pip install -e librosa
```

By calling `pip list` you should see `librosa` now as an installed package:
```
librosa (0.x.x, /path/to/librosa)
```

### Hints for the Installation

`librosa` uses `soundfile` and `audioread` to load audio files.
Note that `soundfile` does not currently support MP3, which will cause librosa to
fall back on the `audioread` library.

#### soundfile

If you're using `conda` to install librosa, then most audio coding dependencies (except MP3) will be handled automatically.

If you're using `pip` on a Linux environment, you may need to install `libsndfile`
manually.  Please refer to the [SoundFile installation documentation](https://pysoundfile.readthedocs.io/#installation) for details.

#### audioread and MP3 support

To fuel `audioread` with more audio-decoding power (e.g., for reading MP3 files),
you may need to install either *ffmpeg* or *GStreamer*.

*Note that on some platforms, `audioread` needs at least one of the programs to work properly.*

If you are using Anaconda, install *ffmpeg* by calling
```
conda install -c conda-forge ffmpeg
```

If you are not using Anaconda, here are some common commands for different operating systems:

* Linux (apt-get): `apt-get install ffmpeg` or `apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
* Linux (yum): `yum install ffmpeg` or `yum install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
* Mac: `brew install ffmpeg` or `brew install gstreamer`
* Windows: download binaries from this [website]( https://gstreamer.freedesktop.org/) 

For GStreamer, you also need to install the Python bindings with 
```
pip install pygobject
```

Discussion
----------

Please direct non-development questions and discussion topics to our web forum at
https://groups.google.com/forum/#!forum/librosa


Citing
------

If you want to cite librosa in a scholarly work, there are two ways to do it.

- If you are using the library for your work, for the sake of reproducibility, please cite
  the version you used as indexed at Zenodo:

    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

- If you wish to cite librosa for its design, motivation etc., please cite the paper
  published at SciPy 2015:

    McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
