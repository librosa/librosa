librosa
=======

A python package for music and audio processing.

* `librosa/`    The librosa package

* `examples/`   Some basic examples of audio processing and feature extraction with librosa

* `tests/`      nosetests for numerical compatibility against reference implementations

Documentation
=============
See http://bmcfee.github.io/librosa/

Demonstration
=============
See http://nbviewer.ipython.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb for a brief introduction to some commonly used features.

Installation
============

The latest stable release is available on PyPI, and you can install it by saying `pip install librosa`.

To build librosa from source, say `python setup.py build`.
Then, to install librosa, say `python setup.py install`.
If all went well, you should be able to execute the demo scripts under `examples/`.

Alternatively, you can download or clone the repository and use `easy_install` to handle dependencies:

```
unzip librosa.zip
easy_install librosa
```
or
```
git clone https://github.com/bmcfee/librosa.git
easy_install librosa
```
