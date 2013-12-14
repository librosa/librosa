librosa
=======

A python package for music and audio processing.

* librosa/    The librosa package

* examples/   Some basic examples of audio processing with librosa

* tests/      nose unit tests to validate against original dpwe implementations

Documentation
=============
See http://bmcfee.github.io/librosa/

Demonstration
=============
See http://nbviewer.ipython.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb

Installation
============

To build librosa, say `python setup.py build`.
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


dependencies
============

* audioread
* numpy >= 1.7.0
* scipy
* sklearn
* matplotlib
* (optional) scikits.samplerate >= 0.3
