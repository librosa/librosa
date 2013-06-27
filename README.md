librosa
=======

A python package for music and audio processing.

Mostly ports and reimplementations of DPWE's matlab code.

* librosa/    The librosa package

* examples/   Some basic examples of audio processing with librosa

* tests/      nose unit tests to validate against original dpwe implementations

Documentation
=============
See http://bmcfee.github.io/librosa/

Demonstration
=============
See http://nbviewer.ipython.org/5878337

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

* numpy >= 1.7.0
* scipy
* sklearn
* audioread
* (optional) scikits.samplerate
* (recommended) pygst
