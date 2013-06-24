librosa
=======

A python module for music and audio processing.

Mostly ports and reimplementations of DPWE's matlab code.

* librosa/    The librosa module

* examples/   Some basic examples of audio processing with librosa

* tests/      nose unit tests to validate against original dpwe implementations


Installation
============

To build librosa, say `python setup.py build`.
Then, to install librosa, say `python setup.py install`.
If all went well, you should be able to execute the demo scripts under `examples/`.


dependencies
============

* numpy >= 1.7.0
* scipy
* sklearn
* audioread
* (optional) scikits.samplerate
