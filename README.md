librosa
=======
A python package for music and audio analysis.  The primary purpose of librosa is to implement common tools for low- and high-level signal-based music analysis.


Documentation
-------------
See http://bmcfee.github.io/librosa/ for a complete reference manual and introductory tutorials.


Demonstration notebooks
-----------------------
What does librosa do?  Here are some quick demonstrations:

* [Introduction notebook](http://nbviewer.ipython.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb): a brief introduction to some commonly used features.
* [Decomposition and IPython integration](http://nbviewer.ipython.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20audio%20effects%20and%20playback.ipynb): an intermediate demonstration, illustrating how to process and play back sound
* [SciKit-Learn integration](http://nbviewer.ipython.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20sklearn%20feature%20pipeline.ipynb): an advanced demonstration, showing how to tie librosa functions to feature extraction pipelines for machine learning


Installation
------------

The latest stable release is available on PyPI, and you can install it by saying 
```
pip install librosa
```

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


Discussion
----------

Please direct non-development questions and discussion topics to our web forum at 
https://groups.google.com/forum/#!forum/librosa 


Citing
------

Please refer to the Zenodo link below for citation information.

[![DOI](https://zenodo.org/badge/doi/10.5072/zenodo.12714.png)](http://dx.doi.org/10.5072/zenodo.12714)


Build Status
============
[![Build Status](https://travis-ci.org/bmcfee/librosa.png)](http://travis-ci.org/bmcfee/librosa)
