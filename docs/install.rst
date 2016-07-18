Installation instructions
=========================

pypi
~~~~
The simplest way to install *librosa* is through the Python Package Index (PyPI).
This will ensure that all required dependencies are fulfilled.
This can be achieved by executing the following command::

    pip install librosa

or::

    sudo pip install librosa

to install system-wide, or::

    pip install -u librosa

to install just for your own user.

conda
~~~~~
If you use conda/Anaconda environments, librosa can be installed from the 
`conda-forge` channel::

    conda install -c conda-forge librosa


Source
~~~~~~

If you've downloaded the archive manually from the `releases
<https://github.com/librosa/librosa/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf librosa-VERSION.tar.gz
    cd librosa-VERSION/
    python setup.py install

Alternately, the latest development version can be installed via pip::

    pip install git+https://github.com/librosa/librosa


ffmpeg
------

To fuel `audioread` with more audio-decoding power, you can install *ffmpeg* which
ships with many audio decoders.  Note that conda users on Linux and OSX will
have this installed by default; Windows users must install ffmpeg separately.

OSX users can use *homebrew* to install ffmpeg by calling
`brew install ffmpeg` or get a binary version from their website https://www.ffmpeg.org.
