Installation instructions
=========================

The simplest way to install *librosa* is through the Python Package Index (PyPI).  This
will ensure that all required dependencies are fulfilled.  This can be achieved by
executing the following command::

    pip install librosa

or::

    sudo pip install librosa

to install system-wide, or::

    pip install -u librosa

to install just for your own user.

If you've downloaded the archive manually from the `releases
<https://github.com/librosa/librosa/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf librosa-VERSION.tar.gz
    cd librosa-VERSION/
    python setup.py install

Additional notes for OS X
-------------------------

ffmpeg
------

To fuel `audioread` with more audio-decoding power, you can install *ffmpeg* which
ships with many audio decoders.

You can use *homebrew* to install the programm by calling
`brew install ffmpeg` or get a binary version from their website https://www.ffmpeg.org.
