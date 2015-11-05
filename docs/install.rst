Installation instructions
=========================

The simplest way to install *librosa* is through the Python Package Index (PyPI).  This
will ensure that all required dependencies are fulfilled.  This can be achieved by
executing the following command::

    pip install librosa

or::

    sudo pip install librosa

to install system-wide.

If you've downloaded the archive manually from the `releases
<https://github.com/bmcfee/librosa/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf librosa-VERSION.tar.gz
    cd librosa-VERSION/
    python setup.py install

Additional notes
----------------

By default, *librosa* will use `scipy.signal` to resample audio signals, which can 
be slow in practice.  It is highly recommended to install `librsamplerate
<http://www.mega-nerd.com/SRC/>`_ and the corresponding python module,
`scikits.samplerate <https://pypi.python.org/pypi/scikits.samplerate>`_.  

Once these are installed, *librosa* will use the faster `scikits.samplerate` for all 
resampling operations.


Additional notes for OS X
-------------------------

libsamplerate
-------------

In order to use *scipy* with *libsamplerate*, you can use *homebrew* (http://brew.sh)
for installation:
```
brew install libsamplerate
```

The Python bindings are installed via `pip install scikits.samplerate`.

ffmpeg
------

To fuel `audioread` with more audio-decoding power, you can install *ffmpeg* which
ships with many audio decoders.

You can use *homebrew* to install the programm by calling
`brew install ffmpeg` or get a binary version from their website https://www.ffmpeg.org.
