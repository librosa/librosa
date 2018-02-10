Advanced I/O Use Cases
^^^^^^^^^^^^^^^^^^^^^^

This section covers advanced use cases for input and output which go beyond the I/O
functionality currently provided by *librosa*.

Read specific formats
---------------------

*librosa* uses `audioread <https://github.com/sampsyo/audioread>`_ for reading audio. While we chose this library for best flexibility and support of various compressed formats like MP3: some specific formats might not be supported. Especially specific WAV subformats like 24bit PCM or 32bit float might cause problems depending on your installed audioread codecs. *libsndfile* covers a `bunch of these formats <http://www.mega-nerd.com/libsndfile/>`_. There is a neat wrapper for
*libsndfile* called `PySoundFile <https://github.com/bastibe/PySoundFile>`_ which makes it easy to use the library from python.

.. note:: See installation instruction for PySoundFile `here <http://pysoundfile.readthedocs.io>`_.

Reading audio files using PySoundFile is similmar to the method in *librosa*. One important difference is that the read data is of shape ``(nb_samples, nb_channels)`` compared to ``(nb_channels, nb_samples)`` in :func:`<librosa.core.load>`. Also the signal is not resampled to 22050 Hz by default, hence it would need be transposed and resampled for further processing in *librosa*. The following example is equivalent to ``librosa.load(librosa.util.example_audio_file())``:

.. code-block:: python
    :linenos:

    import librosa
    import soundfile as sf

    # Get example audio file
    filename = librosa.util.example_audio_file()

    data, samplerate = sf.read(filename, dtype='float32')
    data = data.T
    data_22k = librosa.resample(data, samplerate, 22050)


Blockwise Reading
-----------------

For large audio signals it could be benficial to not load the whole audio file
into memory. *PySoundFile* supports blockwise reading. In the following example
a block of 1024 samples of audio are read and directly fed into the chroma
feature extractor.

.. code-block:: python
    :linenos:

    import numpy as np
    import soundfile as sf
    from librosa.feature import chroma_stft

    block_gen = sf.blocks('stereo_file.wav', blocksize=1024)
    rate = sf.info('stereo_file.wav').samplerate

    chromas = []
    for bl in block_gen:
        # downmix frame to mono (averaging out the channel dimension)
        y=np.mean(bl, axis=1)
        # compute chroma feature
        chromas.append(chroma_stft(y, sr=rate))



Read file-like objects
----------------------

If you want to read audio from file-like objects (also called *virtual files*)
you can use *PySoundFile*, as well.

E.g.: read files from zip compressed archives:

.. code-block:: python
    :linenos:

    import zipfile as zf
    import soundfile as sf
    import io

    with zf.ZipFile('test.zip') as myzip:
        with myzip.open('stereo_file.wav') as myfile:
            tmp = io.BytesIO(myfile.read())
            data, samplerate = sf.read(tmp)

.. warning:: This is a example does only work in python 3. For python 2 please use ``from urllib2 import urlopen``.

Download and read from URL:

.. code-block:: python
    :linenos:

    import soundfile as sf
    import io

    from six.moves.urllib.request import urlopen

    url = "https://raw.githubusercontent.com/librosa/librosa/master/tests/data/test1_44100.wav"

    data, samplerate = sf.read(io.BytesIO(urlopen(url).read()))


Write out audio files
---------------------

*librosa* uses `scipy.io.wavfile <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html>`_ to write out wav files. Please be aware, that this function uses the numpy dtype to determine the PCM subtype. For example if your processed audio array is of dtype ``np.float64`` (which is the default on most machines), your resulting wav file would be of type 64bit float as well. This is not considered to be a `standard PCM wavfile <https://msdn.microsoft.com/en-us/library/windows/hardware/dn653308%28v=vs.85%29.aspx>`_. If you would like to write 16bit PCM you could convert your array before hand:

.. code-block:: python
    :linenos:

    import numpy as np
    import librosa

    rate = 44100
    audio = np.random.uniform(-1, 1, size = (rate * 10, 2))

    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(
        "out_int16.wav", (audio * maxv).astype(np.int16), rate
    )

Writing audio files using pysoundfile is similar to the method in *librosa*, however it can automatically
convert to a given PCM subtype and additionally support several compressed formats like *FLAC* or *OGG*:

.. code-block:: python
    :linenos:

    import numpy as np
    import soundfile as sf

    rate = 44100
    data = np.random.uniform(-1, 1, size = (rate * 10, 2))

    # Write out audio as 24bit PCM WAV
    sf.write('stereo_file.wav', data, samplerate, subtype='PCM_24')

    # Write out audio as 24bit Flac
    sf.write('stereo_file.flac', data, samplerate, format='flac', subtype='PCM_24')

    # Write out audio as 16bit OGG
    sf.write('stereo_file.ogg', data, samplerate, format='ogg', subtype='vorbis')
