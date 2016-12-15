Advanced I/O
============

This section covers advanced input and output which is currently not
convered by *librosa*.

Refer to librosa_test :ref:`librosa.core.load <core>`


Block processing
----------------

Note, that
.. code-block:: python
    :linenos:

    import numpy as np
    import soundfile as sf
    from librosa.feature import chroma_stft

    block_gen = sf.blocks('stereo_file.wav', blocksize=1024)
    rate = sf.info('stereo_file.wav').samplerate

    chromas = [chroma_stft(y=np.mean(bl, axis=1), sr=rate) for bl in block_gen]



Read data using PySoundfile
---------------------------

.. code-block:: python
    :linenos:

    import librosa
    import soundfile as sf

    # Get example audio file
    filename = librosa.util.example_audio_file()

    data, samplerate = sf.read(filename)
    # output is (nb_samples, nb_channels)
    # whereas librosa.load uses
    # (nb_channels, nb_samples)

    # data = data.T


Read file-like objects
----------------------

Read from zipped files

.. code-block:: python
    :linenos:

    import zipfile as zf
    import soundfile as sf
    import io

    with zf.ZipFile('test.zip') as myzip:
        with myzip.open('stereo_file.wav') as myfile:
            tmp = io.BytesIO(myfile.read())
            data, samplerate = sf.read(tmp)


Read from URL

.. code-block:: python
    :linenos:

    import soundfile as sf
    import io

    # from urllib.request import urlopen
    from urllib2 import urlopen

    url = "https://raw.githubusercontent.com/librosa/" +
      "librosa/master/tests/data/test1_44100.wav"

    data, samplerate = sf.read(io.BytesIO(urlopen(url).read()))


Write data using PySoundfile
----------------------------

.. code-block:: python
    :linenos:

    import numpy as np
    import soundfile as sf

    rate = 44100
    data = np.random.randn(rate * 10, 2)

    # Write out audio as 24bit PCM WAV
    sf.write('stereo_file.wav', data, samplerate, subtype='PCM_24')

    # Write out audio as 24bit Flac
    sf.write('stereo_file.flac', data, samplerate, format='flac', subtype='PCM_24')

    # Write out audio as 16bit ogg
    sf.write('stereo_file.ogg', data, samplerate, format='ogg', subtype='vorbis')


Write out using write_wav
-------------------------

librosa is using `scipy.io.wavfile` to write wav files which in fact
seems to do use the dtype to determine the PCM subformat.
From the [docs](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.io.wavfile.write.html):

.. code-block:: python
    :linenos:

    import numpy as np
    import librosa

    rate = 44100
    data = np.random.randn(rate * 10, 2)

    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(
        "out_int16.wav", (audio * maxv).astype(np.int16), rate
    )
