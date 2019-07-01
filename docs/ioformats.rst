Advanced I/O Use Cases
^^^^^^^^^^^^^^^^^^^^^^

This section covers advanced use cases for input and output which go beyond the I/O
functionality currently provided by *librosa*.

Read specific formats
---------------------

*librosa* uses `soundfile <https://github.com/bastibe/PySoundFile>`_ and `audioread <https://github.com/sampsyo/audioread>`_ for reading audio.
As of v0.7, librosa uses `soundfile` by default, and falls back on `audioread` only when dealing with codecs unsupported by `soundfile` (notably, MP3, and some variants of WAV).
For a list of codecs supported by `soundfile`, see the *libsndfile* `documentation <http://www.mega-nerd.com/libsndfile/>`_.

.. note:: See installation instruction for PySoundFile `here <http://pysoundfile.readthedocs.io>`_.

Librosa's load function is meant for the common case where you want to load an entire (fragment of a) recording into memory, but some applications require more flexibility.
In these cases, we recommend using `soundfile` directly.
Reading audio files using `soundfile` is similar to the method in *librosa*. One important difference is that the read data is of shape ``(nb_samples, nb_channels)`` compared to ``(nb_channels, nb_samples)`` in :func:`librosa.core.load`. Also the signal is not resampled to 22050 Hz by default, hence it would need be transposed and resampled for further processing in *librosa*. The following example is equivalent to ``librosa.load(librosa.util.example_audio_file())``:

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

For large audio signals it could be beneficial to not load the whole audio file
into memory.  Librosa 0.7 introduces a streaming interface, which can be used to
work on short fragments of audio sequentially.  :func:`librosa.core.stream` cuts an input
file into *blocks* of audio, which correspond to a given number of *frames*,
which can be iterated over as in the following example:


.. code-block:: python
   :linenos:

   import librosa

   sr = librosa.get_samplerate('/path/to/file.wav')

   # Set the frame parameters to be equivalent to the librosa defaults
   # in the file's native sampling rate
   frame_length = (2048 * sr) // 22050
   hop_length = (512 * sr) // 22050

   # Stream the data, working on 128 frames at a time
   stream = librosa.stream('path/to/file.wav',
                           block_length=128,
                           frame_length=frame_length,
                           hop_length=hop_length)

   chromas = []
   for y in stream:
      chroma_block = librosa.feature.chroma_stft(y=y, sr=sr,
                                                 n_fft=frame_length,
                                                 hop_length=hop_length,
                                                 center=False)
      chromas.append(chromas)
                                                

In this example, each audio fragment ``y`` will consist of 128 frames worth of samples,
or more specifically, ``len(y) == frame_length + (block_length - 1) * hop_length``.
Each fragment ``y`` will overlap with the subsequent fragment by ``frame_length - hop_length``
samples, which ensures that stream processing will provide equivalent results to if the entire
sequence was processed in one step (assuming padding / centering is disabled).

For more details about the streaming interface, refer to :func:`librosa.core.stream`.


Read file-like objects
----------------------

If you want to read audio from file-like objects (also called *virtual files*)
you can use `soundfile` as well.  (This will also work with :func:`librosa.core.load` and :func:`librosa.core.stream`, provided
that the underlying codec is supported by `soundfile`.)

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

*librosa* provides a thin wrapper around `scipy.io.wavfile <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html>`_ to write out WAV files. 

.. code-block:: python
    :linenos:

    import numpy as np

    rate = 44100
    data = np.random.randn(2 * rate)

    librosa.output.write_wav('file.wav', data, rate)



Please be aware that this function only supports floating-point inputs. For example if your processed audio array is of dtype ``np.float64`` (which is the default on most machines), your resulting WAV file would be of type 64-bit float as well. This is not considered to be a `standard PCM wavfile <https://msdn.microsoft.com/en-us/library/windows/hardware/dn653308%28v=vs.85%29.aspx>`_, but most WAV readers should be able to load it without problems.

Writing audio files using `PySoundFile <https://pysoundfile.readthedocs.io/en/latest/>`_ is similar to the method in librosa. However, PySoundFile can automatically convert to a given PCM subtype and additionally support several compressed formats like FLAC or OGG vorbis.

.. code-block:: python
    :linenos:

    import numpy as np
    import soundfile as sf

    rate = 44100
    data = np.random.uniform(-1, 1, size=(rate * 10, 2))

    # Write out audio as 24bit PCM WAV
    sf.write('stereo_file.wav', data, samplerate, subtype='PCM_24')

    # Write out audio as 24bit Flac
    sf.write('stereo_file.flac', data, samplerate, format='flac', subtype='PCM_24')

    # Write out audio as 16bit OGG
    sf.write('stereo_file.ogg', data, samplerate, format='ogg', subtype='vorbis')


In general, we recommend using `PySoundFile` for output rather than ``librosa.output.write_wav``.
