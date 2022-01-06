Multi-channel
^^^^^^^^^^^^^

This section provides an overview of how multi-channel signals are handled in
*librosa*.
The one-sentence summary is that most of the functions which only supported single-channel 
inputs up to librosa 0.8 now support multi-channel audio with no modification necessary.

Dimensions
----------
Before discussing multi-channel, it is worth reviewing how single-channel (monaural)
signals are processed.
Librosa processes all signals and derived data as `numpy.ndarray` (N-dimensional array) objects.
By default, when librosa loads a multichannel signal, it averages all channels to produce a mono mixture.
The resulting object is a 1-dimensional array of shape ``(N_samples,)`` which
represents the time-series of sample values.
Subsequent processing typically produces higher-dimensional transformations of the
time-series, for example, a short-time Fourier transform (``librosa.stft``) produces
a two-dimensional array of shape ``(N_frequencies, N_frames)``.
Note that the second (trailing) dimension corresponds to the number of frames in the
signal, which is proportional to length; the first dimension corresponds to the
number of frequencies (or more generally, features) measured at each frame.

When working with multi-channel signals, we may choose to skip the default down-mixing
step by specifying ``mono=False`` in the call to ``librosa.load``, as in the following:

.. code-block:: python

    import librosa

    # Get the "high-quality" multi-channel version of 
    # an example track
    filename = librosa.ex('trumpet', hq=True)

    # Load as multi-channel data
    y_stereo, sr = librosa.load(filename, mono=False)


The resulting object now has two dimensions instead of one, with ``y_stereo.shape ==
(N_channels, N_samples)``.
This way, we can access the first channel as ``y_stereo[0]``, the second channel as
``y_stereo[1]``, and so on if there are more than two channels.

Librosa represents data according to the following general pattern:

    - trailing dimensions correspond to time (samples, frames, etc)
    - leading dimensions may correspond to channels
    - intermediate dimensions correspond to "features" (or frequencies, harmonics,
      etc).

This pattern is designed so that indexing is *consistent* when slicing out
individual channels.
This is demonstrated in the examples below.


Examples
--------

As a first example, consider computing a short-time Fourier transform of the stereo
example signal loaded above.
This is accomplished in exactly the same way as if the signal was mono, that is:

.. code-block:: python

    D_stereo = librosa.stft(y_stereo)


The shape of the resulting STFT is ``D_stereo.shape == (N_channels, N_frequencies, N_frames)``.
The slice ``D_stereo[0]`` then corresponds to the STFT of the first channel ``y_stereo[0]``, and ``D_stereo[1]`` is the STFT of the second channel ``y_stereo[1]``, and so on.

As a more advanced example, we can construct a multi-channel, harmonic spectrogram
using `librosa.interp_harmonics`:

.. code-block:: python

    S_stereo = np.abs(D_stereo)

    # Get the default Fourier frequencies
    freqs = librosa.fft_frequencies(sr=sr)

    # We'll interpolate the first five harmonics of each frequency
    harmonics = [1, 2, 3, 4, 5]

    S_harmonics = librosa.interp_harmonics(S_stereo, freqs=freqs, h_range=harmonics)


The resulting object has four dimensions now: ``S_harmonics.shape == (N_channels,
N_harmonics, N_frequencies, N_frames)``.
As noted above, the leading dimension corresponds to channels, the trailing
dimension corresponds to time (frames), and the intermediate dimensions correspond
to derived features.
In this way, indexing a specific channel (e.g., ``S_harmonics[1]`` for the second
channel) provides the entire feature array derived from the second channel, and
produces an output of shape ``(N_harmonics, N_frequencies, N_frames)``.


Documentation
-------------

When reading the library documentation, you may come across functions like
`librosa.stft` which describe the input signal parameter as::

    y : np.ndarray [shape=(..., n)], real-valued

The "..." here is analogous to Python's `Ellipsis` object, and in this context, it acts as a place-holder for "0 or more dimensions".
This is analogous to numpy's use of `Ellipsis` to bypass variable numbers of
dimensions in `numpy.ndarray` objects.
For example, to slice a single frame ``n`` out of the multi-channel harmonic spectrogram
above, you could do either::

    S[:, :, :, n]

or::

    S[..., n]

The latter is generally preferred as it generalizes to arbitrarily many leading
dimensions.

Whenever functions are described as accepting shapes containing "...", the
implication is that the (arbitrarily many) leading dimensions are preserved in the
output unless otherwise stated.

Some functions accept an ``axis=`` parameter to specify a target axis along which to
operate.
As a general convention, ``axis=-1`` (the final axis) usually corresponds to "time"
(or samples, or frames), while ``axis=-2`` (the second-to-last axis) usually
corresponds to "frequency" or some other derived feature.


Exceptions
----------

Not all functions in librosa naturally generalize to multi-channel data, though most
do.
Similarly, some functions do generalize, but in ways that may not match your
expectations.
This section briefly summarizes places where multi-channel support is limited.


**Detectors** with ragged output, for example beat tracking (`librosa.beat`) and
onset detection (`librosa.onset.onset_detect`) do not support multi-channel inputs.
This is because the output may have differing numbers of events in each channel, and
therefore cannot be consistently stored in a `numpy.ndarray` output object.
In these cases, it is best to either process each channel separately (if they are
truly independent) or aggregate representations across channels (e.g., by averaging
features) if they are strongly related.


**Self- and cross-similarity matrices**, as computed by `librosa.segment.recurrence_matrix` have limited multi-channel support.
This is because the output objects may be sparse data structures (such as `scipy.sparse.csr_matrix`) which do not generalize to more than two dimensions.
These functions still accept multi-channel input, but flatten the leading dimensions
(channels) when comparing features between different time-steps.
If independent similarity matrices are desired, it is recommended to process each
channel independently.


**Decompositions and sequence alignments**, like similarity matrices, have limited
support.
Harmonic-percussive source separation (`librosa.decompose.hpss`) can fully accept
multi-channel input with independent processing, but other decomposition
(`librosa.decompose.nn_filter` and `librosa.decompose.decompose`) impose some
restrictions on how multi-channel inputs are processed.
Sequence alignment functions like `librosa.decompose.dtw` and
`librosa.decompose.rqa` operate much like similarity matrix functions, and interpret
leading dimensions as additional "feature" dimensions which are flattened prior to
alignment.


**Display** functions have limited multi-channel support.
`librosa.display.waveshow` can accept single or 2-channel input, though the second
channel is only used when zoomed out to envelope mode.
`librosa.display.specshow` does not accept multi-channel input.


Advanced uses and caveats
-------------------------

Multi-channel support is relatively flexible in librosa.
In particular, you may organize channels over two dimensions or more, although a 
single channel dimension is the most common use case.
For example, if you want to simultaneously process a collection of stereo recordings
of equal length, you may collect the signals into an array of shape ``y.shape =
(N_tracks, N_channels, N_samples)``.
Any derived data (e.g. spectrograms like in the example above) would then have *two*
leading dimensions, corresponding first to track and then to channel within the
track.
In theory, any number of leading dimensions can be used, though caution should be
exercised to minimize memory consumption.


Note that although many functions preserve channel independence, this is not
guaranteed in general.
For example, decibel scaling by `librosa.amplitude_to_db` will compare each channel
to a reference value which may be derived from *all channels simultaneously*.
This can lead to differences in behavior when processing channels independently or
simultaneously as a multi-channel input.
Functions which guarantee channel-wise independence are documented accordingly.



