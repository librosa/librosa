#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Pitch-tracking and tuning estimation'''

import warnings
import numpy as np
import six

from .spectrum import _spectrogram
from . import time_frequency
from .._cache import cache
from .. import util

__all__ = ['estimate_tuning', 'pitch_tuning', 'piptrack']


def estimate_tuning(y=None, sr=22050, S=None, n_fft=2048,
                    resolution=0.01, bins_per_octave=12, **kwargs):
    '''Estimate the tuning of an audio time series or spectrogram input.

    Parameters
    ----------
    y: np.ndarray [shape=(n,)] or None
        audio signal

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    S: np.ndarray [shape=(d, t)] or None
        magnitude or power spectrogram

    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if `y` is provided.

    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to measurements in cents.

    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave

    kwargs : additional keyword arguments
        Additional arguments passed to `piptrack`

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin)

    See Also
    --------
    piptrack
        Pitch tracking by parabolic interpolation

    Examples
    --------
    >>> # With time-series input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.estimate_tuning(y=y, sr=sr)
    0.089999999999999969

    >>> # In tenths of a cent
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.estimate_tuning(y=y, sr=sr, resolution=1e-3)
    0.093999999999999972

    >>> # Using spectrogram input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.estimate_tuning(S=S, sr=sr)
    0.089999999999999969

    >>> # Using pass-through arguments to `librosa.piptrack`
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.estimate_tuning(y=y, sr=sr, n_fft=8192,
    ...                         fmax=librosa.note_to_hz('G#9'))
    0.070000000000000062

    '''

    pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)

    # Only count magnitude where frequency is > 0
    pitch_mask = pitch > 0

    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return pitch_tuning(pitch[(mag >= threshold) & pitch_mask],
                        resolution=resolution,
                        bins_per_octave=bins_per_octave)


def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    '''Given a collection of pitches, estimate its tuning offset
    (in fractions of a bin) relative to A440=440.0Hz.

    Parameters
    ----------
    frequencies : array-like, float
        A collection of frequencies detected in the signal.
        See `piptrack`

    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to cents.

    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin)

    See Also
    --------
    estimate_tuning
        Estimating tuning from time-series or spectrogram input

    Examples
    --------
    >>> # Generate notes at +25 cents
    >>> freqs = librosa.cqt_frequencies(24, 55, tuning=0.25)
    >>> librosa.pitch_tuning(freqs)
    0.25

    >>> # Track frequencies from a real spectrogram
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> pitches, magnitudes, stft = librosa.ifptrack(y, sr)
    >>> # Select out pitches with high energy
    >>> pitches = pitches[magnitudes > np.median(magnitudes)]
    >>> librosa.pitch_tuning(pitches)
    0.089999999999999969

    '''

    frequencies = np.atleast_1d(frequencies)

    # Trim out any DC components
    frequencies = frequencies[frequencies > 0]

    if not np.any(frequencies):
        warnings.warn('Trying to estimate tuning from empty frequency set.')
        return 0.0

    # Compute the residual relative to the number of bins
    residual = np.mod(bins_per_octave *
                      time_frequency.hz_to_octs(frequencies), 1.0)

    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05
    # from the next tone up.
    residual[residual >= 0.5] -= 1.0

    bins = np.linspace(-0.5, 0.5, int(np.ceil(1. / resolution)) + 1)

    counts, tuning = np.histogram(residual, bins)

    # return the histogram peak
    return tuning[np.argmax(counts)]


@cache(level=30)
def piptrack(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
             fmin=150.0, fmax=4000.0, threshold=0.1,
             win_length=None, window='hann', center=True, pad_mode='reflect',
             ref=None):
    '''Pitch tracking on thresholded parabolically-interpolated STFT.

    This implementation uses the parabolic interpolation method described by [1]_.

    .. [1] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

    Parameters
    ----------
    y: np.ndarray [shape=(n,)] or None
        audio signal

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    S: np.ndarray [shape=(d, t)] or None
        magnitude or power spectrogram

    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if `y` is provided.

    hop_length : int > 0 [scalar] or None
        number of samples to hop

    threshold : float in `(0, 1)`
        A bin in spectrum `S` is considered a pitch when it is greater than
        `threshold*ref(S)`.

        By default, `ref(S)` is taken to be `max(S, axis=0)` (the maximum value in
        each column).

    fmin : float > 0 [scalar]
        lower frequency cutoff.

    fmax : float > 0 [scalar]
        upper frequency cutoff.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        .. see also:: `filters.get_window`

    center : boolean
        - If `True`, the signal `y` is padded so that frame
          `t` is centered at `y[t * hop_length]`.
        - If `False`, then frame `t` begins at `y[t * hop_length]`

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.

    ref : scalar or callable [default=np.max]
        If scalar, the reference value against which `S` is compared for determining
        pitches.

        If callable, the reference value is computed as `ref(S, axis=0)`.

    .. note::
        One of `S` or `y` must be provided.

        If `S` is not given, it is computed from `y` using
        the default parameters of `librosa.core.stft`.

    Returns
    -------
    pitches : np.ndarray [shape=(d, t)]
    magnitudes : np.ndarray [shape=(d,t)]
        Where `d` is the subset of FFT bins within `fmin` and `fmax`.

        `pitches[f, t]` contains instantaneous frequency at bin
        `f`, time `t`

        `magnitudes[f, t]` contains the corresponding magnitudes.

        Both `pitches` and `magnitudes` take value 0 at bins
        of non-maximal magnitude.

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Computing pitches from a waveform input

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    Or from a spectrogram input

    >>> S = np.abs(librosa.stft(y))
    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr)

    Or with an alternate reference value for pitch detection, where
    values above the mean spectral energy in each frame are counted as pitches

    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr, threshold=1,
    ...                                        ref=np.mean)

    '''

    # Check that we received an audio time series or STFT
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window,
                            center=center, pad_mode=pad_mode)

    # Make sure we're dealing with magnitudes
    S = np.abs(S)

    # Truncate to feasible region
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)

    fft_freqs = time_frequency.fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])

    shift = 2 * S[1:-1] - S[2:] - S[:-2]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (np.abs(shift) < util.tiny(shift)))

    # Pad back up to the same shape as S
    avg = np.pad(avg, ([1, 1], [0, 0]), mode='constant')
    shift = np.pad(shift, ([1, 1], [0, 0]), mode='constant')

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))

    # Compute the column-wise local max of S after thresholding
    # Find the argmax coordinates
    if ref is None:
        ref = np.max

    if six.callable(ref):
        ref_value = threshold * ref(S, axis=0)
    else:
        ref_value = np.abs(ref)

    idx = np.argwhere(freq_mask & util.localmax(S * (S > ref_value)))

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = ((idx[:, 0] + shift[idx[:, 0], idx[:, 1]])
                                     * float(sr) / n_fft)

    mags[idx[:, 0], idx[:, 1]] = (S[idx[:, 0], idx[:, 1]]
                                  + dskew[idx[:, 0], idx[:, 1]])

    return pitches, mags


def yin(y, sr=22050, frame_length=2048, hop_length=None, fmin=40, fmax=None,
        cumulative=False, interpolate=True,
        threshold_1=0.1, threshold_2=0.2, pad_mode='reflect'):
    '''Fundamental frequency (F0) estimation. [1]_


    .. [1] De Cheveigné, A., & Kawahara, H. (2002).
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America, 111(4), 1917-1930.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y` in Hertz

    frame_length : int [scalar]
        length of the frame in samples.
        By default, frame_length=2048 corresponds to a time scale of 93 ms at
        a sampling rate of 22050 Hz. In comparison, the time scale of the
        original publication [1]_ is equal to 25 milliseconds.
        We recommend setting `frame_length` to a power of two for optimizing
        the speed of the fast Fourier transform (FFT) algorithm.

    hop_length : None or int > 0 [scalar]
        number of audio samples between adjacent YIN predictions.
        If `None`, defaults to `frame_length // 4`.

    fmin: number > 0 [scalar]
        minimum frequency in Hertz

    fmax: None or number > 0 [scalar]
        maximum frequency in Hertz. If `None`, defaults to fmax = sr / 4.0

    threshold_1: number >= 0 [scalar]
        absolute threshold for peak estimation

    threshold_2: number >= 0 [scalar]
        absolute threshold for aperiodicity estimation
        NB: we recommend setting threshold_2 >= threshold_1.

    pad_mode: string or function
        This argument is passed to `np.pad` for centering frames before
        computing autocorrelation.
        By default (`pad_mode="reflect"`), `y` is padded on both sides with its
        own reflection, mirrored around its first and last sample respectively.

    Returns
    -------
    f0: np.ndarray [shape=(n_frames,)]
        time series of fundamental frequencies in Hertz.
        Null values represent the absence of a perceptible fundamental frequency.

    See also
    --------
    piptrack: sinusoidal peak interpolation

    Examples
    --------
    Computing a fundamental frequency curve from an audio input:

    >>> y = librosa.chirp(440, 880, duration=7.0)
    >>> yin(y)
    array([  0.        ,   0.        , 222.62295321, ..., 877.12362404,
       885.76086659, 886.80491703])
    '''
    # Set the maximal frequency.
    if fmax is None:
        fmax = sr / 4

    # Set the default hop, if it's not already specified.
    if hop_length is None:
        hop_length = int(frame_length // 4)

    # Check that audio is valid.
    util.valid_audio(y, mono=True)

    # Pad the time series so that frames are centered.
    y = np.pad(y, int(frame_length // 2), mode=pad_mode)

    # Compute autocorrelation in each frame.
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)
    n_frames = y_frames.shape[1]
    acf_frames = librosa.autocorrelate(y_frames, axis=0)

    # Difference function.
    min_period = np.maximum(int(np.floor(sr/fmax)), 2)
    max_period = np.minimum(int(np.ceil(sr/fmin)), frame_length-1)
    boxcar_window = scipy.signal.windows.boxcar(frame_length)
    energy = np.convolve(y*y, boxcar_window, mode="same")
    energy_frames = util.frame(
        energy, frame_length=frame_length, hop_length=hop_length)
    energy_0 = energy_frames[0, :]
    energy_tau = energy_frames[(min_period-1):(max_period+1), :]
    acf_tau = acf_frames[(min_period-1):(max_period+1), :]
    yin_frames = energy_0 + energy_frames - 2*acf_frames

    # Cumulative mean normalized difference function.
    # NB: Equation 8 in de Cheveigné and Kawahara JASA 2002 seems to imply that
    # the denominator is: cumsum(difference_fames)/tau_range, not
    # cumsum(difference_frames/tau_range) as we implement it.
    # However, going back to line 48 of the MATLAB code by AdC, we find the line:
    # dd= d(2:end) ./ (cumsum(d(2:end)) ./ (1:(p.maxprd)));
    # in which the parentheses indicate that the division by tau must happen
    # before the cumulative summation, not after.
    # Therefore, in this implementation, we purposefully diverge from Equation 8
    # and follow the MATLAB reference implementation instead.
    if cumulative:
        yin_numerator = yin_frames[(min_period-1):(max_period+1), :]
        tau_range = np.arange(1, frame_length)[:, np.newaxis]
        cumulative_mean = np.cumsum(yin_frames[1:, :]/tau_range, axis=0)
        epsilon = np.finfo(y.dtype).eps
        yin_denominator = epsilon + cumulative_mean[(min_period-2):max_period, :]
        yin_frames = yin_numerator / yin_denominator

    # Parabolic interpolation.
    # NB: perhaps these operations can be fused to alleviate memory allocation?
    if interpolate:
        parabola_a =\
            (yin_frames[:-2, :]+yin_frames[2:, :]-2*yin_frames[1:-1, :]) / 2
        parabola_b = (yin_frames[2:, :]-yin_frames[:-2, :]) / 2
        parabolic_shifts = -parabola_b / (2*parabola_a)
        parabolic_values =\
            yin_frames[1:-1, :] - parabola_b*parabola_b/(4*parabola_a)
        is_trough = np.logical_and(
            yin_frames[1:-1, :]<yin_frames[:-2, :],
            yin_frames[1:-1, :]<yin_frames[2:, :])
        yin_frames = yin_frames[1:-1, :]
        yin_frames[is_trough] = parabolic_values[is_trough]
    else:
        yin_frames = yin_frames[1:-1, :]

    # Absolute threshold
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    yin_period = np.argmin(yin_frames, axis=0)
    if threshold_1 > 0:
        lower_bound = (yin_frames<threshold_1).argmax(axis=0)
        upper_bound = np.minimum(2*lower_bound, max_period-min_period-1)
        for frame_id in range(n_frames):
            if lower_bound[frame_id]>0:
                bounded_frame = yin_frames[
                    lower_bound[frame_id]:upper_bound[frame_id], frame_id]
                new_argmin = np.argmin(bounded_frame, axis=0)
                yin_period[frame_id] = lower_bound[frame_id] + new_argmin
    yin_aperiodicity = yin_frames[yin_period, range(n_frames)]
    yin_period = min_period + yin_period

    # Refine peak by parabolic interpolation
    if interpolate:
        yin_period = yin_period + parabolic_shifts[yin_period, range(n_frames)]

    # Convert period to fundamental frequency (f0)
    f0 = sr / yin_period
    f0[yin_aperiodicity > threshold_2] = 0
    f0[f0<fmin] = 0
    f0[f0>fmax] = 0
    return f0
