#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display
=======
.. autosummary::
    :toctree: generated/

    specshow
    waveplot
    time_ticks
    cmap
"""

import copy
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt

from . import cache
from . import core
from . import util
from .util.exceptions import ParameterError

_HAS_SEABORN = False
try:
    _matplotlibrc = copy.deepcopy(mpl.rcParams)
    import seaborn as sns
    _HAS_SEABORN = True
    mpl.rcParams.update(**_matplotlibrc)
except ImportError:
    pass


# This function wraps xticks or yticks: star-args is okay
def time_ticks(locs, *args, **kwargs):  # pylint: disable=star-args
    '''Plot time-formatted axis ticks.

    Parameters
    ----------
    locations : list or np.ndarray
        Time-stamps for tick marks

    n_ticks : int > 0 or None
        Show this number of ticks (evenly spaced).

        If none, all ticks are displayed.

        Default: 5

    axis : 'x' or 'y'
        Which axis should the ticks be plotted on?
        Default: 'x'

    time_fmt : None or {'ms', 's', 'm', 'h'}
        - 'ms': milliseconds   (eg, 241ms)
        - 's': seconds         (eg, 1.43s)
        - 'm': minutes         (eg, 1:02)
        - 'h': hours           (eg, 1:02:03)

        If none, formatted is automatically selected by the
        range of the times data.

        Default: None

    fmt : str
        .. warning:: This parameter name was in librosa 0.4.2
            Use the `time_fmt` parameter instead.
            The `fmt` parameter will be removed in librosa 0.5.0.

    kwargs : additional keyword arguments.
        See `matplotlib.pyplot.xticks` or `yticks` for details.


    Returns
    -------
    locs
    labels
        Locations and labels of tick marks


    See Also
    --------
    matplotlib.pyplot.xticks
    matplotlib.pyplot.yticks


    Examples
    --------
    >>> # Tick at pre-computed beat times
    >>> librosa.display.specshow(S)
    >>> librosa.display.time_ticks(beat_times)

    >>> # Set the locations of the time stamps
    >>> librosa.display.time_ticks(locations, timestamps)

    >>> # Format in seconds
    >>> librosa.display.time_ticks(beat_times, time_fmt='s')

    >>> # Tick along the y axis
    >>> librosa.display.time_ticks(beat_times, axis='y')

    '''

    n_ticks = kwargs.pop('n_ticks', 5)
    axis = kwargs.pop('axis', 'x')
    fmt = kwargs.pop('fmt', util.Deprecated())
    time_fmt = kwargs.pop('time_fmt', None)

    time_fmt = util.rename_kw('fmt', fmt, 'time_fmt', time_fmt, '0.4.2', '0.5.0')

    if axis == 'x':
        ticker = plt.xticks
    elif axis == 'y':
        ticker = plt.yticks
    else:
        raise ParameterError("axis must be either 'x' or 'y'.")

    if len(args) > 0:
        times = args[0]
    else:
        times = locs
        locs = np.arange(len(times))

    if n_ticks is not None:
        # Slice the locations and labels evenly between 0 and the last point
        positions = np.linspace(0, len(locs)-1, n_ticks,
                                endpoint=True).astype(int)
        locs = locs[positions]
        times = times[positions]

    # Format the labels by time
    formats = {'ms': lambda t: '{:d}ms'.format(int(1e3 * t)),
               's': '{:0.2f}s'.format,
               'm': lambda t: '{:d}:{:02d}'.format(int(t / 6e1),
                                                   int(np.mod(t, 6e1))),
               'h': lambda t: '{:d}:{:02d}:{:02d}'.format(int(t / 3.6e3),
                                                          int(np.mod(t / 6e1,
                                                                     6e1)),
                                                          int(np.mod(t, 6e1)))}

    if time_fmt is None:
        if max(times) > 3.6e3:
            time_fmt = 'h'
        elif max(times) > 6e1:
            time_fmt = 'm'
        elif max(times) > 1.0:
            time_fmt = 's'
        else:
            time_fmt = 'ms'

    elif time_fmt not in formats:
        raise ParameterError('Invalid format: {:s}'.format(time_fmt))

    times = [formats[time_fmt](t) for t in times]

    return ticker(locs, times, **kwargs)


def frequency_ticks(locs, *args, **kwargs):  # pylint: disable=star-args
    '''Plot frequency-formatted axis ticks.

    Parameters
    ----------
    locations : list or np.ndarray
        Frequency values for tick marks

    n_ticks : int > 0 or None
        Show this number of ticks (evenly spaced).

        If none, all ticks are displayed.

        Default: 5

    axis : 'x' or 'y'
        Which axis should the ticks be plotted on?
        Default: 'x'

    freq_fmt : None or {'mHz', 'Hz', 'kHz', 'MHz', 'GHz'}
        - 'mHz': millihertz
        - 'Hz': hertz
        - 'kHz': kilohertz
        - 'MHz': megahertz
        - 'GHz': gigahertz

        If none, formatted is automatically selected by the
        range of the frequency data.

        Default: None

    kwargs : additional keyword arguments.
        See `matplotlib.pyplot.xticks` or `yticks` for details.


    Returns
    -------
    (locs, labels)
        Locations and labels of tick marks

    label
        Axis label

    See Also
    --------
    matplotlib.pyplot.xticks
    matplotlib.pyplot.yticks


    Examples
    --------
    >>> # Tick at pre-computed beat times
    >>> librosa.display.specshow(S)
    >>> librosa.display.frequency_ticks()

    >>> # Set the locations of the time stamps
    >>> librosa.display.frequency_ticks(locations, frequencies)

    >>> # Format in hertz
    >>> librosa.display.frequency_ticks(frequencies, freq_fmt='Hz')

    >>> # Tick along the y axis
    >>> librosa.display.frequency_ticks(frequencies, axis='y')

    '''

    n_ticks = kwargs.pop('n_ticks', 5)
    axis = kwargs.pop('axis', 'x')
    freq_fmt = kwargs.pop('freq_fmt', None)

    if axis == 'x':
        ticker = plt.xticks
    elif axis == 'y':
        ticker = plt.yticks
    else:
        raise ParameterError("axis must be either 'x' or 'y'.")

    if len(args) > 0:
        freqs = args[0]
    else:
        freqs = locs
        locs = np.arange(len(freqs))

    if n_ticks is not None:
        # Slice the locations and labels evenly between 0 and the last point
        positions = np.linspace(0, len(locs)-1, n_ticks,
                                endpoint=True).astype(int)
        locs = locs[positions]
        freqs = freqs[positions]

    # Format the labels by time
    formats = {'mHz': lambda f: '{:.5g}'.format(f * 1e3),
               'Hz': '{:.5g}'.format,
               'kHz': lambda f: '{:.5g}'.format(f * 1e-3),
               'MHz': lambda f: '{:.5g}'.format(f * 1e-6),
               'GHz': lambda f: '{:.5g}'.format(f * 1e-9)}

    f_max = np.max(freqs)

    if freq_fmt is None:
        if f_max > 1e10:
            freq_fmt = 'GHz'
        elif f_max > 1e7:
            freq_fmt = 'MHz'
        elif f_max > 1e4:
            freq_fmt = 'kHz'
        elif f_max > 1e1:
            freq_fmt = 'Hz'
        else:
            freq_fmt = 'mHz'

    elif freq_fmt not in formats:
        raise ParameterError('Invalid format: {:s}'.format(freq_fmt))

    ticks = [formats[freq_fmt](f) for f in freqs]

    return ticker(locs, ticks, **kwargs), freq_fmt


@cache
def cmap(data, use_sns=True, robust=True):
    '''Get a default colormap from the given data.

    If the data is boolean, use a black and white colormap.

    If the data has both positive and negative values,
    use a diverging colormap ('coolwarm').

    Otherwise, use a sequential map: either cubehelix or 'OrRd'.

    Parameters
    ----------
    data : np.ndarray
        Input data

    use_sns : bool
        If True, and `seaborn` is installed, use cubehelix maps for
        sequential data

    robust : bool
        If True, discard the top and bottom 2% of data when calculating
        range.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        - If `data` has dtype=boolean, `cmap` is 'gray_r'
        - If `data` has only positive or only negative values,
          `cmap` is 'OrRd' (`use_sns==False`) or cubehelix
        - If `data` has both positive and negatives, `cmap` is 'coolwarm'

    See Also
    --------
    matplotlib.pyplot.colormaps
    seaborn.cubehelix_palette
    '''

    data = np.atleast_1d(data)

    if data.dtype == 'bool':
        return plt.get_cmap('gray_r')

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    max_val = np.percentile(data, max_p)
    min_val = np.percentile(data, min_p)

    if min_val >= 0 or max_val <= 0:
        if use_sns and _HAS_SEABORN:
            return sns.cubehelix_palette(light=1.0, as_cmap=True)
        else:
            return plt.get_cmap('OrRd')

    return plt.get_cmap('coolwarm')


def __envelope(x, hop):
    '''Compute the max-envelope of x at a stride/frame length of h'''
    return util.frame(x, hop_length=hop, frame_length=hop).max(axis=0)


def waveplot(y, sr=22050, max_points=5e4, x_axis='time', offset=0.0, max_sr=1000,
             time_fmt=None, **kwargs):
    '''Plot the amplitude envelope of a waveform.

    If `y` is monophonic, a filled curve is drawn between `[-abs(y), abs(y)]`.

    If `y` is stereo, the curve is drawn between `[-abs(y[1]), abs(y[0])]`,
    so that the left and right channels are drawn above and below the axis,
    respectively.

    Long signals (`duration >= max_points`) are down-sampled to at
    most `max_sr` before plotting.

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)

    sr : number > 0 [scalar]
        sampling rate of `y`

    max_points : postive number or None
        Maximum number of time-points to plot: if `max_points` exceeds
        the duration of `y`, then `y` is downsampled.

        If `None`, no downsampling is performed.

    x_axis : str {'time', 'off', 'none'} or None
        If 'time', the x-axis is given time tick-marks.

        See also: `time_ticks`

    offset : float
        Horizontal offset (in time) to start the waveform plot

    max_sr : number > 0 [scalar]
        Maximum sampling rate for the visualization

    time_fmt : None or str
        Formatting for time axis.  None (automatic) by default.

        See `time_ticks`.

    kwargs
        Additional keyword arguments to `matplotlib.pyplot.fill_between`

    Returns
    -------
    pc : matplotlib.collections.PolyCollection
        The PolyCollection created by `fill_between`.

    See also
    --------
    time_ticks
    librosa.core.resample
    matplotlib.pyplot.fill_between


    Examples
    --------
    Plot a monophonic waveform

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=10)
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.waveplot(y, sr=sr)
    >>> plt.title('Monophonic')

    Or a stereo waveform

    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      mono=False, duration=10)
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.waveplot(y, sr=sr)
    >>> plt.title('Stereo')

    Or harmonic and percussive components with transparency

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=10)
    >>> y_harm, y_perc = librosa.effects.hpss(y)
    >>> plt.subplot(3, 1, 3)
    >>> librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
    >>> librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
    >>> plt.title('Harmonic + Percussive')
    >>> plt.tight_layout()
    '''

    util.valid_audio(y, mono=False)

    if not (isinstance(max_sr, int) and max_sr > 0):
        raise ParameterError('max_sr must be a non-negative integer')

    target_sr = sr

    if max_points is not None:
        if max_points <= 0:
            raise ParameterError('max_points must be strictly positive')

        if max_points < y.shape[-1]:
            target_sr = min(max_sr, (sr * y.shape[-1]) // max_points)

        hop_length = sr // target_sr

        if y.ndim == 1:
            y = __envelope(y, hop_length)
        else:
            y = np.vstack([__envelope(_, hop_length) for _ in y])

    if y.ndim > 1:
        y_top = y[0]
        y_bottom = -y[1]
    else:
        y_top = y
        y_bottom = -y

    axes = plt.gca()

    if hasattr(axes._get_lines, 'prop_cycler'):
        # matplotlib >= 1.5
        kwargs.setdefault('color', next(axes._get_lines.prop_cycler)['color'])
    else:
        # matplotlib 1.4
        kwargs.setdefault('color', next(axes._get_lines.color_cycle))

    sample_off = core.time_to_samples(offset, sr=target_sr)

    locs = np.arange(sample_off, sample_off + len(y_top))
    out = axes.fill_between(locs, y_bottom, y_top, **kwargs)

    plt.xlim([locs[0], locs[-1]])

    if x_axis == 'time':
        time_ticks(locs, core.samples_to_time(locs, sr=target_sr), time_fmt=time_fmt)
    elif x_axis is None or x_axis in ['off', 'none']:
        plt.xticks([])
    else:
        raise ParameterError('Unknown x_axis value: {}'.format(x_axis))

    return out


def specshow(data, sr=22050, hop_length=512, x_axis=None, y_axis=None,
             n_xticks=5, n_yticks=5, fmin=None, fmax=None, bins_per_octave=12,
             tmin=16, tmax=240, freq_fmt='Hz', time_fmt=None, **kwargs):
    '''Display a spectrogram/chromagram/cqt/etc.

    Functions as a drop-in replacement for `matplotlib.pyplot.imshow`,
    but with useful defaults.


    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        Matrix to display (e.g., spectrogram)

    sr : number > 0 [scalar]
        Sample rate used to determine time scale in x-axis.

    hop_length : int > 0 [scalar]
        Hop length, also used to determine time scale in x-axis

    x_axis : None or str

    y_axis : None or str
        Range for the x- and y-axes.

        Valid types are:

        - None or 'off' : no axis is displayed.

        Frequency types:

        - 'linear' : frequency range is determined by the FFT window
          and sampling rate.
        - 'log' : the image is displayed on a vertical log scale.
        - 'mel' : frequencies are determined by the mel scale.
        - 'cqt_hz' : frequencies are determined by the CQT scale.
        - 'cqt_note' : pitches are determined by the CQT scale.
        - 'chroma' : pitches are determined by the chroma filters.
        - 'tonnetz' : axes are labeled by Tonnetz dimensions

        Time types:

        - 'time' : markers are shown as milliseconds, seconds,
          minutes, or hours
        - 'lag' : like time, but past the half-way point counts
          as negative values.
        - 'frames' : markers are shown as frame counts.
        - 'tempo' : markers are shown as beats-per-minute

    n_xticks : int > 0 [scalar]
        If x_axis is drawn, the number of ticks to show

    n_yticks : int > 0 [scalar]
        If y_axis is drawn, the number of ticks to show

    fmin : float > 0 [scalar] or None
        Frequency of the lowest spectrogram bin.  Used for Mel and CQT
        scales.

        If `y_axis` is `cqt_hz` or `cqt_note` and `fmin` is not given,
        it is set by default to `note_to_hz('C1')`.

    fmax : float > 0 [scalar] or None
        Used for setting the Mel frequency scales

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave.  Used for CQT frequency scale.

    tmin : float > 0 [scalar]
    tmax : float > 0 [scalar]
        Minimum and maximum tempi displayed when `_axis='tempo'`,
        as measured in beats per minute.

    freq_fmt : None or str
        Formatting for frequency axes.  'Hz', by default.

        See `frequency_ticks`.

    time_fmt : None or str
        Formatting for time axes.  None (automatic) by default.

        See `time_ticks`.

    kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.imshow`.


    Returns
    -------
    image : `matplotlib.image.AxesImage`
        As returned from `matplotlib.pyplot.imshow`.


    See Also
    --------
    cmap : Automatic colormap detection

    time_ticks : time-formatted tick marks

    frequency_ticks : frequency-formatted tick marks

    matplotlib.pyplot.imshow


    Examples
    --------
    Visualize an STFT power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> plt.figure(figsize=(12, 8))

    >>> D = librosa.logamplitude(np.abs(librosa.stft(y))**2, ref_power=np.max)
    >>> plt.subplot(4, 2, 1)
    >>> librosa.display.specshow(D, y_axis='linear')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Linear-frequency power spectrogram')


    Or on a logarithmic scale

    >>> plt.subplot(4, 2, 2)
    >>> librosa.display.specshow(D, y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-frequency power spectrogram')


    Or use a CQT scale

    >>> CQT = librosa.logamplitude(librosa.cqt(y, sr=sr)**2, ref_power=np.max)
    >>> plt.subplot(4, 2, 3)
    >>> librosa.display.specshow(CQT, y_axis='cqt_note')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Constant-Q power spectrogram (note)')

    >>> plt.subplot(4, 2, 4)
    >>> librosa.display.specshow(CQT, y_axis='cqt_hz')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Constant-Q power spectrogram (Hz)')


    Draw a chromagram with pitch classes

    >>> C = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> plt.subplot(4, 2, 5)
    >>> librosa.display.specshow(C, y_axis='chroma')
    >>> plt.colorbar()
    >>> plt.title('Chromagram')


    Force a grayscale colormap (white -> black)

    >>> plt.subplot(4, 2, 6)
    >>> librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Linear power spectrogram (grayscale)')


    Draw time markers automatically

    >>> plt.subplot(4, 2, 7)
    >>> librosa.display.specshow(D, x_axis='time', y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log power spectrogram')


    Draw a tempogram with BPM markers

    >>> plt.subplot(4, 2, 8)
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr)
    >>> tempo = librosa.beat.estimate_tempo(oenv, sr=sr)
    >>> Tgram = librosa.feature.tempogram(y=y, sr=sr)
    >>> librosa.display.specshow(Tgram[:100], x_axis='time', y_axis='tempo',
    ...                          tmin=tempo/4, tmax=tempo*2, n_yticks=4)
    >>> plt.colorbar()
    >>> plt.title('Tempogram')
    >>> plt.tight_layout()


    '''

    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('interpolation', 'nearest')

    if np.issubdtype(data.dtype, np.complex):
        warnings.warn('Trying to display complex-valued input. '
                      'Showing magnitude instead.')
        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))

    axes = plt.imshow(data, **kwargs)

    all_params = dict(kwargs=kwargs,
                      sr=sr,
                      fmin=fmin,
                      fmax=fmax,
                      bins_per_octave=bins_per_octave,
                      tmin=tmin,
                      tmax=tmax,
                      hop_length=hop_length,
                      time_fmt=time_fmt,
                      freq_fmt=freq_fmt)

    # Scale and decorate the axes
    __axis(data, n_xticks, x_axis, horiz=True, minor=y_axis, **all_params)
    __axis(data, n_yticks, y_axis, horiz=False, minor=x_axis, **all_params)

    return axes


def __get_shape_artists(data, horiz):
    '''Return size, ticker, and labeler'''
    if horiz:
        return data.shape[1], plt.xticks, plt.xlabel
    else:
        return data.shape[0], plt.yticks, plt.ylabel


def __axis(data, n_ticks, ax_type, horiz=False, **kwargs):
    '''Dispatch function to decorate axes'''
    axis_map = {'linear': __axis_linear,
                'log': __axis_log,
                'mel': __axis_mel,
                'cqt_hz': __axis_cqt_hz,
                'cqt_note': __axis_cqt_note,
                'chroma': __axis_chroma,
                'tonnetz': __axis_tonnetz,
                'off': __axis_none,
                'time': __axis_time,
                'tempo': __axis_tempo,
                'lag': __axis_lag,
                'frames': __axis_frames}

    if ax_type is None:
        ax_type = 'off'

    if ax_type not in axis_map:
        raise ParameterError('Unknown axis type: {:s}'.format(ax_type))

    func = axis_map[ax_type]

    func(data, n_ticks, horiz=horiz, **kwargs)


def __axis_none(data, n_ticks, horiz, **_kwargs):
    '''Empty axis artist'''

    _, ticker, labeler = __get_shape_artists(data, horiz)

    ticker([])
    labeler('')


def __axis_log(data, n_ticks, horiz, sr=22050, kwargs=None,
               minor=None, **_kwargs):
    '''Plot a log-scaled image'''

    axes_phantom = plt.gca()

    if kwargs is None:
        kwargs = dict()

    aspect = kwargs.pop('aspect', None)
    fmt = _kwargs.pop('freq_fmt', 'Hz')

    n, ticker, labeler = __get_shape_artists(data, horiz)
    t_log, t_inv = __log_scale(n)

    if horiz:
        axis = 'x'

        if minor == 'log':
            ax2 = __log_scale(data.shape[0])[0]
        else:
            ax2 = np.arange(data.shape[0])
        ax1 = t_log
    else:
        axis = 'y'
        if minor == 'log':
            ax1 = __log_scale(data.shape[1])[0]
        else:
            ax1 = np.arange(data.shape[1])

        ax2 = t_log

    args = (ax1, ax2, data)

    # NOTE:  2013-11-14 16:15:33 by Brian McFee <brm2132@columbia.edu>
    #  We draw the image twice here. This is a hack to get around
    #  NonUniformImage not properly setting hooks for color.
    #  Drawing twice enables things like colorbar() to work properly.

    im_phantom = img.NonUniformImage(axes_phantom,
                                     extent=(args[0].min(), args[0].max(),
                                             args[1].min(), args[1].max()),
                                     **kwargs)
    im_phantom.set_data(*args)

    kwargs['aspect'] = aspect

    axes_phantom.images[0] = im_phantom

    positions = np.linspace(0, n-1, n_ticks, endpoint=True).astype(int)
    # One extra value here to catch nyquist
    values = np.linspace(0, 0.5 * sr, n, endpoint=True)

    _, label = frequency_ticks(positions, values[t_inv[positions]],
                               n_ticks=None, axis=axis, freq_fmt=fmt)
    labeler(label)


def __axis_mel(data, n_ticks, horiz, fmin=None, fmax=None, **_kwargs):
    '''Mel-scaled axes'''

    fmt = _kwargs.pop('freq_fmt', 'Hz')

    if horiz:
        axis = 'x'
    else:
        axis = 'y'

    n, ticker, labeler = __get_shape_artists(data, horiz)

    positions = np.linspace(0, n-1, n_ticks).astype(int)

    kwargs = {}

    if fmin is not None:
        kwargs['fmin'] = fmin

    if fmax is not None:
        kwargs['fmax'] = fmax

    # only two star-args here, defined immediately above
    # pylint: disable=star-args
    values = core.mel_frequencies(n_mels=n+2, **kwargs)[positions]
    _, label = frequency_ticks(positions, values,
                               n_ticks=None, axis=axis, freq_fmt=fmt)
    labeler(label)


def __axis_chroma(data, n_ticks, horiz, bins_per_octave=12, **_kwargs):
    '''Chroma axes'''

    n, ticker, labeler = __get_shape_artists(data, horiz)

    # Generate the template positions: C D E F G A B
    pos = np.asarray([0, 2, 4, 5, 7, 9, 11]) * bins_per_octave // 12

    n_octaves = np.ceil(n / float(bins_per_octave))

    positions = pos.copy()
    for i in range(1, int(n_octaves)):
        positions = np.append(positions, pos + i * bins_per_octave, axis=0)

    values = core.midi_to_note(positions * 12 // bins_per_octave, octave=False)
    ticker(positions[:n], values[:n])
    labeler('Pitch class')


def __axis_linear(data, n_ticks, horiz, sr=22050, **_kwargs):
    '''Linear frequency axes'''
    fmt = _kwargs.pop('freq_fmt', 'Hz')

    if horiz:
        axis = 'x'
    else:
        axis = 'y'

    n, ticker, labeler = __get_shape_artists(data, horiz)

    positions = np.linspace(0, n - 1, n_ticks, endpoint=True).astype(int)
    values = (sr * np.linspace(0, 0.5, n_ticks, endpoint=True))

    _, label = frequency_ticks(positions, values,
                               n_ticks=None, axis=axis, freq_fmt=fmt)
    labeler(label)


def __axis_cqt(data, n_ticks, horiz, note=False, fmin=None,
               bins_per_octave=12, **_kwargs):
    '''CQT axes'''
    if fmin is None:
        fmin = core.note_to_hz('C1')

    if horiz:
        axis = 'x'
    else:
        axis = 'y'

    fmt = _kwargs.pop('freq_fmt', 'Hz')

    n, ticker, labeler = __get_shape_artists(data, horiz)

    positions = np.linspace(0, n-1, num=n_ticks, endpoint=True).astype(int)

    values = core.cqt_frequencies(n + 1,
                                  fmin=fmin,
                                  bins_per_octave=bins_per_octave)

    if note:
        values = core.hz_to_note(values[positions])
        label = 'Note'
        ticker(positions, values)
    else:
        values = values[positions]
        _, label = frequency_ticks(positions, values,
                                   n_ticks=None, axis=axis, freq_fmt=fmt)

    labeler(label)


def __axis_cqt_hz(*args, **kwargs):
    '''CQT in Hz'''
    kwargs['note'] = False
    __axis_cqt(*args, **kwargs)


def __axis_cqt_note(*args, **kwargs):
    '''CQT in notes'''
    kwargs['note'] = True
    __axis_cqt(*args, **kwargs)


def __axis_time(data, n_ticks, horiz, sr=22050, hop_length=512, **_kwargs):
    '''Time axes'''
    n, ticker, labeler = __get_shape_artists(data, horiz)

    if horiz:
        axis = 'x'
    else:
        axis = 'y'

    fmt = _kwargs.pop('time_fmt', None)

    positions = np.linspace(0, n-1, n_ticks, endpoint=True).astype(int)

    time_ticks(positions,
               core.frames_to_time(positions, sr=sr, hop_length=hop_length),
               n_ticks=None, time_fmt=fmt, axis=axis)

    labeler('Time')


def __axis_tempo(data, n_ticks, horiz, sr=22050, hop_length=512, tmin=16, tmax=240, **_kwargs):
    '''Tempo axes'''
    n, ticker, labeler = __get_shape_artists(data, horiz)

    nmin = min(n-1, sr * 60.0 / (hop_length * tmin))
    nmax = max(1, sr * 60.0 / (hop_length * tmax))

    positions = np.logspace(np.log2(nmin), np.log2(nmax),
                            num=n_ticks, endpoint=True, base=2).astype(int)

    tempi = ['{:.1f}'.format(60 * float(sr) / (hop_length * t)) for t in positions]
    ticker(positions, tempi)
    labeler('Tempo (BPM)')



def __axis_lag(data, n_ticks, horiz, sr=22050, hop_length=512, **_kwargs):
    '''Lag axes'''
    n, ticker, labeler = __get_shape_artists(data, horiz)

    if horiz:
        axis = 'x'
    else:
        axis = 'y'

    positions = np.linspace(0, n-1, n_ticks, endpoint=True).astype(int)
    times = core.frames_to_time(positions, sr=sr, hop_length=hop_length)
    times[positions >= n//2] -= times[-1]

    time_ticks(positions, times, n_ticks=None, axis=axis)

    labeler('Lag')


def __axis_tonnetz(data, n_ticks, horiz, **_kwargs):
    '''Chroma axes'''

    n, ticker, labeler = __get_shape_artists(data, horiz)

    positions = np.arange(6)

    values = [r'5$_x$', r'5$_y$',
              r'm3$_x$', r'm3$_y$',
              r'M3$_x$', r'M3$_y$']

    ticker(positions, values)
    labeler('Tonnetz')


def __axis_frames(data, n_ticks, horiz, label='Frames', **_kwargs):
    '''Frame axes'''
    n, ticker, labeler = __get_shape_artists(data, horiz)

    positions = np.linspace(0, n-1, n_ticks, endpoint=True).astype(int)

    ticker(positions, positions)
    labeler(label)


def __log_scale(n):
    '''Return a log-scale mapping of bins 0..n, and its inverse.

    Parameters
    ----------
    n : int > 0
        Number of bins

    Returns
    -------
    y   : np.ndarray, shape=(n,)

    y_inv   : np.ndarray, shape=(n+1,)
    '''

    logn = np.log2(n)
    y = n * (1 - np.logspace(-logn, 0, n, base=2, endpoint=True))[::-1]
    y = y.astype(int)

    y_inv = np.arange(len(y))
    for i in range(len(y)-1):
        y_inv[y[i]:y[i+1]] = i

    return y, y_inv
