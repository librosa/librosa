#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display
=======
.. autosummary::
    :toctree: generated/

    specshow
    waveplot
    cmap

    TimeFormatter
    NoteFormatter
    LogHzFormatter
    ChromaFormatter
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter, FixedFormatter, ScalarFormatter
from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator
from matplotlib.ticker import SymmetricalLogLocator

from . import core
from . import util
from .util.exceptions import ParameterError

__all__ = ['specshow',
           'waveplot',
           'cmap',
           'TimeFormatter',
           'NoteFormatter',
           'LogHzFormatter',
           'ChromaFormatter']


class TimeFormatter(Formatter):
    '''A tick formatter for time axes.

    Automatically switches between seconds, minutes:seconds,
    or hours:minutes:seconds.

    Parameters
    ----------
    lag : bool
        If `True`, then the time axis is interpreted in lag coordinates.
        Anything past the mid-point will be converted to negative time.


    See also
    --------
    matplotlib.ticker.Formatter


    Examples
    --------

    For normal time

    >>> import matplotlib.pyplot as plt
    >>> times = np.arange(30)
    >>> values = np.random.randn(len(times))
    >>> plt.figure()
    >>> ax = plt.gca()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    >>> ax.set_xlabel('Time')

    For lag plots

    >>> times = np.arange(60)
    >>> values = np.random.randn(len(times))
    >>> plt.figure()
    >>> ax = plt.gca()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(lag=True))
    >>> ax.set_xlabel('Lag')
    '''

    def __init__(self, lag=False):

        self.lag = lag

    def __call__(self, x, pos=None):
        '''Return the time format as pos'''

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ''
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = '-'
        else:
            value = x
            sign = ''

        if vmax - vmin > 3600:
            s = '{:d}:{:02d}:{:02d}'.format(int(value / 3600.0),
                                            int(np.mod(value / 60.0, 60)),
                                            int(np.mod(value, 60)))
        elif vmax - vmin > 60:
            s = '{:d}:{:02d}'.format(int(value / 60.0),
                                     int(np.mod(value, 60)))
        else:
            s = '{:.2g}'.format(value)

        return '{:s}{:s}'.format(sign, s)


class NoteFormatter(Formatter):
    '''Ticker formatter for Notes

    Parameters
    ----------
    octave : bool
        If `True`, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If `True`, ticks are always labeled.

        If `False`, ticks are only labeled if the span is less than 2 octaves

    See also
    --------
    LogHzFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> plt.figure()
    >>> ax1 = plt.subplot(2,1,1)
    >>> ax1.bar(np.arange(len(values)), values)
    >>> ax1.set_ylabel('Hz')
    >>> ax2 = plt.subplot(2,1,2)
    >>> ax2.bar(np.arange(len(values)), values)
    >>> ax2.yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax2.set_ylabel('Note')
    '''
    def __init__(self, octave=True, major=True):

        self.octave = octave
        self.major = major

    def __call__(self, x, pos=None):

        if x <= 0:
            return ''

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ''

        cents = vmax <= 2 * max(1, vmin)

        return core.hz_to_note(int(x), octave=self.octave, cents=cents)[0]


class LogHzFormatter(Formatter):
    '''Ticker formatter for logarithmic frequency

    Parameters
    ----------
    major : bool
        If `True`, ticks are always labeled.

        If `False`, ticks are only labeled if the span is less than 2 octaves

    See also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> plt.figure()
    >>> ax1 = plt.subplot(2,1,1)
    >>> ax1.bar(np.arange(len(values)), values)
    >>> ax1.yaxis.set_major_formatter(librosa.display.LogHzFormatter())
    >>> ax1.set_ylabel('Hz')
    >>> ax2 = plt.subplot(2,1,2)
    >>> ax2.bar(np.arange(len(values)), values)
    >>> ax2.yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax2.set_ylabel('Note')
    '''
    def __init__(self, major=True):

        self.major = major

    def __call__(self, x, pos=None):

        if x <= 0:
            return ''

        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ''

        return '{:g}'.format(x)


class ChromaFormatter(Formatter):
    '''A formatter for chroma axes

    See also
    --------
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(12)
    >>> plt.figure()
    >>> ax = plt.gca()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.ChromaFormatter())
    >>> ax.set_ylabel('Pitch class')
    '''
    def __call__(self, x, pos=None):
        '''Format for chroma positions'''
        return core.midi_to_note(int(x), octave=False, cents=False)


# A fixed formatter for tonnetz
TONNETZ_FORMATTER = FixedFormatter([r'5$_x$', r'5$_y$',
                                    r'm3$_x$', r'm3$_y$',
                                    r'M3$_x$', r'M3$_y$'])


def cmap(data, robust=True, cmap_seq='magma', cmap_bool='gray_r', cmap_div='coolwarm'):
    '''Get a default colormap from the given data.

    If the data is boolean, use a black and white colormap.

    If the data has both positive and negative values,
    use a diverging colormap.

    Otherwise, use a sequential colormap.

    Parameters
    ----------
    data : np.ndarray
        Input data

    robust : bool
        If True, discard the top and bottom 2% of data when calculating
        range.

    cmap_seq : str
        The sequential colormap name

    cmap_bool : str
        The boolean colormap name

    cmap_div : str
        The diverging colormap name

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap to use for `data`

    See Also
    --------
    matplotlib.pyplot.colormaps
    '''

    data = np.atleast_1d(data)

    if data.dtype == 'bool':
        return plt.get_cmap(cmap_bool)

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    max_val = np.percentile(data, max_p)
    min_val = np.percentile(data, min_p)

    if min_val >= 0 or max_val <= 0:
        return plt.get_cmap(cmap_seq)

    return plt.get_cmap(cmap_div)


def __envelope(x, hop):
    '''Compute the max-envelope of x at a stride/frame length of h'''
    return util.frame(x, hop_length=hop, frame_length=hop).max(axis=0)


def waveplot(y, sr=22050, max_points=5e4, x_axis='time', offset=0.0, max_sr=1000,
             **kwargs):
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

    offset : float
        Horizontal offset (in time) to start the waveform plot

    max_sr : number > 0 [scalar]
        Maximum sampling rate for the visualization

    kwargs
        Additional keyword arguments to `matplotlib.pyplot.fill_between`

    Returns
    -------
    pc : matplotlib.collections.PolyCollection
        The PolyCollection created by `fill_between`.

    See also
    --------
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
    hop_length = 1

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

    kwargs.setdefault('color', next(axes._get_lines.prop_cycler)['color'])

    locs = offset + core.frames_to_time(np.arange(len(y_top)),
                                        sr=sr,
                                        hop_length=hop_length)
    out = axes.fill_between(locs, y_bottom, y_top, **kwargs)

    axes.set_xlim([locs.min(), locs.max()])
    if x_axis == 'time':
        axes.xaxis.set_major_formatter(TimeFormatter(lag=False))
        axes.xaxis.set_label_text('Time')
    elif x_axis is None or x_axis in ['off', 'none']:
        axes.set_xticks([])
    else:
        raise ParameterError('Unknown x_axis value: {}'.format(x_axis))

    return out


def specshow(data, x_coords=None, y_coords=None,
             x_axis=None, y_axis=None,
             sr=22050, hop_length=512,
             fmin=None, fmax=None,
             bins_per_octave=12,
             **kwargs):
    '''Display a spectrogram/chromagram/cqt/etc.


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

        - None, 'none', or 'off' : no axis decoration is displayed.

        Frequency types:

        - 'linear', 'fft', 'hz' : frequency range is determined by
          the FFT window and sampling rate.
        - 'log' : the spectrum is displayed on a log scale.
        - 'mel' : frequencies are determined by the mel scale.
        - 'cqt_hz' : frequencies are determined by the CQT scale.
        - 'cqt_note' : pitches are determined by the CQT scale.

        All frequency types are plotted in units of Hz.

        Categorical types:

        - 'chroma' : pitches are determined by the chroma filters.
          Pitch classes are arranged at integer locations (0-11).

        - 'tonnetz' : axes are labeled by Tonnetz dimensions (0-5)
        - 'frames' : markers are shown as frame counts.


        Time types:

        - 'time' : markers are shown as milliseconds, seconds,
          minutes, or hours
        - 'lag' : like time, but past the half-way point counts
          as negative values.

        All time types are plotted in units of seconds.

        Other:

        - 'tempo' : markers are shown as beats-per-minute (BPM)
            using a logarithmic scale.

    x_coords : np.ndarray [shape=data.shape[1]+1]
    y_coords : np.ndarray [shape=data.shape[0]+1]

        Optional positioning coordinates of the input data.
        These can be use to explicitly set the location of each
        element `data[i, j]`, e.g., for displaying beat-synchronous
        features in natural time coordinates.

        If not provided, they are inferred from `x_axis` and `y_axis`.

    fmin : float > 0 [scalar] or None
        Frequency of the lowest spectrogram bin.  Used for Mel and CQT
        scales.

        If `y_axis` is `cqt_hz` or `cqt_note` and `fmin` is not given,
        it is set by default to `note_to_hz('C1')`.

    fmax : float > 0 [scalar] or None
        Used for setting the Mel frequency scales

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave.  Used for CQT frequency scale.

    kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.pcolormesh`.


    Returns
    -------
    axes
        The axis handle for the figure.


    See Also
    --------
    cmap : Automatic colormap detection

    matplotlib.pyplot.pcolormesh


    Examples
    --------
    Visualize an STFT power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> plt.figure(figsize=(12, 8))

    >>> D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
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

    >>> CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
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
    >>> Tgram = librosa.feature.tempogram(y=y, sr=sr)
    >>> librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
    >>> plt.colorbar()
    >>> plt.title('Tempogram')
    >>> plt.tight_layout()


    Draw beat-synchronous chroma in natural time

    >>> plt.figure()
    >>> tempo, beat_f = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    >>> beat_f = librosa.util.fix_frames(beat_f, x_max=C.shape[1])
    >>> Csync = librosa.util.sync(C, beat_f, aggregate=np.median)
    >>> beat_t = librosa.frames_to_time(beat_f, sr=sr)
    >>> ax1 = plt.subplot(2,1,1)
    >>> librosa.display.specshow(C, y_axis='chroma', x_axis='time')
    >>> plt.title('Chroma (linear time)')
    >>> ax2 = plt.subplot(2,1,2, sharex=ax1)
    >>> librosa.display.specshow(Csync, y_axis='chroma', x_axis='time',
    ...                          x_coords=beat_t)
    >>> plt.title('Chroma (beat time)')
    >>> plt.tight_layout()
    '''

    kwargs.setdefault('shading', 'flat')

    if np.issubdtype(data.dtype, np.complex):
        warnings.warn('Trying to display complex-valued input. '
                      'Showing magnitude instead.')
        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))

    all_params = dict(kwargs=kwargs,
                      sr=sr,
                      fmin=fmin,
                      fmax=fmax,
                      bins_per_octave=bins_per_octave,
                      hop_length=hop_length)

    # Get the x and y coordinates
    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    axes = plt.gca()
    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)
    plt.sci(out)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling
    __scale_axes(axes, x_axis, 'x')
    __scale_axes(axes, y_axis, 'y')

    # Construct tickers and locators
    __decorate_axis(axes.xaxis, x_axis)
    __decorate_axis(axes.yaxis, y_axis)

    return axes


def __mesh_coords(ax_type, coords, n, **kwargs):
    '''Compute axis coordinates'''

    if coords is not None:
        if len(coords) < n:
            raise ParameterError('Coordinate shape mismatch: '
                                 '{}<{}'.format(len(coords), n))
        return coords

    coord_map = {'linear': __coord_fft_hz,
                 'hz': __coord_fft_hz,
                 'log': __coord_fft_hz,
                 'mel': __coord_mel_hz,
                 'cqt': __coord_cqt_hz,
                 'cqt_hz': __coord_cqt_hz,
                 'cqt_note': __coord_cqt_hz,
                 'chroma': __coord_chroma,
                 'time': __coord_time,
                 'lag': __coord_time,
                 'tonnetz': __coord_n,
                 'off': __coord_n,
                 'tempo': __coord_tempo,
                 'frames': __coord_n,
                 None: __coord_n}

    if ax_type not in coord_map:
        raise ParameterError('Unknown axis type: {}'.format(ax_type))

    return coord_map[ax_type](n, **kwargs)


def __scale_axes(axes, ax_type, which):
    '''Set the axis scaling'''

    kwargs = dict()
    if which == 'x':
        thresh = 'linthreshx'
        base = 'basex'
        scale = 'linscalex'
        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        thresh = 'linthreshy'
        base = 'basey'
        scale = 'linscaley'
        scaler = axes.set_yscale
        limit = axes.set_ylim

    # Map ticker scales
    if ax_type == 'mel':
        mode = 'symlog'
        kwargs[thresh] = 1000.0
        kwargs[base] = 2

    elif ax_type == 'log':
        mode = 'symlog'
        kwargs[base] = 2
        kwargs[thresh] = core.note_to_hz('C2')
        kwargs[scale] = 0.5

    elif ax_type in ['cqt', 'cqt_hz', 'cqt_note']:
        mode = 'log'
        kwargs[base] = 2

    elif ax_type == 'tempo':
        mode = 'log'
        kwargs[base] = 2
        limit(16, 480)
    else:
        return

    scaler(mode, **kwargs)


def __decorate_axis(axis, ax_type):
    '''Configure axis tickers, locators, and labels'''

    if ax_type == 'tonnetz':
        axis.set_major_formatter(TONNETZ_FORMATTER)
        axis.set_major_locator(FixedLocator(0.5 + np.arange(6)))
        axis.set_label_text('Tonnetz')

    elif ax_type == 'chroma':
        axis.set_major_formatter(ChromaFormatter())
        axis.set_major_locator(FixedLocator(0.5 +
                                            np.add.outer(12 * np.arange(10),
                                                         [0, 2, 4, 5, 7, 9, 11]).ravel()))
        axis.set_label_text('Pitch class')

    elif ax_type == 'tempo':
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_label_text('BPM')

    elif ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(lag=False))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time')

    elif ax_type == 'lag':
        axis.set_major_formatter(TimeFormatter(lag=True))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Lag')

    elif ax_type == 'cqt_note':
        axis.set_major_formatter(NoteFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_minor_formatter(NoteFormatter(major=False))
        axis.set_minor_locator(LogLocator(base=2.0,
                                          subs=2.0**(np.arange(1, 12)/12.0)))
        axis.set_label_text('Note')

    elif ax_type in ['cqt_hz']:
        axis.set_major_formatter(LogHzFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(LogLocator(base=2.0,
                                          subs=2.0**(np.arange(1, 12)/12.0)))
        axis.set_label_text('Hz')

    elif ax_type in ['mel', 'log']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(SymmetricalLogLocator(axis.get_transform()))
        axis.set_label_text('Hz')

    elif ax_type in ['linear', 'hz']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_label_text('Hz')

    elif ax_type in ['frames']:
        axis.set_label_text('Frames')

    elif ax_type in ['off', 'none', None]:
        axis.set_label_text('')
        axis.set_ticks([])


def __coord_fft_hz(n, sr=22050, **_kwargs):
    '''Get the frequencies for FFT bins'''
    n_fft = 2 * (n - 1)
    return core.fft_frequencies(sr=sr, n_fft=1+n_fft)


def __coord_mel_hz(n, fmin=0, fmax=11025.0, **_kwargs):
    '''Get the frequencies for Mel bins'''

    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = 11025.0

    return core.mel_frequencies(n+2, fmin=fmin, fmax=fmax)[1:]


def __coord_cqt_hz(n, fmin=None, bins_per_octave=12, **_kwargs):
    '''Get CQT bin frequencies'''
    if fmin is None:
        fmin = core.note_to_hz('C1')
    return core.cqt_frequencies(n+1, fmin=fmin, bins_per_octave=bins_per_octave)


def __coord_chroma(n, bins_per_octave=12, **_kwargs):
    '''Get chroma bin numbers'''
    return np.linspace(0, (12.0 * n) / bins_per_octave, num=n+1, endpoint=True)


def __coord_tempo(n, sr=22050, hop_length=512, **_kwargs):
    '''Tempo coordinates'''
    return core.tempo_frequencies(n+1, sr=sr, hop_length=hop_length)


def __coord_n(n, **_kwargs):
    '''Get bare positions'''
    return np.arange(n+1)


def __coord_time(n, sr=22050, hop_length=512, **_kwargs):
    '''Get time coordinates from frames'''
    return core.frames_to_time(np.arange(n+1), sr=sr, hop_length=hop_length)
