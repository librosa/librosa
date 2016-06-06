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
    ChromaFormatter
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter, FixedFormatter, ScalarFormatter
from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator

from . import cache
from . import core
from . import util
from .util.exceptions import ParameterError


class TimeFormatter(Formatter):
    '''A tick formatter for time axes.

    Automatically switches between ms, s, minutes:sec, etc.
    '''

    def __init__(self, lag=False):

        self.lag = lag

    def __call__(self, x, pos=None):
        '''Return the time format as pos'''

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = '-'
        else:
            value = x
            sign = ''

        if vmax - vmin > 3.6e3:
            s = '{:d}:{:02d}:{:02d}'.format(int(value / 3.6e3),
                                            int(np.mod(value / 6e1, 6e1)),
                                            int(np.mod(value, 6e1)))
        elif vmax - vmin > 6e1:
            s = '{:d}:{:02d}'.format(int(value / 6e1),
                                     int(np.mod(value, 6e1)))
        elif vmax - vmin > 1.0:
            s = '{:0.2f}s'.format(value)
        else:
            s = '{:g}ms'.format(1e3 * value)

        return '{:s}{:s}'.format(sign, s)


class NoteFormatter(Formatter):
    '''Ticker formatter for Notes'''
    def __init__(self, octave=True):

        self.octave = octave

    def __call__(self, x, pos=None):

        if x < core.note_to_hz('C0'):
            return ''

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()
        cents = vmax < 2 * max(1, vmin)

        return core.hz_to_note(int(x), octave=self.octave, cents=cents)[0]


class ChromaFormatter(Formatter):
    '''A formatter for chroma'''
    def __call__(self, x, pos=None):
        '''Format for chroma positions'''
        return core.midi_to_note(int(x), octave=False, cents=False)


# A fixed formatter for tonnetz
TONNETZ_FORMATTER = FixedFormatter([r'5$_x$', r'5$_y$',
                                    r'm3$_x$', r'm3$_y$',
                                    r'M3$_x$', r'M3$_y$'])


@cache
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
             tmin=16, tmax=240,
             **kwargs):
    '''Display a spectrogram/chromagram/cqt/etc.

    Images are displayed in natural coordinates, e.g., seconds, Hz, or notes,
    rather than bins or frames.


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
    >>> Tgram = librosa.feature.tempogram(y=y, sr=sr)
    >>> librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
    >>> plt.colorbar()
    >>> plt.title('Tempogram')
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
                      tmin=tmin,
                      tmax=tmax,
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
    __decorate_axis(axes.xaxis, x_axis, all_params)
    __decorate_axis(axes.yaxis, y_axis, all_params)

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
        mode = 'symlog'
        kwargs[base] = 2
        kwargs[thresh] = 32.0
        kwargs[scale] = 1.0
        limit(16, 480)
    else:
        return

    scaler(mode, **kwargs)


def __decorate_axis(axis, ax_type, kwargs):
    '''Configure axis tickers, locators, and labels'''

    if ax_type == 'tonnetz':
        axis.set_major_formatter(TONNETZ_FORMATTER)
        axis.set_major_locator(FixedLocator(0.5 + np.arange(6)))
        axis.set_label_text('Tonnetz')

    elif ax_type == 'chroma':
        axis.set_major_formatter(ChromaFormatter())
        axis.set_major_locator(FixedLocator(0.5 +
                                            np.add.outer(np.arange(0, 96, 12),
                                                         [0, 2, 4, 5, 7, 9, 11]).ravel()))
        axis.set_label_text('Pitch class')

    elif ax_type == 'tempo':
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_label_text('BPM')

    elif ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(lag=False))
        axis.set_major_locator(MaxNLocator(prune=None))
        axis.set_label_text('Time')

    elif ax_type == 'lag':
        axis.set_major_formatter(TimeFormatter(lag=True))
        axis.set_major_locator(MaxNLocator(prune=None))
        axis.set_label_text('Lag')

    elif ax_type == 'cqt_note':
        axis.set_major_formatter(NoteFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_label_text('Note')

    elif ax_type in ['cqt_hz']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_label_text('Hz')

    elif ax_type in ['linear', 'hz', 'mel', 'log']:
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
