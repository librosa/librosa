#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Display module for interacting with matplotlib

Display
=======
.. autosummary::
    :toctree: generated/

    specshow
    time_ticks
    cmap

"""

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import warnings

from . import cache
from . import core


# This function wraps xticks or yticks: star-args is okay
def time_ticks(locs, *args, **kwargs):  # pylint: disable=star-args
    '''Plot time-formatted axis ticks.

    Examples
    --------
    >>> # Tick at pre-computed beat times
    >>> librosa.display.specshow(S)
    >>> librosa.display.time_ticks(beat_times)

    >>> # Set the locations of the time stamps
    >>> librosa.display.time_ticks(locations, timestamps)

    >>> # Format in seconds
    >>> librosa.display.time_ticks(beat_times, fmt='s')

    >>> # Tick along the y axis
    >>> librosa.display.time_ticks(beat_times, axis='y')

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

    fmt : None or {'ms', 's', 'm', 'h'}
        - 'ms': milliseconds   (eg, 241ms)
        - 's': seconds         (eg, 1.43s)
        - 'm': minutes         (eg, 1:02)
        - 'h': hours           (eg, 1:02:03)

        If none, formatted is automatically selected by the
        range of the times data.

        Default: None

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
    '''

    n_ticks = kwargs.pop('n_ticks', 5)
    axis = kwargs.pop('axis', 'x')
    fmt = kwargs.pop('fmt', None)

    if axis == 'x':
        ticker = plt.xticks
    elif axis == 'y':
        ticker = plt.yticks
    else:
        raise ValueError("axis must be either 'x' or 'y'.")

    if len(args) > 0:
        times = args[0]
    else:
        times = locs
        locs = np.arange(len(times))

    if n_ticks is not None:
        # Slice the locations and labels
        locs = locs[::max(1, int(len(locs) / n_ticks))]
        times = times[::max(1, int(len(times) / n_ticks))]

    # Format the labels by time
    formats = {'ms': lambda t: '{:d}ms'.format(int(1e3 * t)),
               's': lambda t: '{:0.2f}s'.format(t),
               'm': lambda t: '{:d}:{:02d}'.format(int(t / 6e1),
                                                   int(np.mod(t, 6e1))),
               'h': lambda t: '{:d}:{:02d}:{:02d}'.format(int(t / 3.6e3),
                                                          int(np.mod(t / 6e1,
                                                                     6e1)),
                                                          int(np.mod(t, 6e1)))}

    if fmt is None:
        if max(times) > 3.6e3:
            fmt = 'h'
        elif max(times) > 6e1:
            fmt = 'm'
        elif max(times) > 1.0:
            fmt = 's'
        else:
            fmt = 'ms'

    elif fmt not in formats:
        raise ValueError('Invalid format: {:s}'.format(fmt))

    times = [formats[fmt](t) for t in times]

    return ticker(locs, times, **kwargs)


@cache
def cmap(data):
    '''Get a default colormap from the given data.

    If the data is boolean, use a black and white colormap.

    If the data has both positive and negative values,
    use a diverging colormap.

    Otherwise, use a sequential map.

    `PuOr` and `OrRd` are chosen to optimize visibility for color-blind people.

    Examples
    --------
    >>> librosa.display.cmap([0, 1, 2])
    'OrRd'
    >>> librosa.display.cmap(np.arange(-10, -5))
    'BuPu_r'
    >>> librosa.display.cmap(np.arange(-10, 10))
    'PuOr_r'

    Parameters
    ----------
    data : np.ndarray
        Input data

    Returns
    -------
    cmap_str : str
        - If `data` has dtype=boolean, `cmap_str` is 'gray_r'
        - If `data` has only positive values, `cmap_str` is 'OrRd'
        - If `data` has only negative values, `cmap_str` is 'BuPu_r'
        - If `data` has both positive and negatives, `cmap_str` is 'PuOr_r'

    See Also
    --------
    matplotlib.pyplot.colormaps
    '''

    if data.dtype == 'bool':
        return 'gray_r'

    data = np.asarray(data)

    if data.min() >= 0:
        return 'OrRd'

    if data.max() <= 0:
        return 'BuPu_r'

    return 'PuOr_r'


def specshow(data, sr=22050, hop_length=512, x_axis=None, y_axis=None,
             n_xticks=5, n_yticks=5, fmin=None, fmax=None, bins_per_octave=12,
             **kwargs):
    '''Display a spectrogram/chromagram/cqt/etc.

    Functions as a drop-in replacement for `matplotlib.pyplot.imshow`,
    but with useful defaults.

    Examples
    --------
    Visualize an STFT power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> plt.figure(figsize=(12, 8))

    >>> D = librosa.logamplitude(np.abs(librosa.stft(y))**2, ref_power=np.max)
    >>> plt.subplot(4, 2, 1)
    >>> librosa.display.specshow(D, y_axis='linear')
    >>> plt.colorbar()
    >>> plt.title('Linear-frequency power spectrogram')


    Or on a logarithmic scale

    >>> plt.subplot(4, 2, 2)
    >>> librosa.display.specshow(D, y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Log-frequency power spectrogram')


    Or use a CQT scale

    >>> CQT = librosa.logamplitude(librosa.cqt(y, sr=sr)**2, ref_power=np.max)
    >>> plt.subplot(4, 2, 3)
    >>> librosa.display.specshow(CQT, y_axis='cqt_note')
    >>> plt.colorbar()
    >>> plt.title('Constant-Q power spectrogram (note)')

    >>> plt.subplot(4, 2, 4)
    >>> librosa.display.specshow(CQT, y_axis='cqt_hz')
    >>> plt.colorbar()
    >>> plt.title('Constant-Q power spectrogram (Hz)')


    Draw a chromagram with pitch classes

    >>> C = librosa.feature.chromagram(y=y, sr=sr)
    >>> plt.subplot(4, 2, 5)
    >>> librosa.display.specshow(C, y_axis='chroma')
    >>> plt.colorbar()
    >>> plt.title('Chromagram')


    Force a grayscale colormap (white -> black)

    >>> plt.subplot(4, 2, 6)
    >>> librosa.display.specshow(D, cmap='gray_r')
    >>> plt.colorbar()
    >>> plt.title('Linear power spectrogram (grayscale)')


    Draw time markers automatically

    >>> plt.subplot(4, 2, 7)
    >>> librosa.display.specshow(D, x_axis='time', y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Log power spectrogram with time')
    >>> plt.tight_layout()


    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        Matrix to display (e.g., spectrogram)

    sr : int > 0 [scalar]
        Sample rate used to determine time scale in x-axis.

    hop_length : int > 0 [scalar]
        Hop length, also used to determine time scale in x-axis

    x_axis : None or {'time', 'frames', 'off'}
        - If `None` or `'off'`, no x axis is displayed.
        - If `'time'`, markers are shown as milliseconds, seconds,
          minutes, or hours
        - If `'frames'`, markers are shown as frame counts.

    y_axis : None or str
        Range for the y-axis.  Valid types are:

        - None or 'off': no y axis is displayed.
        - 'linear': frequency range is determined by the FFT window
            and sampling rate.
        - 'log': the image is displayed on a vertical log scale.
        - 'mel': frequencies are determined by the mel scale.
        - 'cqt_hz': frequencies are determined by the CQT scale.
        - 'cqt_note': pitches are determined by the CQT scale.
        - 'chroma': pitches are determined by the chroma filters.

    n_xticks : int > 0 [scalar]
        If x_axis is drawn, the number of ticks to show

    n_yticks : int > 0 [scalar]
        If y_axis is drawn, the number of ticks to show

    fmin : float > 0 [scalar] or None
        Frequency of the lowest spectrogram bin.  Used for Mel and CQT
        scales.

        If `y_axis` is `cqt_hz` or `cqt_note` and `fmin` is not given,
        it is set by default to `note_to_hz('C2')`.

    fmax : float > 0 [scalar] or None
        Used for setting the Mel frequency scales

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave.  Used for CQT frequency scale.

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
    matplotlib.pyplot.imshow
    '''

    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('origin', 'lower')
    kwargs.setdefault('interpolation', 'nearest')

    if np.issubdtype(data.dtype, np.complex):
        warnings.warn('Trying to display complex-valued input. ' +
                      'Showing magnitude instead.')
        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))

    # NOTE:  2013-11-14 16:15:33 by Brian McFee <brm2132@columbia.edu>
    #  We draw the image twice here. This is a hack to get around
    #  NonUniformImage not properly setting hooks for color.
    #  Drawing twice enables things like colorbar() to work properly.

    axes = plt.imshow(data, **kwargs)

    if y_axis is 'log':
        axes_phantom = plt.gca()

        # Non-uniform imshow doesn't like aspect
        del kwargs['aspect']
        im_phantom = img.NonUniformImage(axes_phantom, **kwargs)

        y_log, y_inv = __log_scale(data.shape[0])

        im_phantom.set_data(np.arange(0, data.shape[1]), y_log, data)
        axes_phantom.images.append(im_phantom)
        axes_phantom.set_ylim(0, data.shape[0])
        axes_phantom.set_xlim(0, data.shape[1])

    # Set up the y ticks
    positions = np.asarray(np.linspace(0, data.shape[0], n_yticks), dtype=int)

    if y_axis is 'linear':
        values = np.asarray(np.linspace(0, 0.5 * sr, data.shape[0] + 1),
                            dtype=int)

        plt.yticks(positions, values[positions])
        plt.ylabel('Hz')

    elif y_axis is 'log':
        values = np.asarray(np.linspace(0, 0.5 * sr, data.shape[0] + 1),
                            dtype=int)
        plt.yticks(positions, values[y_inv[positions]])

        plt.ylabel('Hz')

    elif y_axis is 'mel':
        m_args = {}
        if fmin is not None:
            m_args['fmin'] = fmin
        if fmax is not None:
            m_args['fmax'] = fmax

        # only two star-args here, defined immediately above
        # pylint: disable=star-args
        values = core.mel_frequencies(n_mels=data.shape[0], extra=True,
                                      **m_args)[positions].astype(int)
        plt.yticks(positions, values)
        plt.ylabel('Hz')

    elif y_axis is 'cqt_hz':
        if fmin is None:
            fmin = core.note_to_hz('C2')

        positions = np.arange(0, data.shape[0],
                              np.ceil(float(data.shape[0]) / n_yticks),
                              dtype=int)

        # Get frequencies
        values = core.cqt_frequencies(data.shape[0], fmin=fmin,
                                      bins_per_octave=bins_per_octave)
        plt.yticks(positions, values[positions].astype(int))
        plt.ylabel('Hz')

    elif y_axis is 'cqt_note':
        if fmin is None:
            fmin = core.note_to_hz('C2')

        positions = np.arange(0, data.shape[0],
                              np.ceil(float(data.shape[0]) / n_yticks),
                              dtype=int)

        # Get frequencies
        values = core.cqt_frequencies(data.shape[0], fmin=fmin,
                                      bins_per_octave=bins_per_octave)
        values = values[positions]
        values = core.hz_to_note(values)

        plt.yticks(positions, values)
        plt.ylabel('Note')

    elif y_axis is 'chroma':
        positions = np.arange(0,
                              data.shape[0],
                              max(1, float(data.shape[0]) / 12))

        # Labels start at 9 here because chroma starts at A.
        values = core.midi_to_note(np.arange(9, 9+12), octave=False)
        plt.yticks(positions, values)
        plt.ylabel('Pitch class')

    elif y_axis is None or y_axis is 'off':
        plt.yticks([])
        plt.ylabel('')

    else:
        raise ValueError('Unknown y_axis parameter: {:s}'.format(y_axis))

    # Set up the x ticks
    positions = np.asarray(np.linspace(0, data.shape[1], n_xticks), dtype=int)

    if x_axis is 'time':
        time_ticks(positions,
                   core.frames_to_time(positions, sr=sr,
                                       hop_length=hop_length),
                   n_ticks=None, axis='x')

        plt.xlabel('Time')

    elif x_axis is 'frames':
        # Nothing to do here, plot is in frames
        plt.xticks(positions, positions)
        plt.xlabel('Frames')

    elif x_axis is None or x_axis is 'off':
        plt.xticks([])
        plt.xlabel('')

    else:
        raise ValueError('Unknown x_axis parameter: {:s}'.format(x_axis))

    return axes


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

    y_inv = np.arange(len(y)+1)
    for i in range(len(y)-1):
        y_inv[y[i]:y[i+1]] = i

    return y, y_inv
