#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Display module for interacting with matplotlib"""

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

import warnings

import librosa.core
from . import cache


# This function wraps xticks or yticks: star-args is okay
def time_ticks(locs, *args, **kwargs):  # pylint: disable=star-args
    '''Plot time-formatted axis ticks.

    :usage:
        >>> # Tick at pre-computed beat times
        >>> librosa.display.specshow(S)
        >>> librosa.display.time_ticks(beat_times)

        >>> # Set the locations of the time stamps
        >>> librosa.display.time_ticks(locations, timestamps)

        >>> # Format in seconds
        >>> librosa.display.time_ticks(beat_times, fmt='s')

        >>> # Tick along the y axis
        >>> librosa.display.time_ticks(beat_times, axis='y')

    :parameters:
       - locations : list or np.ndarray
           Time-stamps for tick marks

       - n_ticks : int > 0 or None
           Show this number of ticks (evenly spaced).
           If none, all ticks are displayed.
           Default: 5

       - axis : 'x' or 'y'
           Which axis should the ticks be plotted on?
           Default: 'x'

       - fmt : None or {'ms', 's', 'm', 'h'}
           ms: milliseconds   (eg, 241ms)
           s: seconds         (eg, 1.43s)
           m: minutes         (eg, 1:02)
           h: hours           (eg, 1:02:03)

           If none, formatted is automatically selected by the
           range of the times data.

           Default: None

       - *kwargs*
          Additional keyword arguments.
          See ``matplotlib.pyplot.xticks`` or ``yticks`` for details.

    :returns:
       - See ``matplotlib.pyplot.xticks`` or ``yticks`` for details.
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

    PuOr and OrRd are chosen to optimize visibility for color-blind people.

    :usage:
        >>> librosa.display.cmap([0, 1, 2])
        'OrRd'
        >>> librosa.display.cmap(np.arange(-10, -5))
        'BuPu_r'
        >>> librosa.display.cmap(np.arange(-10, 10))
        'PuOr_r'

    :parameters:
      - data : np.ndarray
          Input data

    :returns:
      - cmap_str
          - If data is type=boolean, cmap_Str is 'gray_r'
          - If data has only positive values, cmap_str is 'OrRd'
          - If data has only negative values, cmap_str is 'BuPu_r'
          - If data has both positive and negatives, cmap_str is 'PuOr_r'
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

    Functions as a drop-in replacement for ``matplotlib.pyplot.imshow``,
    but with useful defaults.

    :usage:
        >>> # Visualize an STFT with linear frequency scaling
        >>> D = np.abs(librosa.stft(y))
        >>> librosa.display.specshow(D, sr=sr, y_axis='linear')

        >>> # Or with logarithmic frequency scaling
        >>> librosa.display.specshow(D, sr=sr, y_axis='log')

        >>> # Visualize a CQT with note markers
        >>> CQT = librosa.cqt(y, sr=sr)
        >>> librosa.display.specshow(CQT, sr=sr, y_axis='cqt_note',
                                     fmin=librosa.note_to_hz('C2'))

        >>> # Draw time markers automatically
        >>> librosa.display.specshow(D, sr=sr, hop_length=hop_length,
                                     x_axis='time')

        >>> # Draw a chromagram with pitch classes
        >>> C = librosa.feature.chromagram(y, sr)
        >>> librosa.display.specshow(C, y_axis='chroma')

        >>> # Force a grayscale colormap (white -> black)
        >>> librosa.display.specshow(librosa.logamplitude(D),
                                     cmap='gray_r')

    :parameters:
      - data : np.ndarray [shape=(d, n)]
          Matrix to display (e.g., spectrogram)

      - sr : int > 0 [scalar]
          Sample rate used to determine time scale in x-axis.

      - hop_length : int > 0 [scalar]
          Hop length, also used to determine time scale in x-axis

      - x_axis : None or {'time', 'frames', 'off'}
          If None or 'off', no x axis is displayed.

          If 'time', markers are shown as milliseconds, seconds,
          minutes, or hours.  (See :func:`time_ticks()` for details.)

          If 'frames', markers are shown as frame counts.

      - y_axis : None or str
          Range for the y-axis.  Valid types are:

          - None or 'off': no y axis is displayed.
          - 'linear': frequency range is determined by the FFT window
            and sampling rate.
          - 'log': the image is displayed on a vertical log scale.
          - 'mel': frequencies are determined by the mel scale.
          - 'cqt_hz': frequencies are determined by the CQT scale.
          - 'cqt_note': pitches are determined by the CQT scale.
          - 'chroma': pitches are determined by the chroma filters.

      - n_xticks : int > 0 [scalar]
          If x_axis is drawn, the number of ticks to show

      - n_yticks : int > 0 [scalar]
          If y_axis is drawn, the number of ticks to show

      - fmin : float > 0 [scalar] or None
          Frequency of the lowest spectrogram bin.  Used for Mel and CQT
          scales.

      - fmax : float > 0 [scalar] or None
          Used for setting the Mel frequency scales

      - bins_per_octave : int > 0 [scalar]
          Number of bins per octave.  Used for CQT frequency scale.

      - *kwargs*
          Additional keyword arguments passed through to
          ``matplotlib.pyplot.imshow``.

    :returns:
      - image : ``matplotlib.image.AxesImage``
          As returned from ``matplotlib.pyplot.imshow``.

    :raises:
      - ValueError
          If y_axis is 'cqt_hz' or 'cqt_note' and ``fmin`` is not supplied.
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
        values = librosa.core.mel_frequencies(n_mels=data.shape[0], extra=True,
                                              **m_args)[positions].astype(int)
        plt.yticks(positions, values)
        plt.ylabel('Hz')

    elif y_axis is 'cqt_hz':
        if fmin is None:
            raise ValueError('fmin must be supplied for CQT display')

        positions = np.arange(0, data.shape[0],
                              np.ceil(float(data.shape[0]) / n_yticks),
                              dtype=int)

        # Get frequencies
        values = librosa.core.cqt_frequencies(data.shape[0], fmin=fmin,
                                              bins_per_octave=bins_per_octave)
        plt.yticks(positions, values[positions].astype(int))
        plt.ylabel('Hz')

    elif y_axis is 'cqt_note':
        if fmin is None:
            raise ValueError('fmin must be supplied for CQT display')

        positions = np.arange(0, data.shape[0],
                              np.ceil(float(data.shape[0]) / n_yticks),
                              dtype=int)

        # Get frequencies
        values = librosa.core.cqt_frequencies(data.shape[0], fmin=fmin,
                                              bins_per_octave=bins_per_octave)
        values = values[positions]
        values = librosa.core.midi_to_note(librosa.core.hz_to_midi(values))

        plt.yticks(positions, values)
        plt.ylabel('Note')

    elif y_axis is 'chroma':
        positions = np.arange(0,
                              data.shape[0],
                              max(1, float(data.shape[0]) / 12))

        # Labels start at 9 here because chroma starts at A.
        values = librosa.core.midi_to_note(np.arange(9, 9+12), octave=False)
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
                   librosa.core.frames_to_time(positions, sr=sr,
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

    :parameters:
      - n : int > 0
          Number of bins

    :returns:
      - y   : np.ndarray, shape=(n,)

      - y_inv   : np.ndarray, shape=(n,)
    '''

    logn = np.log2(n)
    y = n * (1 - 2.0**np.linspace(-logn, 0, n, endpoint=True))[::-1]
    y = y.astype(int)

    y_inv = np.arange(len(y)+1)
    for i in range(len(y)-1):
        y_inv[y[i]:y[i+1]] = i

    return y, y_inv
