#!/usr/bin/env python
"""Display module for interacting with matplotlib"""

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

import warnings

import librosa.core

def time_ticks(locs, *args, **kwargs): 
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
       - locations : array 
           Time-stamps for tick marks

       - n_ticks : int or None
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

       - kwargs : additional keyword arguments
           See `matplotlib.pyplot.xticks` or `yticks` for details.

    :returns:
       - See `matplotlib.pyplot.xticks` or `yticks` for details.
    '''

    n_ticks = kwargs.pop('n_ticks', 5)
    axis    = kwargs.pop('axis', 'x')
    fmt     = kwargs.pop('fmt', None)

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
        locs  = range(len(times))

    if n_ticks is not None:
        # Slice the locations and labels
        locs    = locs[::max(1, len(locs)/n_ticks)]
        times   = times[::max(1, len(times)/n_ticks)]

    # Format the labels by time
    formatters = {'ms': lambda t: '%dms' % (1e3 * t),
                  's':  lambda t: '%0.2fs' % t,
                  'm':  lambda t: '%d:%02d' % ( t / 60, np.mod(t, 60)),
                  'h':  lambda t: '%d:%02d:%02d' % (t / 3600, t / 60, np.mod(t, 60))}

    if fmt is None:
        if max(times) > 3600.0:
            fmt = 'h'
        elif max(times) > 60.0:
            fmt = 'm'
        elif max(times) > 1.0:
            fmt = 's'
        else:
            fmt = 'ms'

    elif fmt not in formatters:
        raise ValueError('Invalid format: %s' % fmt)

    times = map(formatters[fmt], times)

    return ticker(locs, times, **kwargs)

def cmap(data):
    '''Get a default colormap from the given data.

    If the data is boolean, use a black and white colormap.

    If the data has both positive and negative values, use a diverging colormap.

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

    positives = (data > 0).any()
    negatives = (data < 0).any()

    if positives and not negatives:
        return 'OrRd'
    elif negatives and not positives:
        return 'BuPu_r'
    
    return 'PuOr_r'

def specshow(data, sr=22050, hop_length=512, x_axis=None, y_axis=None, n_xticks=5, n_yticks=5, 
    fmin=None, fmax=None, **kwargs):
    '''Display a spectrogram/chromagram/cqt/etc.

    Functions as a drop-in replacement for ``matplotlib.pyplot.imshow``, but with useful defaults.

    :usage:
        >>> # Visualize an STFT with linear frequency scaling
        >>> D = np.abs(librosa.stft(y))
        >>> librosa.display.specshow(D, sr=sr, y_axis='linear')

        >>> # Or with logarithmic frequency scaling
        >>> librosa.display.specshow(D, sr=sr, y_axis='log')

        >>> # Visualize a CQT with note markers
        >>> CQT = librosa.cqt(y, sr, fmin=55, fmax=880)
        >>> librosa.display.specshow(CQT, sr=sr, y_axis='cqt_note', fmin=55, fmax=880)

        >>> # Draw time markers automatically
        >>> librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time')

        >>> # Draw a chromagram with pitch classes
        >>> C = librosa.feature.chromagram(y, sr)
        >>> librosa.display.specshow(C, y_axis='chroma')

        >>> # Force a grayscale colormap (white -> black)
        >>> librosa.display.specshow(librosa.logamplitude(D), cmap='gray_r')

    :parameters:
      - data : np.ndarray
          Matrix to display (eg, spectrogram)

      - sr : int > 0
          Sample rate. Used to determine time scale in x-axis

      - hop_length : int > 0
          Hop length. Also used to determine time scale in x-axis

      - x_axis : None or {'time', 'frames', 'off'}
          If None or 'off', no x axis is displayed.
          If 'time', markers are shown as milliseconds, seconds, minutes, or hours.
          (See ``time_ticks()`` for details.)
          If 'frames', markers are shown as frame counts.

      - y_axis : None or {'linear', 'mel', 'cqt_hz', 'cqt_note', 'chroma', 'off'}
          - None or 'off': no y axis is displayed.
          - 'linear': frequency range is determined by the FFT window and sampling rate.
          - 'log': the image is displayed on a vertical log scale.
          - 'mel': frequencies are determined by the mel scale.
          - 'cqt_hz': frequencies are determined by the fmin and fmax values.
          - 'cqt_note': pitches are determined by the fmin and fmax values.
          - 'chroma': pitches are determined by the chroma filters.

      - n_xticks : int > 0
          If x_axis is drawn, the number of ticks to show

      - n_yticks : int > 0
          If y_axis is drawn, the number of ticks to show

      - fmin : float > 0 or None

      - fmax : float > 0 or None
          Used for setting the Mel or constantq frequency scales

      - kwargs 
          Additional keyword arguments passed through to ``matplotlib.pyplot.imshow``.

    :returns:
      - image : ``matplotlib.image.AxesImage``
          As returned from ``matplotlib.pyplot.imshow``.

    :raises:
      - ValueError 
          If y_axis is 'cqt_hz' or 'cqt_note' and fmin and fmax are not supplied.
    '''

    kwargs.setdefault('aspect',          'auto')
    kwargs.setdefault('origin',          'lower')
    kwargs.setdefault('interpolation',   'nearest')

    if np.issubdtype(data.dtype, np.complex):
        warnings.warn('Trying to display complex-valued input. Showing magnitude instead.')
        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))

    # NOTE:  2013-11-14 16:15:33 by Brian McFee <brm2132@columbia.edu>pitch 
    #  We draw the image twice here. This is a hack to get around NonUniformImage
    #  not properly setting hooks for color: drawing twice enables things like
    #  colorbar() to work properly.

    axes = plt.imshow(data, **kwargs)

    if y_axis is 'log':
        axes_phantom = plt.gca()

        # Non-uniform imshow doesn't like aspect
        del kwargs['aspect']
        im_phantom   = img.NonUniformImage(axes_phantom, **kwargs)

        y_log, y_inv = __log_scale(data.shape[0])

        im_phantom.set_data( np.arange(0, data.shape[1]), y_log, data)
        axes_phantom.images.append(im_phantom)
        axes_phantom.set_ylim(0, data.shape[0])
        axes_phantom.set_xlim(0, data.shape[1])

    # Set up the y ticks
    positions = np.asarray(np.linspace(0, data.shape[0], n_yticks), dtype=int)

    if y_axis is 'linear':
        values = np.asarray(np.linspace(0, 0.5 * sr,  data.shape[0] + 1), dtype=int)

        plt.yticks(positions, values[positions])
        plt.ylabel('Hz')
    
    elif y_axis is 'log':
    
        values = np.asarray(np.linspace(0, 0.5 * sr,  data.shape[0] + 1), dtype=int)
        plt.yticks(positions, values[y_inv[positions]])
    
        plt.ylabel('Hz')

    elif y_axis is 'mel':
        m_args = {}
        if fmin is not None:
            m_args['fmin'] = fmin
        if fmax is not None:
            m_args['fmax'] = fmax

        values = librosa.core.mel_frequencies(n_mels=data.shape[0], extra=True, **m_args)[positions].astype(np.int)
        plt.yticks(positions, values)
        plt.ylabel('Hz')
    
    elif y_axis is 'cqt_hz':
        if fmax is None and fmin is None:
            raise ValueError('fmin and fmax must be supplied for CQT axis display')

        positions = np.arange(0, data.shape[0], 
                             np.ceil(data.shape[0] / float(n_yticks)), 
                             dtype=int)


        # Get frequencies
        values = librosa.core.cqt_frequencies(data.shape[0], fmin=fmin, 
                                    bins_per_octave=int(data.shape[0] / np.ceil(np.log2(fmax) - np.log2(fmin))))
        plt.yticks(positions, values[positions].astype(int))
        plt.ylabel('Hz')

    elif y_axis is 'cqt_note':
        if fmax is None and fmin is None:
            raise ValueError('fmin and fmax must be supplied for CQT axis display')

        positions = np.arange(0, data.shape[0], 
                             np.ceil(data.shape[0] / float(n_yticks)), 
                             dtype=int)

        # Get frequencies
        values = librosa.core.cqt_frequencies(data.shape[0], fmin=fmin, 
                                    bins_per_octave=int(data.shape[0] / np.ceil(np.log2(fmax) - np.log2(fmin))))
        values = librosa.core.midi_to_note(librosa.core.hz_to_midi(values[positions]))
        plt.yticks(positions, values)
        plt.ylabel('Note')

    elif y_axis is 'chroma':
        positions = np.arange(0, data.shape[0], max(1, data.shape[0] / 12))
        # Labels start at 9 here because chroma starts at A.
        values = librosa.core.midi_to_note(range(9, 9+12), octave=False)
        plt.yticks(positions, values)
        plt.ylabel('Note')
    
    elif y_axis is None or y_axis is 'off':
        plt.yticks([])
        plt.ylabel('')

    else:
        raise ValueError('Unknown y_axis parameter: %s' % y_axis)

    # Set up the x ticks
    positions = np.asarray(np.linspace(0, data.shape[1], n_xticks), dtype=int)

    if x_axis is 'time':
        time_ticks( positions, 
                    librosa.core.frames_to_time(positions, sr=sr, hop_length=hop_length),
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
        raise ValueError('Unknown x_axis parameter: %s' % x_axis)
    
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

    y_inv = np.arange(len(y)+1)
    for i in range(len(y)-1):
        y_inv[y[i]:y[i+1]] = i

    return y, y_inv

