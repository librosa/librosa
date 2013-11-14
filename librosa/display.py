#!/usr/bin/env python
"""Display module for interacting with matplotlib"""

import numpy as np
import matplotlib.pyplot as plt

import librosa.core

def specshow(X, sr=22050, hop_length=64, x_axis=None, y_axis=None, fmin=None, fmax=None, **kwargs):
    """Display a spectrogram. Wraps to `~matplotlib.pyplot.imshow` with some handy defaults.
    
    :parameters:
      - X : np.ndarray
          Matrix to display (eg, spectrogram)

      - sr : int > 0
          Sample rate. Used to determine time scale in x-axis

      - hop_length : int > 0
          Hop length. Also used to determine time scale in x-axis

      - x_axis : None or {'time', 'frames', 'off'}
          If None or 'off', no x axis is displayed.
          If 'time', markers are shown as seconds, minutes, or hours.
          If 'frames', markers are shown as frame counts.

      - y_axis : None or {'linear', 'mel', 'chroma', 'off'}
          If None or 'off', no y axis is displayed.
          If 'linear', frequency range is determined by the FFT window and sample rate.
          If 'mel', frequencies are determined by the mel scale.
          If 'chroma', pitches are determined by the chroma filters.

     - fmin, fmax : float > 0 or None
          Used for setting the Mel frequency scale

     - kwargs : dict
          Additional arguments passed through to ``matplotlib.pyplot.imshow``.

    :returns:
     - image : ``matplotlib.image.AxesImage``
          As returned from ``matplotlib.pyplot.imshow``.

    """

    kwargs['aspect']        = kwargs.get('aspect',          'auto')
    kwargs['origin']        = kwargs.get('origin',          'lower')
    kwargs['interpolation'] = kwargs.get('interpolation',   'nearest')

    kwargs['cmap']          = kwargs.get('cmap',            'OrRd')

    axes = plt.imshow(X, **kwargs)

    # Set up the x ticks
    x_pos = np.arange(0, X.shape[1]+1, max(1, X.shape[1] / 5))
    if x_axis is 'time':
        # Reformat into seconds, or minutes:seconds
        x_val = x_pos * (hop_length / np.float(sr))

        if max(x_val) > 3600.0:
            # reformat into hours:minutes:seconds
            x_val = map(lambda y: '%d:%02d:%02d' % (int(y / 3600), int(np.mod(y, 3600)), int(np.mod(y, 60))), x_val)
        elif max(x_val) > 60.0:
            # reformat into minutes:seconds
            x_val = map(lambda y: '%d:%02d' % (int(y / 60), int(np.mod(y, 60))), x_val)
        else:
            # reformat into seconds, down to the millisecond
            x_val = np.around(x_val, 3)

        plt.xticks(x_pos, x_val)
        plt.xlabel('Time')

    elif x_axis is 'frames':
        # Nothing to do here, plot is in frames
        plt.xticks(x_pos, x_pos)
        plt.xlabel('Frames')
        pass
    elif x_axis is None or x_axis is 'off':
        plt.xticks([])
        plt.xlabel('')
        pass
    else:
        raise ValueError('Unknown x_axis parameter: %s' % x_axis)

    # Set up the y ticks
    y_pos = np.arange(0, X.shape[0], max(1, X.shape[0] / 6))
    if y_axis is 'linear':
        y_val = np.fft.fftfreq( (X.shape[0] -1) * 2, 1./sr)[y_pos].astype(np.int)
        plt.yticks(y_pos, y_val)
        plt.ylabel('Hz')
        pass
    elif y_axis is 'mel':
        m_args = {}
        if fmin is not None:
            m_args['fmin'] = fmin
        if fmax is not None:
            m_args['fmax'] = fmax

        y_val = librosa.core.mel_frequencies(X.shape[0], **m_args)[y_pos].astype(np.int)
        plt.yticks(y_pos, y_val)
        plt.ylabel('Hz')
        pass
    elif y_axis is 'chroma':
        y_pos = np.arange(0, X.shape[0], max(1, X.shape[0] / 12))
        y_val = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        plt.yticks(y_pos, y_val)
        plt.ylabel('Note')
        pass
    elif y_axis is None or y_axis is 'off':
        plt.yticks([])
        plt.ylabel('')
        pass
    else:
        raise ValueError('Unknown y_axis parameter: %s' % y_axis)
    
    return axes
