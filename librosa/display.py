#!/usr/bin/env python
"""Display module for interacting with matplotlib"""

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

import librosa.core

def specshow(data, sr=22050, hop_length=512, x_axis=None, y_axis=None, n_xticks=5, n_yticks=5, 
    fmin=None, fmax=None, **kwargs):
    """Display a spectrogram. Wraps to `~matplotlib.pyplot.imshow` with some handy defaults.
    
    :parameters:
      - data : np.ndarray
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
          If 'log', the image is displayed on a vertical log scale.
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

    # Determine the colormap automatically
    if (data < 0).any() and (data > 0).any():
        kwargs['cmap']          = kwargs.get('cmap',            'RdGy_r')
    else:
        kwargs['cmap']          = kwargs.get('cmap',            'OrRd')

    # NOTE:  2013-11-14 16:15:33 by Brian McFee <brm2132@columbia.edu>
    #  We draw the image twice here. This is a hack to get around NonUniformImage
    #  not properly setting hooks for color: drawing twice enables things like
    #  colorbar() to work properly.

    axes = plt.imshow(data, **kwargs)

    if y_axis is 'log':
        axes_phantom = plt.gca()

        # Non-uniform imshow doesn't like aspect
        del kwargs['aspect']
        im_phantom   = img.NonUniformImage(axes_phantom, **kwargs)

        y_log = (data.shape[0] - np.logspace( 0, np.log2( data.shape[0] ), data.shape[0], base=2.0))[::-1]
        y_inv = np.arange(len(y_log)+1)
        for i in range(len(y_log)-1):
            y_inv[y_log[i]:y_log[i+1]] = i

        im_phantom.set_data( np.arange(0, data.shape[1]), y_log, data)
        axes_phantom.images.append(im_phantom)
        axes_phantom.set_ylim(0, data.shape[0])
        axes_phantom.set_xlim(0, data.shape[1])

    # Set up the y ticks
    y_pos = np.asarray(np.linspace(0, data.shape[0], n_yticks), dtype=int)

    if y_axis is 'linear':
        y_val = np.asarray(np.linspace(0, 0.5 * sr,  data.shape[0] + 1), dtype=int)

        plt.yticks(y_pos, y_val[y_pos])
        plt.ylabel('Hz')
    
    elif y_axis is 'log':
    
        y_val = np.asarray(np.linspace(0, 0.5 * sr,  data.shape[0] + 1), dtype=int)
        plt.yticks(y_pos, y_val[y_inv[y_pos]])
    
        plt.ylabel('Hz')
    
    elif y_axis is 'mel':
        m_args = {}
        if fmin is not None:
            m_args['fmin'] = fmin
        if fmax is not None:
            m_args['fmax'] = fmax

        y_val = librosa.core.mel_frequencies(data.shape[0], **m_args)[y_pos].astype(np.int)
        plt.yticks(y_pos, y_val)
        plt.ylabel('Hz')
    
    elif y_axis is 'chroma':
        y_pos = np.arange(0, data.shape[0], max(1, data.shape[0] / 12))
        # Labels start at 9 here because chroma starts at A.
        y_val = librosa.core.midi_to_note(range(9, 9+12), octave=False)
        plt.yticks(y_pos, y_val)
        plt.ylabel('Note')
    
    elif y_axis is None or y_axis is 'off':
        plt.yticks([])
        plt.ylabel('')

    else:
        raise ValueError('Unknown y_axis parameter: %s' % y_axis)

    # Set up the x ticks
    x_pos = np.asarray(np.linspace(0, data.shape[1], n_xticks), dtype=int)

    if x_axis is 'time':
        # Reformat into seconds, or minutes:seconds
        x_val = librosa.core.frames_to_time(x_pos, sr=sr, hop_length=hop_length)

        if max(x_val) > 3600.0:
            # reformat into hours:minutes:seconds
            x_val = map(lambda y: '%d:%02d:%02d' % (int(y / 3600), 
                                                    int(np.mod(y, 3600)), 
                                                    int(np.mod(y, 60))), 
                                                    x_val)
        elif max(x_val) > 60.0:
            # reformat into minutes:seconds
            x_val = map(lambda y: '%d:%02d' % ( int(y / 60), 
                                                int(np.mod(y, 60))), 
                                                x_val)
        else:
            # reformat into seconds, down to the millisecond
            x_val = np.around(x_val, 3)

        plt.xticks(x_pos, x_val)
        plt.xlabel('Time')

    elif x_axis is 'frames':
        # Nothing to do here, plot is in frames
        plt.xticks(x_pos, x_pos)
        plt.xlabel('Frames')
    elif x_axis is None or x_axis is 'off':
        plt.xticks([])
        plt.xlabel('')
    else:
        raise ValueError('Unknown x_axis parameter: %s' % x_axis)
    
    return axes

