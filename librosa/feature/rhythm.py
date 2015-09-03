#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Rhythmic feature extraction'''

import numpy as np
import scipy.signal
import six

from .. import util

from ..core.audio import autocorrelate
from ..util.exceptions import ParameterError


__all__ = ['tempogram']


# -- Rhythmic features -- #
def tempogram(y=None, sr=22050, onset_envelope=None, hop_length=512,
              win_length=384, center=True, window=None, norm=np.inf):
    '''Compute the tempogram: local autocorrelation of the onset strength envelope. [1]_

    .. [1] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram—A mid-level tempo representation for musicsignals."
        ICASSP, 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        Audio time series.

    sr : number > 0 [scalar]
        sampling rate of `y`

    onset_envelope : np.ndarray [shape=(n,)] or None
        Optional pre-computed onset strength envelope as provided by
        `onset.onset_strength`

    hop_length : int > 0
        number of audio samples between successive onset measurements

    win_length : int > 0
        length of the onset autocorrelation window (in frames/onset measurements)
        The default settings (384) corresponds to `384 * hop_length / sr ~= 8.9s`.

    center : bool
        If `True`, onset autocorrelation windows are centered.
        If `False`, windows are left-aligned.

    window : None, function, np.ndarray [shape=(win_length,)]
        Window function to apply to onset strength function.
        By default (`None`), an asymmetric Hann window.

    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode.  Set to `None` to disable normalization.

    Returns
    -------
    tempogram : np.ndarray [shape=(win_length, n)]
        Localized autocorrelation of the onset strength envelope

    Raises
    ------
    ParameterError
        if neither `y` nor `onset_envelope` are provided

        if `win_length < 1`

        if `window` is an array and `len(window) != win_length`

    See Also
    --------
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.core.stft


    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, centering=False)
    >>> tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    >>> # Compute global onset autocorrelation
    >>> ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    >>> ac_global = librosa.util.normalize(ac_global)

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(3, 1, 1)
    >>> plt.plot(oenv, label='Onset strength')
    >>> plt.xticks([])
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(tempogram, x_axis='time')
    >>> plt.ylabel('Tempogram')
    >>> plt.subplot(3, 1, 3)
    >>> x = np.linspace(0, tempogram.shape[0] * 512. / sr, num=tempogram.shape[0])
    >>> plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    >>> plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    >>> plt.xlabel('Lag (seconds)')
    >>> plt.axis('tight')
    >>> plt.legend(frameon=True)
    >>> plt.tight_layout()
    '''

    from ..onset import onset_strength

    if onset_envelope is None:
        if y is None:
            raise ParameterError('Either y or onset_envelope must be provided')

        onset_envelope = onset_strength(y=y, sr=sr,
                                        hop_length=hop_length,
                                        centering=False)

    if win_length < 1:
        raise ParameterError('win_length must be a positive integer')

    if window is None:
        ac_window = scipy.signal.hann(win_length, sym=False)
    elif six.callable(window):
        ac_window = window(win_length)
    else:
        ac_window = np.asarray(window)
        if ac_window.size != win_length:
            raise ParameterError('Size mismatch between win_length and len(window)')

    # Pad the envelope so that autocorrelation windows are centered on the input
    if center:
        onset_envelope = np.pad(onset_envelope,
                                win_length // 2,
                                mode='linear_ramp', end_values=[0, 0])

    # Hann-window to avoid edge effects
    odf_frame = util.frame(onset_envelope,
                           frame_length=win_length,
                           hop_length=1)

    return util.normalize(autocorrelate(odf_frame * ac_window[:, np.newaxis], axis=0),
                          norm=norm, axis=0)
