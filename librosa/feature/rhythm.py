#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Rhythmic feature extraction'''

import numpy as np
import scipy.signal
import six

from .. import util

from ..core.audio import autocorrelate
from ..util.exceptions import ParameterError
from ..filters import get_window


__all__ = ['tempogram']


# -- Rhythmic features -- #
def tempogram(y=None, sr=22050, onset_envelope=None, hop_length=512,
              win_length=384, center=True, window='hann', norm=np.inf):
    '''Compute the tempogram: local autocorrelation of the onset strength envelope. [1]_

    .. [1] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
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

    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `core.stft`.

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

    See Also
    --------
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.core.stft


    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> hop_length = 512
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    >>> tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
    ...                                       hop_length=hop_length)
    >>> # Compute global onset autocorrelation
    >>> ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    >>> ac_global = librosa.util.normalize(ac_global)
    >>> # Estimate the global tempo for display purposes
    >>> tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
    ...                            hop_length=hop_length)[0]

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 8))
    >>> plt.subplot(4, 1, 1)
    >>> plt.plot(oenv, label='Onset strength')
    >>> plt.xticks([])
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')
    >>> plt.subplot(4, 1, 2)
    >>> # We'll truncate the display to a narrower range of tempi
    >>> librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='tempo')
    >>> plt.axhline(tempo, color='w', linestyle='--', alpha=1,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> plt.legend(frameon=True, framealpha=0.75)
    >>> plt.subplot(4, 1, 3)
    >>> x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
    ...                 num=tempogram.shape[0])
    >>> plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    >>> plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    >>> plt.xlabel('Lag (seconds)')
    >>> plt.axis('tight')
    >>> plt.legend(frameon=True)
    >>> plt.subplot(4,1,4)
    >>> # We can also plot on a BPM axis
    >>> freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    >>> plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
    ...              label='Mean local autocorrelation', basex=2)
    >>> plt.semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
    ...              label='Global autocorrelation', basex=2)
    >>> plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> plt.legend(frameon=True)
    >>> plt.xlabel('BPM')
    >>> plt.axis('tight')
    >>> plt.grid()
    >>> plt.tight_layout()
    '''

    from ..onset import onset_strength

    if win_length < 1:
        raise ParameterError('win_length must be a positive integer')

    ac_window = get_window(window, win_length, fftbins=True)

    if onset_envelope is None:
        if y is None:
            raise ParameterError('Either y or onset_envelope must be provided')

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Center the autocorrelation windows
    n = len(onset_envelope)

    if center:
        onset_envelope = np.pad(onset_envelope, int(win_length // 2),
                                mode='linear_ramp', end_values=[0, 0])

    # Carve onset envelope into frames
    odf_frame = util.frame(onset_envelope,
                           frame_length=win_length,
                           hop_length=1)

    # Truncate to the length of the original signal
    if center:
        odf_frame = odf_frame[:, :n]

    # Window, autocorrelate, and normalize
    return util.normalize(autocorrelate(odf_frame * ac_window[:, np.newaxis],
                                        axis=0),
                          norm=norm, axis=0)
