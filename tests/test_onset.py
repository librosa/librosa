#!/usr/bin/env python
# CREATED:2013-03-11 18:14:30 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.beat

from __future__ import print_function
from nose.tools import nottest, raises, eq_

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import numpy as np
import librosa

__EXAMPLE_FILE = 'data/test1_22050.wav'


def test_onset_strength_audio():

    def __test(y, sr, feature, n_fft, hop_length, detrend, centering):

        oenv = librosa.onset.onset_strength(y=y, sr=sr,
                                            S=None,
                                            detrend=detrend,
                                            centering=centering,
                                            aggregate=aggregate,
                                            feature=feature,
                                            n_fft=n_fft,
                                            hop_length=hop_length)

        assert oenv.ndim == 1

        S = librosa.feature.melspectrogram(y=y,
                                           n_fft=n_fft,
                                           hop_length=hop_length)

        target_shape = S.shape[-1]

        if centering:
            target_shape += n_fft // (2 * hop_length)

        if not detrend:
            assert np.all(oenv >= 0)

        eq_(oenv.shape[-1], target_shape)

    y, sr = librosa.load(__EXAMPLE_FILE)

    for feature in [None,
                    librosa.feature.melspectrogram,
                    librosa.feature.chromagram]:
        for n_fft in [512, 2048]:
            for hop_length in [n_fft // 2, n_fft // 4]:
                for detrend in [False, True]:
                    for centering in [False, True]:
                        for aggregate in [None, np.mean, np.max]:
                            yield (__test, y, sr, feature, n_fft,
                                   hop_length, detrend, centering)
                            tf = raises(ValueError)(__test)
                            yield (tf, None, sr, feature, n_fft,
                                   hop_length, detrend, centering)
