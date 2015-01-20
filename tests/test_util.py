#!/usr/bin/env python
# CREATED:2014-01-18 14:09:05 by Brian McFee <brm2132@columbia.edu>
# unit tests for util routines 

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import numpy as np
import librosa


def test_example_audio_file():

    assert os.path.exists(librosa.util.example_audio_file())


def test_frame():

    # Generate a random time series
    def __test(P):
        frame, hop = P

        y = np.random.randn(8000)
        y_frame = librosa.util.frame(y, frame_length=frame, hop_length=hop)

        for i in range(y_frame.shape[1]):
            assert np.allclose(y_frame[:, i], y[ i * hop : (i * hop + frame)])

    for frame in [256, 1024, 2048]:
        for hop_length in [64, 256, 512]:
            yield (__test, [frame, hop_length])
