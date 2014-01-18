#!/usr/bin/env python
# CREATED:2014-01-18 14:09:05 by Brian McFee <brm2132@columbia.edu>
# unit tests for util routines 

import numpy as np
import librosa

def test_frame():

    # Generate a random time series
    y = np.random.randn(8000)

    for frame in [256, 1024, 2048]:
        for hop_length in [64, 256, 512]:
            y_frame = librosa.util.frame(y, frame_length=frame, hop_length=hop_length)

            for i in xrange(y_frame.shape[1]):
                assert np.allclose(y_frame[:, i], y[ i * hop_length : (i * hop_length + frame)])
