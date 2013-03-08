#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa core (__init__.py)

import librosa
import os, glob
import numpy, scipy.io

def test_hz_to_mel():
    test_files = glob.glob('data/hz_to_mel*.mat')
    test_files.sort()

    def __check_hz_to_mel(infile):
        DATA    = scipy.io.loadmat(infile)
        z       = librosa.hz_to_mel(DATA['f'], DATA['htk'])

        assert numpy.allclose(z, DATA['result'])
    
    for infile in test_files:
        yield (__check_hz_to_mel, infile)
        pass
    pass
