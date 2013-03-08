#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa core (__init__.py)

import librosa
import os, glob
import numpy, scipy.io

from nose.tools import nottest

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

def test_mel_to_hz():
    test_files = glob.glob('data/mel_to_hz*.mat')
    test_files.sort()

    def __check_mel_to_hz(infile):
        DATA    = scipy.io.loadmat(infile)
        z       = librosa.mel_to_hz(DATA['f'], DATA['htk'])

        assert numpy.allclose(z, DATA['result'])
    
    for infile in test_files:
        yield (__check_mel_to_hz, infile)
        pass
    pass

def test_hz_to_octs():
    test_files = glob.glob('data/hz_to_octs*.mat')
    test_files.sort()

    def __check_hz_to_octs(infile):
        DATA    = scipy.io.loadmat(infile)
        z       = librosa.hz_to_octs(DATA['f'])

        assert numpy.allclose(z, DATA['result'])

    for infile in test_files:
        yield (__check_hz_to_octs, infile)
        pass
    pass

def test_load():
    test_files = glob.glob('data/load*.mat')
    test_files.sort()

    # Note: this does not test resampling.
    # That is a separate unit test.

    def __check_load(infile):
        DATA    = scipy.io.loadmat(infile)
        (y, sr) = librosa.load(DATA['infile'][0], target_sr=None, mono=DATA['mono'])

        # Verify that the sample rate is correct
        assert sr == DATA['sr']

        # Transpose here because matlab is row-oriented
        assert numpy.allclose(y, DATA['y'].T)

    for infile in test_files:
        yield (__check_load, infile)
        pass
    pass

