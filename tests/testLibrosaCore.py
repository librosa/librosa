#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa core (__init__.py)
#
# Run me as follows:
#   cd tests/
#   nosetests -v

import librosa
import os, glob
import numpy, scipy.io

from nose.tools import nottest

def fileGenerator(pattern):
    test_files = glob.glob(pattern)
    test_files.sort()
    return test_files

def test_hz_to_mel():
    def __check_hz_to_mel(infile):
        DATA    = scipy.io.loadmat(infile)
        z       = librosa.hz_to_mel(DATA['f'], DATA['htk'])

        assert numpy.allclose(z, DATA['result'])
    
    for infile in fileGenerator('data/*hz_to_mel-*.mat'):
        yield (__check_hz_to_mel, infile)

    pass

def test_mel_to_hz():

    def __check_mel_to_hz(infile):
        DATA    = scipy.io.loadmat(infile)
        z       = librosa.mel_to_hz(DATA['f'], DATA['htk'])

        assert numpy.allclose(z, DATA['result'])
    
    for infile in fileGenerator('data/*mel_to_hz-*.mat'):
        yield (__check_mel_to_hz, infile)

    pass

def test_hz_to_octs():
    def __check_hz_to_octs(infile):
        DATA    = scipy.io.loadmat(infile)
        z       = librosa.hz_to_octs(DATA['f'])

        assert numpy.allclose(z, DATA['result'])

    for infile in fileGenerator('data/*hz_to_octs-*.mat'):
        yield (__check_hz_to_octs, infile)

    pass

def test_load():
    # Note: this does not test resampling.
    # That is a separate unit test.

    def __check_load(infile):
        DATA    = scipy.io.loadmat(infile)
        (y, sr) = librosa.load(DATA['infile'][0], target_sr=None, mono=DATA['mono'])

        # Verify that the sample rate is correct
        assert sr == DATA['sr']

        # Transpose here because matlab is row-oriented
        assert numpy.allclose(y, DATA['y'].T)

    for infile in fileGenerator('data/*load-*.mat'):
        yield (__check_load, infile)

    pass

