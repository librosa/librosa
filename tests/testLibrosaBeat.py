#!/usr/bin/env python
# CREATED:2013-03-11 18:14:30 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.beat

from __future__ import print_function
from nose.tools import nottest

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import numpy as np
import librosa

from testLibrosaCore import files, load


def test_onset_strength():

    def __test(infile):
        DATA = load(infile)

        # Compute onset envelope using the same spectrogram
        onsets = librosa.onset.onset_strength(y=None,
                                              sr=8000,
                                              S=DATA['D'],
                                              centering=False,
                                              detrend=True)

        assert np.allclose(onsets[1:], DATA['onsetenv'][0])

        pass

    for infile in files('data/beat-onset-*.mat'):
        yield (__test, infile)


def test_tempo():

    def __test(infile):
        DATA = load(infile)

        # Estimate tempo from the given onset envelope
        tempo = librosa.beat.estimate_tempo(DATA['onsetenv'][0],
                                            sr=8000,
                                            hop_length=32,
                                            start_bpm=120.0)

        assert (np.allclose(tempo, DATA['t'][0, 0]) or
                np.allclose(tempo, DATA['t'][0, 1]))

    for infile in files('data/beat-tempo-*.mat'):
        yield (__test, infile)


# Beat tracking test is no longer enabled due to librosa's various corrections
@nottest
def test_beat():

    def __test(infile):

        DATA = load(infile)

        (bpm, beats) = librosa.beat.beat_track(y=None,
                                               sr=8000,
                                               hop_length=32,
                                               onset_envelope=DATA['onsetenv'][0])

        beat_times = librosa.frames_to_time(beats, sr=8000, hop_length=32)
        assert np.allclose(beat_times, DATA['beats'])

    for infile in files('data/beat-beat-*.mat'):
        yield (__test, infile)
