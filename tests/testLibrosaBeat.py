#!/usr/bin/env python
# CREATED:2013-03-11 18:14:30 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.beat


import librosa, numpy, scipy.io

from testLibrosaCore import files, load

def test_onset_strength():

    def __test(infile):
        DATA    = load(infile)

        # Compute onset envelope using the same spectrogram
        onsets  = librosa.beat.onset_strength(y=None, sr=8000, S=DATA['D'])

        assert numpy.allclose(onsets, DATA['onsetenv'][0])

        pass

    for infile in files('data/beat-onset-*.mat'):
        yield (__test, infile)
    pass

def test_tempo():
    def __test(infile):
        DATA    = load(infile)

        # Estimate tempo from the given onset envelope
        tempo   = librosa.beat.onset_estimate_bpm(  DATA['onsetenv'][0], 
                                                    120.0,  # starting bpm
                                                    8000 / 32)

        assert  (numpy.allclose(tempo, DATA['t'][0,0]) or 
                 numpy.allclose(tempo, DATA['t'][0,1]))
        pass

    for infile in files('data/beat-tempo-*.mat'):
        yield (__test, infile)
    pass

def test_beat():
    def __test(infile):
        DATA    = load(infile)
        
        (bpm, beats) = librosa.beat.beat_track(y=None, sr=8000, hop_length=32, onsets=DATA['onsetenv'][0])

        assert numpy.allclose(librosa.frames_to_time(beats, sr=8000, hop_length=32), DATA['beats'])
        pass
    for infile in files('data/beat-beat-*.mat'):
        yield (__test, infile)
    pass
