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
import warnings
import librosa

from test_core import files, load

__EXAMPLE_FILE = 'data/test1_22050.wav'


warnings.resetwarnings()
warnings.simplefilter('always')


def test_onset_strength():

    def __test(infile):
        DATA = load(infile)

        # Compute onset envelope using the same spectrogram
        onsets = librosa.onset.onset_strength(y=None,
                                              sr=8000,
                                              S=DATA['D'],
                                              lag=1,
                                              max_size=1,
                                              center=False,
                                              detrend=True,
                                              aggregate=np.mean)

        assert np.allclose(onsets[1:], DATA['onsetenv'][0])

        pass

    for infile in files('data/beat-onset-*.mat'):
        yield (__test, infile)


def test_estimate_tempo():

    def __test(infile):
        DATA = load(infile)

        # Estimate tempo from the given onset envelope
        tempo = librosa.beat.estimate_tempo(DATA['onsetenv'][0],
                                            sr=8000,
                                            hop_length=32,
                                            start_bpm=120.0)

        assert (np.allclose(tempo, DATA['t'][0, 0]) or
                np.allclose(tempo, DATA['t'][0, 1])), (tempo, DATA['t'])

    for infile in files('data/beat-tempo-*.mat'):
        yield (__test, infile)


def test_tempo():

    def __test(tempo, sr, hop_length, ac_size, aggregate, y):

        tempo_est = librosa.beat.tempo(y=y, sr=sr, hop_length=hop_length,
                                       ac_size=ac_size,
                                       aggregate=aggregate)

        # Being within 5% for the stable frames is close enough
        if aggregate is None:
            win_size = int(ac_size * sr // hop_length)
            assert np.all(np.abs(tempo_est[win_size:-win_size] - tempo) <= 0.05 * tempo)
        else:
            assert np.abs(tempo_est - tempo) <= 0.05 * tempo, (tempo, tempo_est)

    for sr in [22050, 44100]:
        for tempo in [40, 60, 80, 110, 160]:
            # Make a pulse train at the target tempo
            y = np.zeros(20 * sr)
            delay = np.asscalar(librosa.time_to_samples(60./tempo, sr=sr))
            y[::delay] = 1
            for hop_length in [512, 1024]:
                for ac_size in [4, 8]:
                    for aggregate in [None, np.mean]:
                        yield __test, tempo, sr, hop_length, ac_size, aggregate, y


@raises(librosa.ParameterError)
def test_beat_no_input():

    librosa.beat.beat_track(y=None, onset_envelope=None)


def test_beat_no_onsets():

    sr = 22050
    hop_length = 512
    duration = 30

    onsets = np.zeros(duration * sr // hop_length)

    tempo, beats = librosa.beat.beat_track(onset_envelope=onsets,
                                           sr=sr,
                                           hop_length=hop_length)

    assert np.allclose(tempo, 0)
    eq_(len(beats), 0)


def test_estimate_tempo_no_onsets():

    sr = 22050
    hop_length = 512
    duration = 30
    onsets = np.zeros(duration * sr // hop_length)

    def __test(start_bpm):
        tempo = librosa.beat.estimate_tempo(onsets, sr=sr,
                                            hop_length=hop_length,
                                            start_bpm=start_bpm)
        eq_(tempo, start_bpm)

    for start_bpm in [40, 60, 120, 240]:
        yield __test, start_bpm


def test_tempo_no_onsets():

    sr = 22050
    hop_length = 512
    duration = 30
    onsets = np.zeros(duration * sr // hop_length)

    def __test(start_bpm, aggregate):
        tempo = librosa.beat.tempo(onset_envelope=onsets, sr=sr,
                                   hop_length=hop_length,
                                   start_bpm=start_bpm,
                                   aggregate=aggregate)
        assert np.allclose(tempo, start_bpm)

    for start_bpm in [40, 60, 120, 240]:
        for aggregate in [None, np.mean]:
            yield __test, start_bpm, aggregate


def test_beat():

    y, sr = librosa.load(__EXAMPLE_FILE)

    hop_length = 512

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    def __test(with_audio, with_tempo, start_bpm, bpm, trim, tightness):
        if with_audio:
            _y = y
            _ons = None
        else:
            _y = None
            _ons = onset_env

        tempo, beats = librosa.beat.beat_track(y=_y,
                                               sr=sr,
                                               onset_envelope=_ons,
                                               hop_length=hop_length,
                                               start_bpm=start_bpm,
                                               tightness=tightness,
                                               trim=trim,
                                               bpm=bpm)

        assert tempo >= 0

        if len(beats) > 0:
            assert beats.min() >= 0
            assert beats.max() <= len(onset_env)

    for with_audio in [False, True]:
        for with_tempo in [False, True]:
            for trim in [False, True]:
                for start_bpm in [-20, 0, 60, 120, 240]:
                    for bpm in [-20, 0, None, 150, 360]:
                        for tightness in [0, 100, 10000]:

                            if (tightness <= 0 or
                                (bpm is not None and bpm <= 0) or
                                (start_bpm is not None and
                                 bpm is None and
                                 start_bpm <= 0)):

                                tf = raises(librosa.ParameterError)(__test)
                            else:
                                tf = __test
                            yield (tf, with_audio, with_tempo,
                                   start_bpm, bpm, trim, tightness)


def test_beat_units():

    def __test(units, hop_length, y, sr):

        tempo, b1 = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        _, b2 = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length,
                                        units=units)

        t1 = librosa.frames_to_time(b1, sr=sr, hop_length=hop_length)

        if units == 'time':
            t2 = b2

        elif units == 'samples':
            t2 = librosa.samples_to_time(b2, sr=sr)

        elif units == 'frames':
            t2 = librosa.frames_to_time(b2, sr=sr, hop_length=hop_length)

        assert np.allclose(t1, t2)

    for sr in [None, 44100]:
        y, sr = librosa.load(__EXAMPLE_FILE, sr=sr)
        for hop_length in [512, 1024]:
            for units in ['frames', 'time', 'samples']:
                yield __test, units, hop_length, y, sr
            yield raises(librosa.ParameterError)(__test), 'bad units', hop_length, y, sr


# Beat tracking regression test is no longer enabled due to librosa's
# corrections
@nottest
def deprecated_test_beat():

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
