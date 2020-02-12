#!/usr/bin/env python
# CREATED:2013-03-11 18:14:30 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.beat

from __future__ import print_function

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import pytest

import numpy as np
import scipy.stats
import librosa

from test_core import files, load

__EXAMPLE_FILE = os.path.join('tests', 'data', 'test1_22050.wav')


@pytest.mark.parametrize('infile', files(os.path.join('data', 'beat-onset-*.mat')))
def test_onset_strength(infile):

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


@pytest.mark.parametrize('tempo', [60, 80, 110, 160])
@pytest.mark.parametrize('sr', [22050, 44100])
@pytest.mark.parametrize('hop_length', [512, 1024])
@pytest.mark.parametrize('ac_size', [4, 8])
@pytest.mark.parametrize('aggregate', [None, np.mean])
@pytest.mark.parametrize('prior', [None, scipy.stats.uniform(60, 240)])
def test_tempo(tempo, sr, hop_length, ac_size, aggregate, prior):

    y = np.zeros(20 * sr)
    delay = librosa.time_to_samples(60./tempo, sr=sr).item()
    y[::delay] = 1

    tempo_est = librosa.beat.tempo(y=y, sr=sr, hop_length=hop_length,
                                   ac_size=ac_size,
                                   aggregate=aggregate,
                                   prior=prior)

    # Being within 5% for the stable frames is close enough
    if aggregate is None:
        win_size = int(ac_size * sr // hop_length)
        assert np.all(np.abs(tempo_est[win_size:-win_size] - tempo) <= 0.05 * tempo)
    else:
        assert np.abs(tempo_est - tempo) <= 0.05 * tempo, (tempo, tempo_est)


@pytest.mark.xfail(raises=librosa.ParameterError)
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
    assert len(beats) == 0


@pytest.mark.parametrize('start_bpm', [40, 60, 117, 235])
@pytest.mark.parametrize('aggregate', [None, np.mean])
@pytest.mark.parametrize('onsets', [np.zeros(30 * 22050 // 512)])
@pytest.mark.parametrize('sr', [22050])
@pytest.mark.parametrize('hop_length', [512])
def test_tempo_no_onsets(start_bpm, aggregate, onsets, sr, hop_length):

    tempo = librosa.beat.tempo(onset_envelope=onsets, sr=sr,
                               hop_length=hop_length,
                               start_bpm=start_bpm,
                               aggregate=aggregate)
    # Depending on bin resolution, we might not be able to match exactly
    assert np.allclose(tempo, start_bpm, atol=1e0)


def test_beat():

    y, sr = librosa.load(__EXAMPLE_FILE)

    hop_length = 512

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    def __test(with_audio, with_tempo, start_bpm, bpm, trim, tightness, prior):
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
                                               bpm=bpm,
                                               prior=prior)

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
                            for prior in [None, scipy.stats.uniform(60, 240)]:
                                if (tightness <= 0 or
                                       (bpm is not None and bpm <= 0) or
                                       (start_bpm is not None and
                                       bpm is None and start_bpm <= 0)):

                                    tf = pytest.mark.xfail(__test, raises=librosa.ParameterError)
                                else:
                                    tf = __test
                                yield (tf, with_audio, with_tempo,
                                        start_bpm, bpm, trim, tightness, prior)


@pytest.mark.parametrize('sr', [None, 44100])
@pytest.mark.parametrize('hop_length', [512, 1024])
@pytest.mark.parametrize('units', ['frames', 'time', 'samples',
                                   pytest.mark.xfail('bad units', raises=librosa.ParameterError)])
def test_beat_units(sr, hop_length, units):

    y, sr = librosa.load(__EXAMPLE_FILE, sr=sr)

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


@pytest.mark.parametrize('sr', [22050])
@pytest.mark.parametrize('hop_length', [256, 512])
@pytest.mark.parametrize('win_length', [192, 384])
@pytest.mark.parametrize('use_onset', [False, True])
@pytest.mark.parametrize('tempo_min, tempo_max', [(30, 300), 
                                                  (None, 240),
                                                  (60, None),
                                                  pytest.mark.xfail((120, 80),
                                                      raises=librosa.ParameterError)])
@pytest.mark.parametrize('prior', [None, scipy.stats.lognorm(s=1, loc=np.log(120), scale=120)])
def test_plp(sr, hop_length, win_length, tempo_min, tempo_max, use_onset, prior):

    y, sr = librosa.load(__EXAMPLE_FILE, sr=sr)

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    if use_onset:
        pulse = librosa.beat.plp(y=y, sr=sr, onset_envelope=oenv,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 tempo_min=tempo_min,
                                 tempo_max=tempo_max,
                                 prior=prior)
    else:
        pulse = librosa.beat.plp(y=y, sr=sr,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 tempo_min=tempo_min,
                                 tempo_max=tempo_max,
                                 prior=prior)

    assert len(pulse) == len(oenv)

    assert np.all(pulse >= 0)
    assert np.all(pulse <= 1)


# Beat tracking regression test is no longer enabled due to librosa's
# corrections
@pytest.mark.skip
@pytest.mark.parametrize('infile', files(os.path.join('data', 'beat-beat-*.mat')))
def deprecated_test_beat(infile):

    DATA = load(infile)

    (bpm, beats) = librosa.beat.beat_track(y=None,
                                           sr=8000,
                                           hop_length=32,
                                           onset_envelope=DATA['onsetenv'][0])

    beat_times = librosa.frames_to_time(beats, sr=8000, hop_length=32)
    assert np.allclose(beat_times, DATA['beats'])
