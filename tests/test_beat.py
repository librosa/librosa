#!/usr/bin/env python
# CREATED:2013-03-11 18:14:30 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.beat

from __future__ import print_function

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except:
    pass

import pytest
from contextlib2 import nullcontext as dnr

import numpy as np
import scipy.stats
import librosa

from test_core import files, load

__EXAMPLE_FILE = os.path.join("tests", "data", "test1_22050.wav")


@pytest.fixture(scope="module", params=[22050, 44100])
def ysr(request):
    return librosa.load(__EXAMPLE_FILE, sr=request.param)


@pytest.mark.parametrize("infile", files(os.path.join("data", "beat-onset-*.mat")))
def test_onset_strength(infile):

    DATA = load(infile)

    # Compute onset envelope using the same spectrogram
    onsets = librosa.onset.onset_strength(
        y=None, sr=8000, S=DATA["D"], lag=1, max_size=1, center=False, detrend=True, aggregate=np.mean
    )

    assert np.allclose(onsets[1:], DATA["onsetenv"][0])


@pytest.mark.parametrize("tempo", [60, 80, 110, 160])
@pytest.mark.parametrize("sr", [22050, 44100])
@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize("ac_size", [4, 8])
@pytest.mark.parametrize("aggregate", [None, np.mean])
@pytest.mark.parametrize("prior", [None, scipy.stats.uniform(60, 240)])
def test_tempo(tempo, sr, hop_length, ac_size, aggregate, prior):

    y = np.zeros(20 * sr)
    delay = librosa.time_to_samples(60.0 / tempo, sr=sr).item()
    y[::delay] = 1

    tempo_est = librosa.beat.tempo(y=y, sr=sr, hop_length=hop_length, ac_size=ac_size, aggregate=aggregate, prior=prior)

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

    tempo, beats = librosa.beat.beat_track(onset_envelope=onsets, sr=sr, hop_length=hop_length)

    assert np.allclose(tempo, 0)
    assert len(beats) == 0


@pytest.mark.parametrize("start_bpm", [40, 60, 117, 235])
@pytest.mark.parametrize("aggregate", [None, np.mean])
@pytest.mark.parametrize("onsets", [np.zeros(30 * 22050 // 512)])
@pytest.mark.parametrize("sr", [22050])
@pytest.mark.parametrize("hop_length", [512])
def test_tempo_no_onsets(start_bpm, aggregate, onsets, sr, hop_length):

    tempo = librosa.beat.tempo(
        onset_envelope=onsets, sr=sr, hop_length=hop_length, start_bpm=start_bpm, aggregate=aggregate
    )
    # Depending on bin resolution, we might not be able to match exactly
    assert np.allclose(tempo, start_bpm, atol=1e0)


@pytest.fixture(scope="module")
def hop():
    return 512


@pytest.fixture(scope="module")
def oenv(ysr, hop):
    y, sr = ysr
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize("with_tempo", [False, True])
@pytest.mark.parametrize("trim", [False, True])
@pytest.mark.parametrize("start_bpm", [60, 120, 240])
@pytest.mark.parametrize("bpm", [None, 150, 360])
@pytest.mark.parametrize("tightness", [1e2, 1e4])
@pytest.mark.parametrize("prior", [None, scipy.stats.uniform(60, 240)])
def test_beat(ysr, hop, oenv, with_audio, with_tempo, start_bpm, bpm, trim, tightness, prior):

    y, sr = ysr

    if with_audio:
        _y = y
        _ons = None
    else:
        _y = None
        _ons = oenv

    tempo, beats = librosa.beat.beat_track(
        y=_y,
        sr=sr,
        onset_envelope=_ons,
        hop_length=hop,
        start_bpm=start_bpm,
        tightness=tightness,
        trim=trim,
        bpm=bpm,
        prior=prior,
    )

    assert tempo >= 0

    if len(beats) > 0:
        assert beats.min() >= 0
        assert beats.max() <= len(oenv)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("tightness", [-1, -0.5, 0])
def test_beat_bad_tightness(ysr, tightness):
    y, sr = ysr
    librosa.beat.beat_track(y=y, sr=sr, tightness=tightness)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("bpm", [-1, -0.5, 0])
def test_beat_bad_bpm(ysr, bpm):
    y, sr = ysr
    librosa.beat.beat_track(y=y, sr=sr, bpm=bpm)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("start_bpm", [-1, -0.5, 0])
def test_beat_bad_start_bpm(ysr, start_bpm):
    y, sr = ysr
    librosa.beat.beat_track(y=y, sr=sr, start_bpm=start_bpm)


@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize(
    "units,ctx",
    [("frames", dnr()), ("time", dnr()), ("samples", dnr()), ("bad units", pytest.raises(librosa.ParameterError))],
)
def test_beat_units(ysr, hop_length, units, ctx):

    y, sr = ysr
    with ctx:
        _, b2 = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units=units)
        tempo, b1 = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        t1 = librosa.frames_to_time(b1, sr=sr, hop_length=hop_length)

        if units == "time":
            t2 = b2

        elif units == "samples":
            t2 = librosa.samples_to_time(b2, sr=sr)

        elif units == "frames":
            t2 = librosa.frames_to_time(b2, sr=sr, hop_length=hop_length)

        assert np.allclose(t1, t2)


@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("win_length", [192, 384])
@pytest.mark.parametrize("use_onset", [False, True])
@pytest.mark.parametrize(
    "tempo_min,tempo_max,ctx",
    [(30, 300, dnr()), (None, 240, dnr()), (60, None, dnr()), (120, 80, pytest.raises(librosa.ParameterError))],
)
@pytest.mark.parametrize("prior", [None, scipy.stats.lognorm(s=1, loc=np.log(120), scale=120)])
def test_plp(ysr, hop_length, win_length, tempo_min, tempo_max, use_onset, prior, ctx):

    y, sr = ysr
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    with ctx:
        if use_onset:
            pulse = librosa.beat.plp(
                y=y,
                sr=sr,
                onset_envelope=oenv,
                hop_length=hop_length,
                win_length=win_length,
                tempo_min=tempo_min,
                tempo_max=tempo_max,
                prior=prior,
            )
        else:
            pulse = librosa.beat.plp(
                y=y,
                sr=sr,
                hop_length=hop_length,
                win_length=win_length,
                tempo_min=tempo_min,
                tempo_max=tempo_max,
                prior=prior,
            )

        assert len(pulse) == len(oenv)

        assert np.all(pulse >= 0)
        assert np.all(pulse <= 1)


# Beat tracking regression test is no longer enabled due to librosa's
# corrections
@pytest.mark.skip
@pytest.mark.parametrize("infile", files(os.path.join("data", "beat-beat-*.mat")))
def deprecated_test_beat(infile):

    DATA = load(infile)

    (bpm, beats) = librosa.beat.beat_track(y=None, sr=8000, hop_length=32, onset_envelope=DATA["onsetenv"][0])

    beat_times = librosa.frames_to_time(beats, sr=8000, hop_length=32)
    assert np.allclose(beat_times, DATA["beats"])
