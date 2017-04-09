#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Unit tests for the effects module'''
import warnings

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

from nose.tools import raises, eq_

import librosa
import numpy as np

__EXAMPLE_FILE = 'data/test1_22050.wav'
warnings.resetwarnings()
warnings.simplefilter('always')


def test_time_stretch():

    def __test(infile, rate):
        y, sr = librosa.load(infile, duration=4.0)
        ys = librosa.effects.time_stretch(y, rate)

        orig_duration = librosa.get_duration(y, sr=sr)
        new_duration = librosa.get_duration(ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        assert np.allclose(orig_duration, rate * new_duration,
                           rtol=1e-2, atol=1e-3)

    for rate in [0.25, 0.5, 1.0, 2.0, 4.0]:
        yield __test, 'data/test1_22050.wav', rate

    for rate in [-1, 0]:
        yield raises(librosa.ParameterError)(__test), 'data/test1_22050.wav', rate


def test_pitch_shift():

    def __test(infile, n_steps, bins_per_octave):
        y, sr = librosa.load(infile, duration=4.0)
        ys = librosa.effects.pitch_shift(y, sr, n_steps,
                                         bins_per_octave=bins_per_octave)

        orig_duration = librosa.get_duration(y, sr=sr)
        new_duration = librosa.get_duration(ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        eq_(orig_duration, new_duration)

    for n_steps in np.linspace(-1.5, 1.5, 5):
        for bins_per_octave in [12, 24]:
            yield __test, 'data/test1_22050.wav', n_steps, bins_per_octave

    for bins_per_octave in [-1, 0]:
        yield (raises(librosa.ParameterError)(__test), 'data/test1_22050.wav',
               1, bins_per_octave)


def test_remix_mono():

    # without zc alignment
    y = np.asarray([1, 1, -1, -1, 2, 2, -1, -1, 1, 1])
    y_t = np.asarray([-1, -1, -1, -1, 1, 1, 1, 1, 2, 2])
    intervals = np.asarray([[2, 4],
                            [6, 8],
                            [0, 2],
                            [8, 10],
                            [4, 6]])

    def __test(y, y_t, intervals, align_zeros):
        y_out = librosa.effects.remix(y, intervals,
                                      align_zeros=align_zeros)
        assert np.allclose(y_out, y_t)

    for align_zeros in [False, True]:
        yield __test, y, y_t, intervals, align_zeros


def test_remix_stereo():

    # without zc alignment
    y = np.asarray([1, 1, -1, -1, 2, 2, -1, -1, 1, 1])
    y_t = np.asarray([-1, -1, -1, -1, 1, 1, 1, 1, 2, 2])
    y = np.vstack([y, y])
    y_t = np.vstack([y_t, y_t])

    intervals = np.asarray([[2, 4],
                            [6, 8],
                            [0, 2],
                            [8, 10],
                            [4, 6]])

    def __test(y, y_t, intervals, align_zeros):
        y_out = librosa.effects.remix(y, intervals,
                                      align_zeros=align_zeros)
        assert np.allclose(y_out, y_t), str(y_out)

    for align_zeros in [False, True]:
        yield __test, y, y_t, intervals, align_zeros


def test_hpss():

    y, sr = librosa.load(__EXAMPLE_FILE)

    y_harm, y_perc = librosa.effects.hpss(y)

    # Make sure that the residual energy is generally small
    y_residual = y - y_harm - y_perc

    rms_orig = librosa.feature.rmse(y=y)
    rms_res = librosa.feature.rmse(y=y_residual)

    assert np.percentile(rms_orig, 0.01) > np.percentile(rms_res, 0.99)


def test_percussive():

    y, sr = librosa.load('data/test1_22050.wav')

    yh1, yp1 = librosa.effects.hpss(y)

    yp2 = librosa.effects.percussive(y)

    assert np.allclose(yp1, yp2)


def test_harmonic():

    y, sr = librosa.load('data/test1_22050.wav')

    yh1, yp1 = librosa.effects.hpss(y)

    yh2 = librosa.effects.harmonic(y)

    assert np.allclose(yh1, yh2)


def test_trim():

    def __test(y, top_db, ref, trim_duration):
        yt, idx = librosa.effects.trim(y, top_db=top_db,
                                       ref=ref)

        # Test for index position
        fidx = [slice(None)] * y.ndim
        fidx[-1] = slice(*idx.tolist())
        assert np.allclose(yt, y[fidx])

        # Verify logamp
        rms = librosa.feature.rmse(librosa.to_mono(yt))
        logamp = librosa.logamplitude(rms**2, ref=ref, top_db=None)
        assert np.all(logamp > - top_db)

        # Verify logamp
        rms_all = librosa.feature.rmse(librosa.to_mono(y)).squeeze()
        logamp_all = librosa.logamplitude(rms_all**2, ref=ref,
                                          top_db=None)

        start = int(librosa.samples_to_frames(idx[0]))
        stop = int(librosa.samples_to_frames(idx[1]))
        assert np.all(logamp_all[:start] <= - top_db)
        assert np.all(logamp_all[stop:] <= - top_db)

        # Verify duration
        duration = librosa.get_duration(yt)
        assert np.allclose(duration, trim_duration, atol=1e-1), duration

    # construct 5 seconds of stereo silence
    # Stick a sine wave in the middle three seconds
    sr = float(22050)
    trim_duration = 3.0
    y = np.sin(2 * np.pi * 440. * np.arange(0, trim_duration * sr) / sr)
    y = librosa.util.pad_center(y, 5 * sr)
    y = np.vstack([y, np.zeros_like(y)])

    for top_db in [60, 40, 20]:
        for ref in [1, np.max]:
            # Test stereo
            yield __test, y, top_db, ref, trim_duration
            # Test mono
            yield __test, y[0], top_db, ref, trim_duration


def test_split():

    def __test(hop_length, frame_length, top_db):

        intervals = librosa.effects.split(y,
                                          top_db=top_db,
                                          frame_length=frame_length,
                                          hop_length=hop_length)

        int_match = librosa.util.match_intervals(intervals, idx_true)

        for i in range(len(intervals)):
            i_true = idx_true[int_match[i]]

            assert np.all(np.abs(i_true - intervals[i]) <= frame_length), intervals[i]

    # Make some high-frequency noise
    sr = 8192

    y = np.ones(5 * sr)
    y[::2] *= -1

    # Zero out all but two intervals
    y[:sr] = 0
    y[2 * sr:3 * sr] = 0
    y[4 * sr:] = 0

    # The true non-silent intervals
    idx_true = np.asarray([[sr, 2 * sr],
                           [3 * sr, 4 * sr]])

    for frame_length in [1024, 2048, 4096]:
        for hop_length in [256, 512, 1024]:
            for top_db in [20, 60, 80]:
                yield __test, hop_length, frame_length, top_db

    # Do it again, but without the silence at the beginning
    y = np.ones(5 * sr)
    y[::2] *= -1
    y[4*sr:] = 0
    idx_true = np.asarray([[0, 4 * sr]])
    for frame_length in [1024, 2048, 4096]:
        for hop_length in [256, 512, 1024]:
            for top_db in [20, 60, 80]:
                yield __test, hop_length, frame_length, top_db

    # And without the silence at the end
    y = np.ones(5 * sr)
    y[::2] *= -1
    idx_true = np.asarray([[0, 5 * sr]])
    for frame_length in [1024, 2048, 4096]:
        for hop_length in [256, 512, 1024]:
            for top_db in [20, 60, 80]:
                yield __test, hop_length, frame_length, top_db
