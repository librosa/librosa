#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 19:13:49 by Brian McFee <brian.mcfee@nyu.edu>
'''Unit tests for time and frequency conversion'''
import warnings

import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

import librosa
import numpy as np
from nose.tools import raises, eq_

warnings.resetwarnings()
warnings.simplefilter('always')


def test_frames_to_samples():

    def __test(x, y, hop_length, n_fft):
        y_test = librosa.frames_to_samples(x,
                                           hop_length=hop_length,
                                           n_fft=n_fft)
        assert np.allclose(y, y_test)

    x = np.arange(2)
    for hop_length in [512, 1024]:
        for n_fft in [None, 1024]:
            y = x * hop_length
            if n_fft is not None:
                y += n_fft // 2
            yield __test, x, y, hop_length, n_fft


def test_samples_to_frames():

    def __test(x, y, hop_length, n_fft):
        y_test = librosa.samples_to_frames(x,
                                           hop_length=hop_length,
                                           n_fft=n_fft)
        assert np.allclose(y, y_test)

    x = np.arange(2)
    for hop_length in [512, 1024]:
        for n_fft in [None, 1024]:
            y = x * hop_length
            if n_fft is not None:
                y += n_fft // 2
            yield __test, y, x, hop_length, n_fft


def test_frames_to_time():

    def __test(sr, hop_length, n_fft):

        # Generate frames at times 0s, 1s, 2s
        frames = np.arange(3) * sr // hop_length

        if n_fft:
            frames -= n_fft // (2 * hop_length)

        times = librosa.frames_to_time(frames,
                                       sr=sr,
                                       hop_length=hop_length,
                                       n_fft=n_fft)

        # we need to be within one frame
        assert np.all(np.abs(times - np.asarray([0, 1, 2])) * sr
                      < hop_length)

    for sr in [22050, 44100]:
        for hop_length in [256, 512]:
            for n_fft in [None, 2048]:
                yield __test, sr, hop_length, n_fft


def test_time_to_samples():

    def __test(sr):
        assert np.allclose(librosa.time_to_samples([0, 1, 2], sr=sr),
                           [0, sr, 2 * sr])

    for sr in [22050, 44100]:
        yield __test, sr


def test_samples_to_time():

    def __test(sr):
        assert np.allclose(librosa.samples_to_time([0, sr, 2 * sr], sr=sr),
                           [0, 1, 2])

    for sr in [22050, 44100]:
        yield __test, sr


def test_time_to_frames():

    def __test(sr, hop_length, n_fft):

        # Generate frames at times 0s, 1s, 2s
        times = np.arange(3)

        frames = librosa.time_to_frames(times,
                                        sr=sr,
                                        hop_length=hop_length,
                                        n_fft=n_fft)

        if n_fft:
            frames -= n_fft // (2 * hop_length)

        # we need to be within one frame
        assert np.all(np.abs(times - np.asarray([0, 1, 2])) * sr
                      < hop_length)

    for sr in [22050, 44100]:
        for hop_length in [256, 512]:
            for n_fft in [None, 2048]:
                yield __test, sr, hop_length, n_fft


def test_octs_to_hz():
    def __test(a440):
        freq = np.asarray([55, 110, 220, 440]) * (float(a440) / 440.0)
        freq_out = librosa.octs_to_hz([1, 2, 3, 4], A440=a440)

        assert np.allclose(freq, freq_out)

    for a440 in [415, 430, 435, 440, 466]:
        yield __test, a440


def test_hz_to_octs():
    def __test(a440):
        freq = np.asarray([55, 110, 220, 440]) * (float(a440) / 440.0)
        octs = [1, 2, 3, 4]
        oct_out = librosa.hz_to_octs(freq, A440=a440)

        assert np.allclose(octs, oct_out)

    for a440 in [415, 430, 435, 440, 466]:
        yield __test, a440


def test_note_to_midi():

    def __test(tuning, accidental, octave, round_midi):

        note = 'C{:s}'.format(accidental)

        if octave is not None:
            note = '{:s}{:d}'.format(note, octave)
        else:
            octave = 0

        if tuning is not None:
            note = '{:s}{:+d}'.format(note, tuning)
        else:
            tuning = 0

        midi_true = 12 * (octave + 1) + tuning * 0.01

        if accidental == '#':
            midi_true += 1
        elif accidental in list('b!'):
            midi_true -= 1

        midi = librosa.note_to_midi(note, round_midi=round_midi)
        if round_midi:
            midi_true = np.round(midi_true)
        eq_(midi, midi_true)

        midi = librosa.note_to_midi([note], round_midi=round_midi)
        eq_(midi[0], midi_true)

    @raises(librosa.ParameterError)
    def __test_fail():
        librosa.note_to_midi('does not pass')

    for tuning in [None, -25, 0, 25]:
        for octave in [None, 1, 2, 3]:
            if octave is None and tuning is not None:
                continue
            for accidental in ['', '#', 'b', '!']:
                for round_midi in [False, True]:
                    yield __test, tuning, accidental, octave, round_midi

    yield __test_fail


def test_note_to_hz():

    def __test(tuning, accidental, octave, round_midi):

        note = 'A{:s}'.format(accidental)

        if octave is not None:
            note = '{:s}{:d}'.format(note, octave)
        else:
            octave = 0

        if tuning is not None:
            note = '{:s}{:+d}'.format(note, tuning)
        else:
            tuning = 0

        if round_midi:
            tuning = np.around(tuning, -2)

        hz_true = 440.0 * (2.0**(tuning * 0.01 / 12)) * (2.0**(octave - 4))

        if accidental == '#':
            hz_true *= 2.0**(1./12)
        elif accidental in list('b!'):
            hz_true /= 2.0**(1./12)

        hz = librosa.note_to_hz(note, round_midi=round_midi)
        assert np.allclose(hz[0], hz_true)

    @raises(librosa.ParameterError)
    def __test_fail():
        librosa.note_to_midi('does not pass')

    for tuning in [None, -25, 0, 25]:
        for octave in [None, 1, 2, 3]:
            if octave is None and tuning is not None:
                continue
            for accidental in ['', '#', 'b', '!']:
                for round_midi in [False, True]:
                    yield __test, tuning, accidental, octave, round_midi

    yield __test_fail


def test_midi_to_note():

    def __test(midi_num, note, octave, cents):
        note_out = librosa.midi_to_note(midi_num, octave=octave, cents=cents)

        eq_(note_out, note)

    midi_num = 24.25

    yield __test, midi_num, 'C', False, False
    yield __test, midi_num, 'C1', True, False
    yield raises(librosa.ParameterError)(__test), midi_num, 'C+25', False, True
    yield __test, midi_num, 'C1+25', True, True
    yield __test, [midi_num], ['C'], False, False


def test_midi_to_hz():

    assert np.allclose(librosa.midi_to_hz([33, 45, 57, 69]),
                       [55, 110, 220, 440])


def test_hz_to_midi():
    assert np.allclose(librosa.hz_to_midi([55, 110, 220, 440]),
                       [33, 45, 57, 69])


def test_hz_to_note():
    def __test(hz, note, octave, cents):
        note_out = librosa.hz_to_note(hz, octave=octave, cents=cents)

        eq_(list(note_out), list([note]))

    hz = 440

    yield __test, hz, 'A', False, False
    yield __test, hz, 'A4', True, False
    yield raises(librosa.ParameterError)(__test), hz, 'A+0', False, True
    yield __test, hz, 'A4+0', True, True


def test_fft_frequencies():

    def __test(sr, n_fft):
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # DC
        eq_(freqs[0], 0)

        # Nyquist, positive here for more convenient display purposes
        eq_(freqs[-1], sr / 2.0)

        # Ensure that the frequencies increase linearly
        dels = np.diff(freqs)
        assert np.allclose(dels, dels[0])

    for n_fft in [1024, 2048]:
        for sr in [8000, 22050]:
            yield __test, sr, n_fft


def test_cqt_frequencies():

    def __test(n_bins, fmin, bins_per_octave, tuning):

        freqs = librosa.cqt_frequencies(n_bins,
                                        fmin,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning)

        # Make sure we get the right number of bins
        eq_(len(freqs), n_bins)

        # And that the first bin matches fmin by tuning
        assert np.allclose(freqs[0],
                           fmin * 2.0**(float(tuning) / bins_per_octave))

        # And that we have constant Q
        Q = np.diff(np.log2(freqs))
        assert np.allclose(Q, 1./bins_per_octave)

    for n_bins in [12, 24, 36]:
        for fmin in [440.0]:
            for bins_per_octave in [12, 24, 36]:
                for tuning in [-0.25, 0.0, 0.25]:
                    yield __test, n_bins, fmin, bins_per_octave, tuning


def test_tempo_frequencies():

    def __test(n_bins, hop_length, sr):

        freqs = librosa.tempo_frequencies(n_bins, hop_length=hop_length, sr=sr)

        # Verify the length
        eq_(len(freqs), n_bins)

        # 0-bin should be infinite
        assert not np.isfinite(freqs[0])

        # remaining bins should be spaced by 1/hop_length
        if n_bins > 1:
            invdiff = (freqs[1:]**-1) * (60.0 * sr)
            assert np.allclose(invdiff[0], hop_length)
            assert np.allclose(np.diff(invdiff), np.asarray(hop_length)), np.diff(invdiff)

    for n_bins in [1, 16, 128]:
        for hop_length in [256, 512, 1024]:
            for sr in [11025, 22050, 44100]:
                yield __test, n_bins, hop_length, sr


def test_A_weighting():

    def __test(min_db):
        # Check that 1KHz is around 0dB
        a_khz = librosa.A_weighting(1000.0, min_db=min_db)
        assert np.allclose(a_khz, 0, atol=1e-3)

        a_range = librosa.A_weighting(np.linspace(2e1, 2e4),
                                      min_db=min_db)
        # Check that the db cap works
        if min_db is not None:
            assert not np.any(a_range < min_db)

    for min_db in [None, -40, -80]:
        yield __test, min_db
