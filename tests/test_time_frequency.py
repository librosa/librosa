#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 19:13:49 by Brian McFee <brian.mcfee@nyu.edu>
'''Unit tests for time and frequency conversion'''

import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

import numpy as np
import librosa
from nose.tools import raises


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

        midi_true = 12 * octave + tuning * 0.01

        if accidental == '#':
            midi_true += 1
        elif accidental in list('b!'):
            midi_true -= 1

        midi = librosa.note_to_midi(note, round_midi=round_midi)
        print midi, midi_true, note
        if round_midi:
            midi_true = np.round(midi_true)
        assert midi == midi_true

        midi = librosa.note_to_midi([note], round_midi=round_midi)
        assert midi[0] == midi_true

    @raises(ValueError)
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


def test_cqt_frequencies():

    def __test(n_bins, fmin, bins_per_octave, tuning):

        freqs = librosa.cqt_frequencies(n_bins,
                                        fmin,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning)

        # Make sure we get the right number of bins
        assert len(freqs) == n_bins

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
