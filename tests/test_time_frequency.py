#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 19:13:49 by Brian McFee <brian.mcfee@nyu.edu>
"""Unit tests for time and frequency conversion"""
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import warnings
import librosa
import numpy as np
import pytest


@pytest.mark.parametrize("frames", [100, np.arange(10.0), np.ones((3, 3))], ids=["0d", "1d", "2d"])
@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize("n_fft", [None, 1024])
def test_frames_to_samples(frames, hop_length, n_fft):

    samples = librosa.frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    frames = np.asanyarray(frames)
    assert frames.shape == samples.shape
    assert frames.ndim == samples.ndim
    if n_fft is None:
        assert np.allclose(samples, frames * hop_length)
    else:
        assert np.allclose((samples - n_fft // 2) // hop_length, frames)


@pytest.mark.parametrize(
    "samples", [1024 * 100, 1024 * np.arange(10.0), 1024 * np.ones((3, 3))], ids=["0d", "1d", "2d"]
)
@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize("n_fft", [None, 1024])
def test_samples_to_frames(samples, hop_length, n_fft):

    frames = librosa.samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)
    samples = np.asanyarray(samples)
    assert frames.shape == samples.shape
    assert frames.ndim == samples.ndim
    if n_fft is None:
        assert np.allclose(samples, frames * hop_length)
    else:
        assert np.allclose((samples - n_fft // 2) // hop_length, frames)


@pytest.mark.parametrize("sr", [22050, 44100])
@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("n_fft", [None, 2048])
def test_frames_to_time(sr, hop_length, n_fft):

    # Generate frames at times 0s, 1s, 2s
    frames = np.arange(3) * sr // hop_length

    if n_fft:
        frames -= n_fft // (2 * hop_length)

    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length, n_fft=n_fft)

    # we need to be within one frame
    assert np.all(np.abs(times - np.asarray([0, 1, 2])) * sr < hop_length)


@pytest.mark.parametrize("sr", [22050, 44100])
def test_time_to_samples(sr):

    assert np.allclose(librosa.time_to_samples([0, 1, 2], sr=sr), [0, sr, 2 * sr])


@pytest.mark.parametrize("sr", [22050, 44100])
def test_samples_to_time(sr):

    assert np.allclose(librosa.samples_to_time([0, sr, 2 * sr], sr=sr), [0, 1, 2])


@pytest.mark.parametrize("sr", [22050, 44100])
@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("n_fft", [None, 2048])
def test_time_to_frames(sr, hop_length, n_fft):

    # Generate frames at times 0s, 1s, 2s
    times = np.arange(3)

    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop_length, n_fft=n_fft)

    if n_fft:
        frames -= n_fft // (2 * hop_length)

    # we need to be within one frame
    assert np.all(np.abs(times - np.asarray([0, 1, 2])) * sr < hop_length)


@pytest.mark.parametrize("tuning", [0.0, -0.2, 0.1])
@pytest.mark.parametrize("bins_per_octave", [12, 24, 36])
def test_octs_to_hz(tuning, bins_per_octave):
    freq = np.asarray([55, 110, 220, 440]) * (2.0 ** (tuning / bins_per_octave))
    freq_out = librosa.octs_to_hz([1, 2, 3, 4], tuning=tuning, bins_per_octave=bins_per_octave)

    assert np.allclose(freq, freq_out)


@pytest.mark.parametrize("tuning", [0.0, -0.2, 0.1])
@pytest.mark.parametrize("bins_per_octave", [12, 24, 36])
def test_hz_to_octs(tuning, bins_per_octave):
    freq = np.asarray([55, 110, 220, 440]) * (2.0 ** (tuning / bins_per_octave))
    octs = [1, 2, 3, 4]
    oct_out = librosa.hz_to_octs(freq, tuning=tuning, bins_per_octave=bins_per_octave)

    assert np.allclose(octs, oct_out)


@pytest.mark.parametrize(
    "A4,bins_per_octave,tuning",
    [
        (440.0, 12, 0.0),
        ([440.0, 444.0], 24, [0.0, 0.31335]),
        ([432.0], 12, [-0.317667]),
        (432.0, 36, -0.953)
    ],
)
def test_A4_to_tuning(A4, bins_per_octave, tuning):
    tuning_out = librosa.A4_to_tuning(A4=A4, bins_per_octave=bins_per_octave)
    assert np.allclose(np.asarray(tuning), tuning_out)


@pytest.mark.parametrize(
    "tuning,bins_per_octave,A4",
    [
        (0.0, 12, 440.0),
        ([-0.2], 24, [437.466]),
        ([0.1, 0.9], 36, [440.848, 447.691]),
        (0.0, 24, 440.0)
    ],
)
def test_tuning_to_A4(tuning, bins_per_octave, A4):
    A4_out = librosa.tuning_to_A4(tuning=tuning, bins_per_octave=bins_per_octave)
    assert np.allclose(np.asarray(A4), A4_out)


@pytest.mark.parametrize(
    "tuning,octave",
    [(None, None), (None, 1), (None, 2), (None, 3), (-25, 1), (-25, 2), (-25, 3), (0, 1), (0, 2), (0, 3)],
)
@pytest.mark.parametrize("accidental", ["", "#", "b", "!"])
@pytest.mark.parametrize("round_midi", [False, True])
def test_note_to_midi(tuning, accidental, octave, round_midi):

    note = "C{:s}".format(accidental)

    if octave is not None:
        note = "{:s}{:d}".format(note, octave)
    else:
        octave = 0

    if tuning is not None:
        note = "{:s}{:+d}".format(note, tuning)
    else:
        tuning = 0

    midi_true = 12 * (octave + 1) + tuning * 0.01

    if accidental == "#":
        midi_true += 1
    elif accidental in list("b!"):
        midi_true -= 1

    midi = librosa.note_to_midi(note, round_midi=round_midi)
    if round_midi:
        midi_true = np.round(midi_true)
    assert midi == midi_true

    midi = librosa.note_to_midi([note], round_midi=round_midi)
    assert midi[0] == midi_true


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_note_to_midi_badnote():
    librosa.note_to_midi("does not pass")


@pytest.mark.parametrize(
    "tuning,octave",
    [(None, None), (None, 1), (None, 2), (None, 3), (-25, 1), (-25, 2), (-25, 3), (0, 1), (0, 2), (0, 3)],
)
@pytest.mark.parametrize("accidental", ["", "#", "b", "!"])
@pytest.mark.parametrize("round_midi", [False, True])
def test_note_to_hz(tuning, octave, accidental, round_midi):

    note = "A{:s}".format(accidental)

    if octave is not None:
        note = "{:s}{:d}".format(note, octave)
    else:
        octave = 0

    if tuning is not None:
        note = "{:s}{:+d}".format(note, tuning)
    else:
        tuning = 0

    if round_midi:
        tuning = np.around(tuning, -2)

    hz_true = 440.0 * (2.0 ** (tuning * 0.01 / 12)) * (2.0 ** (octave - 4))

    if accidental == "#":
        hz_true *= 2.0 ** (1.0 / 12)
    elif accidental in list("b!"):
        hz_true /= 2.0 ** (1.0 / 12)

    hz = librosa.note_to_hz(note, round_midi=round_midi)
    assert np.allclose(hz, hz_true)

    hz = librosa.note_to_hz([note], round_midi=round_midi)
    assert np.allclose(hz[0], hz_true)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_note_to_hz_badnote():
    librosa.note_to_hz("does not pass")


@pytest.mark.parametrize(
    "midi_num,note,octave,cents",
    [
        (24.25, "C", False, False),
        (24.25, "C1", True, False),
        (24.25, "C1+25", True, True),
        ([24.25], ["C"], False, False),
    ],
)
def test_midi_to_note(midi_num, note, octave, cents):

    note_out = librosa.midi_to_note(midi_num, octave=octave, cents=cents)
    assert note_out == note


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_midi_to_note_cents_nooctave():
    librosa.midi_to_note(24.25, octave=False, cents=True)


def test_midi_to_hz():

    assert np.allclose(librosa.midi_to_hz([33, 45, 57, 69]), [55, 110, 220, 440])


def test_hz_to_midi():
    assert np.allclose(librosa.hz_to_midi(55), 33)
    assert np.allclose(librosa.hz_to_midi([55, 110, 220, 440]), [33, 45, 57, 69])


@pytest.mark.parametrize(
    "hz,note,octave,cents",
    [
        (440, "A", False, False),
        (440, "A4", True, False),
        (440, "A4+0", True, True),
        ([440, 880], ["A4+0", "A5+0"], True, True),
    ],
)
def test_hz_to_note(hz, note, octave, cents):
    note_out = librosa.hz_to_note(hz, octave=octave, cents=cents)
    assert note_out == note


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_hz_to_note_cents_nooctave():
    librosa.hz_to_note(440, octave=False, cents=True)


@pytest.mark.parametrize("sr", [8000, 22050])
@pytest.mark.parametrize("n_fft", [1024, 2048])
def test_fft_frequencies(sr, n_fft):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # DC
    assert freqs[0] == 0

    # Nyquist, positive here for more convenient display purposes
    assert freqs[-1] == sr / 2.0

    # Ensure that the frequencies increase linearly
    dels = np.diff(freqs)
    assert np.allclose(dels, dels[0])


@pytest.mark.parametrize("n_bins", [12, 24, 36])
@pytest.mark.parametrize("fmin", [440.0])
@pytest.mark.parametrize("bins_per_octave", [12, 24, 36])
@pytest.mark.parametrize("tuning", [-0.25, 0.0, 0.25])
def test_cqt_frequencies(n_bins, fmin, bins_per_octave, tuning):

    freqs = librosa.cqt_frequencies(n_bins, fmin, bins_per_octave=bins_per_octave, tuning=tuning)

    # Make sure we get the right number of bins
    assert len(freqs) == n_bins

    # And that the first bin matches fmin by tuning
    assert np.allclose(freqs[0], fmin * 2.0 ** (float(tuning) / bins_per_octave))

    # And that we have constant Q
    Q = np.diff(np.log2(freqs))
    assert np.allclose(Q, 1.0 / bins_per_octave)


@pytest.mark.parametrize("n_bins", [1, 16, 128])
@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("sr", [11025, 22050])
def test_tempo_frequencies(n_bins, hop_length, sr):

    freqs = librosa.tempo_frequencies(n_bins, hop_length=hop_length, sr=sr)

    # Verify the length
    assert len(freqs) == n_bins

    # 0-bin should be infinite
    assert not np.isfinite(freqs[0])

    # remaining bins should be spaced by 1/hop_length
    if n_bins > 1:
        invdiff = (freqs[1:] ** -1) * (60.0 * sr)
        assert np.allclose(invdiff[0], hop_length)
        assert np.allclose(np.diff(invdiff), np.asarray(hop_length)), np.diff(invdiff)


@pytest.mark.parametrize("sr", [8000, 22050])
@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("win_length", [192, 384])
def test_fourier_tempo_frequencies(sr, hop_length, win_length):
    freqs = librosa.fourier_tempo_frequencies(sr=sr, hop_length=hop_length, win_length=win_length)

    # DC
    assert freqs[0] == 0

    # Nyquist, positive here for more convenient display purposes
    assert freqs[-1] == sr * 60 / 2.0 / hop_length

    # Ensure that the frequencies increase linearly
    dels = np.diff(freqs)
    assert np.allclose(dels, dels[0])


@pytest.mark.parametrize("min_db", [None, -40, -80])
def test_A_weighting(min_db):

    # Check that 1KHz is around 0dB
    a_khz = librosa.A_weighting(1000.0, min_db=min_db)
    assert np.allclose(a_khz, 0, atol=1e-3)

    a_range = librosa.A_weighting(np.linspace(2e1, 2e4), min_db=min_db)
    # Check that the db cap works
    if min_db is not None:
        assert not np.any(a_range < min_db)


def test_samples_like():

    X = np.ones((3, 4, 5))
    hop_length = 512

    for axis in (0, 1, 2, -1):

        samples = librosa.samples_like(X, hop_length=hop_length, axis=axis)

        expected_samples = np.arange(X.shape[axis]) * hop_length

        assert np.allclose(samples, expected_samples)


def test_samples_like_scalar():

    X = 7
    hop_length = 512

    samples = librosa.samples_like(X, hop_length=hop_length)

    expected_samples = np.arange(7) * hop_length

    assert np.allclose(samples, expected_samples)


def test_times_like():

    X = np.ones((3, 4, 5))
    sr = 22050
    hop_length = 512

    for axis in (0, 1, 2, -1):

        times = librosa.times_like(X, sr=sr, hop_length=hop_length, axis=axis)

        expected_times = np.arange(X.shape[axis]) * hop_length / float(sr)

        assert np.allclose(times, expected_times)


def test_times_like_scalar():

    X = 7
    sr = 22050
    hop_length = 512

    times = librosa.times_like(X, sr=sr, hop_length=hop_length)

    expected_times = np.arange(7) * hop_length / float(sr)

    assert np.allclose(times, expected_times)


@pytest.mark.parametrize("blocks", [0, 1, [10, 20]])
@pytest.mark.parametrize("block_length", [1, 4, 8])
def test_blocks_to_frames(blocks, block_length):
    frames = librosa.blocks_to_frames(blocks, block_length)

    # Check shape
    assert frames.ndim == np.asarray(blocks).ndim
    assert frames.size == np.asarray(blocks).size

    # Check values
    assert np.allclose(frames, block_length * np.asanyarray(blocks))

    # Check dtype
    assert np.issubdtype(frames.dtype, np.int)


@pytest.mark.parametrize("blocks", [0, 1, [10, 20]])
@pytest.mark.parametrize("block_length", [1, 4, 8])
@pytest.mark.parametrize("hop_length", [1, 512])
def test_blocks_to_samples(blocks, block_length, hop_length):
    samples = librosa.blocks_to_samples(blocks, block_length, hop_length)

    # Check shape
    assert samples.ndim == np.asarray(blocks).ndim
    assert samples.size == np.asarray(blocks).size

    # Check values
    assert np.allclose(samples, np.asanyarray(blocks) * hop_length * block_length)

    # Check dtype
    assert np.issubdtype(samples.dtype, np.int)


@pytest.mark.parametrize("blocks", [0, 1, [10, 20]])
@pytest.mark.parametrize("block_length", [1, 4, 8])
@pytest.mark.parametrize("hop_length", [1, 512])
@pytest.mark.parametrize("sr", [22050, 44100])
def test_blocks_to_time(blocks, block_length, hop_length, sr):
    times = librosa.blocks_to_time(blocks, block_length, hop_length, sr)

    # Check shape
    assert times.ndim == np.asarray(blocks).ndim
    assert times.size == np.asarray(blocks).size

    # Check values
    assert np.allclose(times, np.asanyarray(blocks) * hop_length * block_length / float(sr))

    # Check dtype
    assert np.issubdtype(times.dtype, np.float)
