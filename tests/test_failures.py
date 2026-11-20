#!/usr/bin/env python
# CREATED:2014-12-29 10:52:23 by Brian McFee <brian.mcfee@nyu.edu>
# unit tests for ill-formed inputs

# Disable cache
import contextlib
import os

with contextlib.suppress(KeyError):
    os.environ.pop("LIBROSA_CACHE_DIR")

import numpy as np
import pytest

import librosa


def test_valid_audio_int():
    y = np.zeros(10, dtype=int)
    with pytest.raises(librosa.ParameterTypeError):
        librosa.util.valid_audio(y)


def test_valid_audio_scalar():
    y = np.array(0.0)
    with pytest.raises(librosa.ParameterShapeError):
        librosa.util.valid_audio(y)


def test_valid_stereo_or_mono():
    """valid_audio: mono=False, y.ndim==1"""
    y = np.zeros(1000)
    librosa.util.valid_audio(y)


def test_valid_stereo():
    """valid_audio: mono=False, y.ndim==2"""
    y = np.zeros((1000, 2)).T
    librosa.util.valid_audio(y)


def test_valid_audio_type():
    """valid_audio: list input"""
    y = list(np.zeros(1000))
    with pytest.raises(librosa.ParameterTypeError):
        librosa.util.valid_audio(y)  # type: ignore


def test_valid_audio_nan():
    """valid_audio: NaN"""
    y = np.zeros(1000)
    y[10] = np.nan
    with pytest.raises(librosa.ParameterValueError):
        librosa.util.valid_audio(y)


def test_valid_audio_inf():
    """valid_audio: Inf"""
    y = np.zeros(1000)
    y[10] = np.inf
    with pytest.raises(librosa.ParameterValueError):
        librosa.util.valid_audio(y)


def test_valid_audio_strided():
    """valid_audio: strided"""
    y = np.zeros(1000)[::2]
    librosa.util.valid_audio(y)


def test_valid_audio_clang():
    """valid_audio: C-contiguous"""
    y = np.zeros(1000).reshape(2, 500)
    librosa.util.valid_audio(y)


def test_frame_hop():
    """frame: hop_length=0"""
    y = np.zeros(128)
    with pytest.raises(librosa.ParameterValueError):
        librosa.util.frame(y, frame_length=10, hop_length=0)


def test_frame_size():
    # frame: len(y) == 128, frame_length==256, hop_length=128
    y = np.zeros(64)
    with pytest.raises(librosa.ParameterShapeError):
        librosa.util.frame(y, frame_length=256, hop_length=128)


def test_frame_size_difference():
    # In response to issue #385
    # https://github.com/librosa/librosa/issues/385
    # frame: len(y) == 129, frame_length==256, hop_length=128
    y = np.zeros(129)
    with pytest.raises(librosa.ParameterShapeError):
        librosa.util.frame(y, frame_length=256, hop_length=128)


def test_stft_bad_window():

    y = np.zeros(22050 * 5)

    n_fft = 2048
    window = np.ones(n_fft // 2)

    with pytest.raises(librosa.ParameterShapeError):
        librosa.stft(y, n_fft=n_fft, window=window)


def test_istft_bad_window():

    D = np.zeros((1025, 10), dtype=np.complex64)

    n_fft = 2 * (D.shape[0] - 1)

    window = np.ones(n_fft // 2)

    with pytest.raises(librosa.ParameterShapeError):
        librosa.istft(D, window=window)


@pytest.mark.parametrize("y", [np.empty(22050)])
@pytest.mark.parametrize("mode", ["wrap", "maximum", "minimum", "median", "mean"])
def test_stft_bad_pad(y, mode):
    with pytest.raises(librosa.ParameterValueError):
        librosa.stft(y, pad_mode=mode)
