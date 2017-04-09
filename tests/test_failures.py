#!/usr/bin/env python
# CREATED:2014-12-29 10:52:23 by Brian McFee <brian.mcfee@nyu.edu>
# unit tests for ill-formed inputs

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import numpy as np
import librosa
from nose.tools import raises
import warnings
warnings.resetwarnings()
warnings.simplefilter('always')


@raises(librosa.ParameterError)
def test_mono_valid_stereo():
    '''valid_audio: mono=True,  y.ndim==2'''
    y = np.zeros((2, 1000))
    librosa.util.valid_audio(y, mono=True)


def test_valid_stereo_or_mono():
    '''valid_audio: mono=False, y.ndim==1'''
    y = np.zeros(1000)
    librosa.util.valid_audio(y, mono=False)


def test_valid_mono():
    '''valid_audio: mono=True,  y.ndim==1'''
    y = np.zeros(1000)
    librosa.util.valid_audio(y, mono=True)


def test_valid_stereo():
    '''valid_audio: mono=False, y.ndim==2'''
    y = np.zeros((2, 1000))
    librosa.util.valid_audio(y, mono=False)


@raises(librosa.ParameterError)
def test_valid_audio_type():
    '''valid_audio: list input'''
    y = list(np.zeros(1000))
    librosa.util.valid_audio(y)


@raises(librosa.ParameterError)
def test_valid_audio_nan():
    '''valid_audio: NaN'''
    y = np.zeros(1000)
    y[10] = np.NaN
    librosa.util.valid_audio(y)


@raises(librosa.ParameterError)
def test_valid_audio_inf():
    '''valid_audio: Inf'''
    y = np.zeros(1000)
    y[10] = np.inf
    librosa.util.valid_audio(y)


def test_valid_audio_ndim():
    '''valid_audio: y.ndim > 2'''

    y = np.zeros((3, 10, 10))

    @raises(librosa.ParameterError)
    def __test(mono):
        librosa.util.valid_audio(y, mono=mono)

    for mono in [False, True]:
        yield __test, mono


@raises(librosa.ParameterError)
def test_frame_hop():
    '''frame: hop_length=0'''
    y = np.zeros(128)
    librosa.util.frame(y, frame_length=10, hop_length=0)


@raises(librosa.ParameterError)
def test_frame_discontiguous():
    '''frame: discontiguous input'''
    y = np.zeros((128, 2)).T
    librosa.util.frame(y[0], frame_length=64, hop_length=64)


def test_frame_contiguous():
    '''frame: discontiguous input'''
    y = np.zeros((2, 128))
    librosa.util.frame(y[0], frame_length=64, hop_length=64)


@raises(librosa.ParameterError)
def test_frame_size():
    # frame: len(y) == 128, frame_length==256, hop_length=128
    y = np.zeros(64)
    librosa.util.frame(y, frame_length=256, hop_length=128)


@raises(librosa.ParameterError)
def test_frame_size_difference():
    # In response to issue #385
    # https://github.com/librosa/librosa/issues/385
    # frame: len(y) == 129, frame_length==256, hop_length=128
    y = np.zeros(129)
    librosa.util.frame(y, frame_length=256, hop_length=128)


@raises(librosa.ParameterError)
def test_stft_bad_window():

    y = np.zeros(22050 * 5)

    n_fft = 2048
    window = np.ones(n_fft // 2)

    librosa.stft(y, n_fft=n_fft, window=window)


@raises(librosa.ParameterError)
def test_istft_bad_window():

    D = np.zeros((1025, 10), dtype=np.complex64)

    n_fft = 2 * (D.shape[0] - 1)

    window = np.ones(n_fft // 2)

    librosa.istft(D, window=window)
