#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 22:51:01 by Brian McFee <brian.mcfee@nyu.edu>
'''Unit tests for display module'''


# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='white')

import librosa
import numpy as np

from nose.tools import nottest
from mpl_ic import image_comparison


@nottest
def get_spec():

    __EXAMPLE_FILE = 'data/test1_22050.wav'

    y, sr = librosa.load(__EXAMPLE_FILE)

    C = librosa.cqt(y, sr=sr)
    return librosa.stft(y), C, sr

S, C, sr = get_spec()
S_abs = np.abs(S)
S_signed = np.abs(S) - np.median(np.abs(S))
S_bin = S_signed > 0


@image_comparison(baseline_images=['complex'], extensions=['png'])
def test_complex_input():
    plt.figure()
    librosa.display.specshow(S)


@image_comparison(baseline_images=['abs'], extensions=['png'])
def test_abs_input():
    plt.figure()
    librosa.display.specshow(S_abs)


@image_comparison(baseline_images=['cqt_note'], extensions=['png'])
def test_cqt_note():
    plt.figure()
    librosa.display.specshow(C, y_axis='cqt_note')


@image_comparison(baseline_images=['cqt_hz'], extensions=['png'])
def test_cqt_hz():
    plt.figure()
    librosa.display.specshow(C, y_axis='cqt_hz')


@image_comparison(baseline_images=['chroma'], extensions=['png'])
def test_chroma():
    chr1 = librosa.feature.chromagram(S=S_abs**2, sr=sr)

    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(chr1, y_axis='chroma')

    chr2 = librosa.feature.chromagram(S=S_abs**2, sr=sr, n_chroma=24)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(chr2, y_axis='chroma', bins_per_octave=24)


@image_comparison(baseline_images=['x_none_y_linear'], extensions=['png'])
def test_xaxis_none_yaxis_linear():
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs, x_axis='linear')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed, x_axis='linear')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin, x_axis='linear')


@image_comparison(baseline_images=['x_none_y_log'], extensions=['png'])
def test_xaxis_none_yaxis_log():
    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs, y_axis='log')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed, y_axis='log')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin, y_axis='log')


@image_comparison(baseline_images=['x_linear_y_none'], extensions=['png'])
def test_xaxis_linear_yaxis_none():
    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis='linear')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis='linear')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin.T, x_axis='linear')


@image_comparison(baseline_images=['x_log_y_none'], extensions=['png'])
def test_xaxis_log_yaxis_none():

    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis='log')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis='log')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin.T, x_axis='log')


@image_comparison(baseline_images=['x_time_y_none'], extensions=['png'])
def test_xaxis_time_yaxis_none():

    plt.figure()
    librosa.display.specshow(S_abs, x_axis='time')


@image_comparison(baseline_images=['x_none_y_time'], extensions=['png'])
def test_xaxis_none_yaxis_time():

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis='time')


@image_comparison(baseline_images=['x_frames_y_none'], extensions=['png'])
def test_xaxis_frames_yaxis_none():

    plt.figure()
    librosa.display.specshow(S_abs, x_axis='frames')


@image_comparison(baseline_images=['x_none_y_frames'], extensions=['png'])
def test_xaxis_none_yaxis_frames():

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis='frames')


@image_comparison(baseline_images=['x_lag_y_none'], extensions=['png'])
def test_xaxis_lag_yaxis_none():

    plt.figure()
    librosa.display.specshow(S_abs, x_axis='lag')


@image_comparison(baseline_images=['x_none_y_lag'], extensions=['png'])
def test_xaxis_time_yaxis_lag():

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis='lag')
