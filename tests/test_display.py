#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 22:51:01 by Brian McFee <brian.mcfee@nyu.edu>
'''Unit tests for display module'''

import warnings

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

import matplotlib.style
matplotlib.style.use('seaborn-ticks')

import matplotlib.pyplot as plt

import librosa
import librosa.display
import numpy as np

from nose.tools import nottest, raises, eq_
from mpl_ic import image_comparison
warnings.resetwarnings()
warnings.simplefilter('always')


@nottest
def get_spec(y, sr):

    C = np.abs(librosa.cqt(y, sr=sr))
    return librosa.stft(y), C, sr


__EXAMPLE_FILE = 'data/test1_22050.wav'
y, sr = librosa.load(__EXAMPLE_FILE)
S, C, sr = get_spec(y, sr)
S_abs = np.abs(S)
S_signed = np.abs(S) - np.median(np.abs(S))
S_bin = S_signed > 0

tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
beats = librosa.util.fix_frames(beats, x_max=C.shape[1])
beat_t = librosa.frames_to_time(beats, sr=sr)
Csync = librosa.util.sync(C, beats, aggregate=np.median)


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


@image_comparison(baseline_images=['tempo'], extensions=['png'])
def test_tempo():
    T = librosa.feature.tempogram(y=y, sr=sr)

    plt.figure()
    librosa.display.specshow(T, y_axis='tempo', cmap='magma')


@image_comparison(baseline_images=['tonnetz'], extensions=['png'])
def test_tonnetz():
    plt.figure()
    chroma = librosa.feature.chroma_cqt(C=C)
    ton = librosa.feature.tonnetz(chroma=chroma)
    librosa.display.specshow(ton, y_axis='tonnetz')


@image_comparison(baseline_images=['chroma'], extensions=['png'])
def test_chroma():
    plt.figure()
    plt.subplot(3, 1, 1)
    chr1 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr)
    librosa.display.specshow(chr1, y_axis='chroma')

    plt.subplot(3, 1, 2)
    chr2 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr, n_chroma=2*12)
    librosa.display.specshow(chr2, y_axis='chroma', bins_per_octave=2*12)

    plt.subplot(3, 1, 3)
    chr3 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr, n_chroma=3*12)
    librosa.display.specshow(chr3, y_axis='chroma', bins_per_octave=3*12)


@image_comparison(baseline_images=['double_chroma'], extensions=['png'])
def test_double_chroma():
    plt.figure()

    chr1 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr)
    chr1 = np.vstack((chr1, chr1))
    librosa.display.specshow(chr1, y_axis='chroma', bins_per_octave=12)


@image_comparison(baseline_images=['x_mel'], extensions=['png'])
def test_x_mel():
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M.T, x_axis='mel')


@image_comparison(baseline_images=['y_mel'], extensions=['png'])
def test_y_mel():
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M, y_axis='mel')


@image_comparison(baseline_images=['y_mel_bounded'], extensions=['png'])
def test_y_mel_bounded():
    plt.figure()

    fmin, fmax = 110, 880
    M = librosa.feature.melspectrogram(S=S_abs**2, fmin=fmin, fmax=fmax)
    librosa.display.specshow(M, y_axis='mel', fmin=fmin, fmax=fmax)


@image_comparison(baseline_images=['x_none_y_linear'], extensions=['png'])
def test_xaxis_none_yaxis_linear():
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs, y_axis='linear')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed, y_axis='linear')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin, y_axis='linear')


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


@image_comparison(baseline_images=['time_scales_auto'], extensions=['png'])
def test_time_scales_auto():

    # sr = 22050, hop_length = 512, S.shape[1] = 198
    # 197 * 512 / 22050 ~= 4.6s
    plt.figure()
    plt.subplot(4, 1, 1)
    # sr * 10 -> ms
    librosa.display.specshow(S_abs, sr=10 * sr, x_axis='time')

    plt.subplot(4, 1, 2)
    # sr -> s
    librosa.display.specshow(S_abs, sr=sr, x_axis='time')

    plt.subplot(4, 1, 3)
    # sr / 20 -> m
    librosa.display.specshow(S_abs, sr=sr // 20, x_axis='time')

    plt.subplot(4, 1, 4)
    # sr / (60 * 20) -> h
    librosa.display.specshow(S_abs, sr=sr // (60 * 20), x_axis='time')

    plt.tight_layout()


@image_comparison(baseline_images=['waveplot_mono'], extensions=['png'])
def test_waveplot_mono():

    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(y, sr=sr, max_points=None, x_axis='off')

    plt.subplot(3, 1, 2)
    librosa.display.waveplot(y, sr=sr, x_axis='off')

    plt.subplot(3, 1, 3)
    librosa.display.waveplot(y, sr=sr, x_axis='time')


@image_comparison(baseline_images=['waveplot_stereo'], extensions=['png'])
def test_waveplot_stereo():

    ys = np.vstack([y[np.newaxis, :], 2 * y[np.newaxis, :]])

    plt.figure()
    librosa.display.waveplot(ys, sr=sr)


@raises(librosa.ParameterError)
def test_unknown_wavaxis():

    plt.figure()
    librosa.display.waveplot(y, sr=sr, x_axis='something not in the axis map')


@raises(librosa.ParameterError)
def test_waveplot_bad_maxsr():

    plt.figure()
    librosa.display.waveplot(y, sr=sr, max_sr=0)


@raises(librosa.ParameterError)
def test_waveplot_bad_maxploints():
    plt.figure()
    librosa.display.waveplot(y, sr=sr, max_points=0)


def test_unknown_axis():

    @raises(librosa.ParameterError)
    def __test(axis):
        kwargs = dict()
        kwargs.setdefault(axis, 'something not in the axis map')
        plt.figure()
        librosa.display.specshow(S_abs, **kwargs)

    yield __test, 'x_axis'
    yield __test, 'y_axis'


def test_cmap_robust():

    def __test(data):
        cmap1 = librosa.display.cmap(data, robust=False)
        cmap2 = librosa.display.cmap(data, robust=True)

        assert type(cmap1) is type(cmap2)

        if isinstance(cmap1, matplotlib.colors.ListedColormap):
            assert np.allclose(cmap1.colors, cmap2.colors)
        elif isinstance(cmap1, matplotlib.colors.LinearSegmentedColormap):
            eq_(cmap1.name, cmap2.name)
        else:
            eq_(cmap1, cmap2)

    # Inputs here are constructed to not need robust sign estimation
    for D in [1.0 + S_abs, -(1.0 + S_abs), S_signed, S_bin]:
        yield __test, D


@image_comparison(baseline_images=['coords'], extensions=['png'])
def test_coords():

    plt.figure()
    librosa.display.specshow(Csync, x_coords=beat_t, x_axis='time', y_axis='cqt_note')


@raises(librosa.ParameterError)
def test_bad_coords():

    librosa.display.specshow(S_abs, x_coords=np.arange(S.shape[1] // 2))
