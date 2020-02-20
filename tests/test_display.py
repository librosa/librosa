#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 22:51:01 by Brian McFee <brian.mcfee@nyu.edu>
"""Unit tests for display module"""

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

import matplotlib.style

matplotlib.style.use("seaborn-ticks")

import matplotlib.pyplot as plt

import librosa
import librosa.display
import numpy as np

import pytest


@pytest.fixture
def audio():

    __EXAMPLE_FILE = os.path.join("tests", "data", "test1_22050.wav")
    y, sr = librosa.load(__EXAMPLE_FILE)
    return y, sr


@pytest.fixture
def y(audio):
    return audio[0]


@pytest.fixture
def sr(audio):
    return audio[1]


@pytest.fixture
def S(y):
    return librosa.stft(y)


@pytest.fixture
def S_abs(S):
    return np.abs(S)


@pytest.fixture
def C(y, sr):
    return np.abs(librosa.cqt(y, sr=sr))


@pytest.fixture
def S_signed(S):
    return np.abs(S) - np.median(np.abs(S))


@pytest.fixture
def S_bin(S_signed):
    return S_signed > 0


@pytest.fixture
def rhythm(y, sr):
    return librosa.beat.beat_track(y=y, sr=sr)


@pytest.fixture
def tempo(rhythm):
    return rhythm[0]


@pytest.fixture
def beats(rhythm, C):
    return librosa.util.fix_frames(rhythm[1], x_max=C.shape[1])


@pytest.fixture
def beat_t(beats, sr):
    return librosa.frames_to_time(beats, sr)


@pytest.fixture
def Csync(C, beats):
    return librosa.util.sync(C, beats, aggregate=np.median)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_unknown_time_unit(y):
    times = np.arange(len(y))
    plt.figure()
    ax = plt.gca()
    ax.plot(times, y)
    ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit="neither s, nor ms, nor None"))
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["complex"], extensions=["png"], tolerance=6)
def test_complex_input(S):
    plt.figure()
    librosa.display.specshow(S)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["abs"], extensions=["png"], tolerance=6)
def test_abs_input(S_abs):
    plt.figure()
    librosa.display.specshow(S_abs)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["cqt_note"], extensions=["png"], tolerance=6)
def test_cqt_note(C):
    plt.figure()
    librosa.display.specshow(C, y_axis="cqt_note")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["cqt_hz"], extensions=["png"], tolerance=6)
def test_cqt_hz(C):
    plt.figure()
    librosa.display.specshow(C, y_axis="cqt_hz")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["tempo"], extensions=["png"], tolerance=6)
def test_tempo(y, sr):
    T = librosa.feature.tempogram(y=y, sr=sr)

    plt.figure()
    librosa.display.specshow(T, y_axis="tempo", cmap="magma")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["fourier_tempo"], extensions=["png"], tolerance=6)
def test_fourier_tempo(y, sr):
    T = librosa.feature.fourier_tempogram(y=y, sr=sr)

    plt.figure()
    librosa.display.specshow(np.abs(T), y_axis="fourier_tempo", cmap="magma")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["tonnetz"], extensions=["png"], tolerance=6)
def test_tonnetz(C):
    plt.figure()
    chroma = librosa.feature.chroma_cqt(C=C)
    ton = librosa.feature.tonnetz(chroma=chroma)
    librosa.display.specshow(ton, y_axis="tonnetz")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["chroma"], extensions=["png"], tolerance=6)
def test_chroma(S_abs, sr):
    plt.figure()
    plt.subplot(3, 1, 1)
    chr1 = librosa.feature.chroma_stft(S=S_abs ** 2, sr=sr)
    librosa.display.specshow(chr1, y_axis="chroma")

    plt.subplot(3, 1, 2)
    chr2 = librosa.feature.chroma_stft(S=S_abs ** 2, sr=sr, n_chroma=2 * 12)
    librosa.display.specshow(chr2, y_axis="chroma", bins_per_octave=2 * 12)

    plt.subplot(3, 1, 3)
    chr3 = librosa.feature.chroma_stft(S=S_abs ** 2, sr=sr, n_chroma=3 * 12)
    librosa.display.specshow(chr3, y_axis="chroma", bins_per_octave=3 * 12)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["double_chroma"], extensions=["png"], tolerance=6)
def test_double_chroma(S_abs, sr):
    plt.figure()

    chr1 = librosa.feature.chroma_stft(S=S_abs ** 2, sr=sr)
    chr1 = np.vstack((chr1, chr1))
    librosa.display.specshow(chr1, y_axis="chroma", bins_per_octave=12)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_mel"], extensions=["png"], tolerance=6)
def test_x_mel(S_abs):
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs ** 2)
    librosa.display.specshow(M.T, x_axis="mel")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["y_mel"], extensions=["png"], tolerance=6)
def test_y_mel(S_abs):
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs ** 2)
    librosa.display.specshow(M, y_axis="mel")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["y_mel_bounded"], extensions=["png"], tolerance=6)
def test_y_mel_bounded(S_abs):
    plt.figure()

    fmin, fmax = 110, 880
    M = librosa.feature.melspectrogram(S=S_abs ** 2, fmin=fmin, fmax=fmax)
    librosa.display.specshow(M, y_axis="mel", fmin=fmin, fmax=fmax)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_linear"], extensions=["png"], tolerance=6)
def test_xaxis_none_yaxis_linear(S_abs, S_signed, S_bin):
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs, y_axis="linear")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed, y_axis="linear")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin, y_axis="linear")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["specshow_ext_axes"], extensions=["png"], tolerance=6)
def test_specshow_ext_axes(S_abs):
    plt.figure()
    ax_left = plt.subplot(1, 2, 1)
    ax_right = plt.subplot(1, 2, 2)

    # implicitly ax_right
    librosa.display.specshow(S_abs, cmap="gray")
    librosa.display.specshow(S_abs, cmap="magma", ax=ax_left)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_log"], extensions=["png"], tolerance=6)
def test_xaxis_none_yaxis_log(S_abs, S_signed, S_bin):
    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs, y_axis="log")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed, y_axis="log")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin, y_axis="log")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_linear_y_none"], extensions=["png"], tolerance=6)
def test_xaxis_linear_yaxis_none(S_abs, S_signed, S_bin):
    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="linear")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="linear")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin.T, x_axis="linear")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_log_y_none"], extensions=["png"], tolerance=6)
def test_xaxis_log_yaxis_none(S_abs, S_signed, S_bin):

    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="log")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="log")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin.T, x_axis="log")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_time_y_none"], extensions=["png"], tolerance=6)
def test_xaxis_time_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="time")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_time"], extensions=["png"], tolerance=6)
def test_xaxis_none_yaxis_time(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="time")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_frames_y_none"], extensions=["png"], tolerance=6)
def test_xaxis_frames_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="frames")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_frames"], extensions=["png"], tolerance=6)
def test_xaxis_none_yaxis_frames(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="frames")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_lag_y_none"], extensions=["png"], tolerance=6)
def test_xaxis_lag_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="lag")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_lag"], extensions=["png"], tolerance=6)
def test_xaxis_time_yaxis_lag(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="lag")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["time_scales_auto"], extensions=["png"], tolerance=6)
def test_time_scales_auto(S_abs, sr):

    # sr = 22050, hop_length = 512, S.shape[1] = 198
    # 197 * 512 / 22050 ~= 4.6s
    plt.figure(figsize=(10, 10))
    plt.subplot(4, 1, 1)
    # sr * 10 -> ms
    librosa.display.specshow(S_abs, sr=10 * sr, x_axis="time")

    plt.subplot(4, 1, 2)
    # sr -> s
    librosa.display.specshow(S_abs, sr=sr, x_axis="time")

    plt.subplot(4, 1, 3)
    # sr / 20 -> m
    librosa.display.specshow(S_abs, sr=sr // 20, x_axis="time")

    plt.subplot(4, 1, 4)
    # sr / (60 * 20) -> h
    librosa.display.specshow(S_abs, sr=sr // (60 * 20), x_axis="time")

    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["time_unit"], extensions=["png"], tolerance=6)
def test_time_unit(S_abs, sr):

    # sr = 22050, hop_length = 512, S.shape[1] = 198
    # 197 * 512 / 22050 ~= 4.6s
    plt.figure(figsize=(9, 10))
    plt.subplot(3, 1, 1)
    # time scale auto
    librosa.display.specshow(S_abs, sr=sr, x_axis="time")

    plt.subplot(3, 1, 2)
    # time unit fixed to 's'
    librosa.display.specshow(S_abs, sr=sr, x_axis="s")

    plt.subplot(3, 1, 3)
    # time unit fixed to 'ms'
    librosa.display.specshow(S_abs, sr=sr, x_axis="ms")

    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["time_unit_lag"], extensions=["png"], tolerance=6)
def test_time_unit_lag(S_abs, sr):

    plt.figure(figsize=(9, 10))
    plt.subplot(3, 1, 1)
    # time scale auto in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag")

    plt.subplot(3, 1, 2)
    # time unit fixed to 's' in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag_s")

    plt.subplot(3, 1, 3)
    # time unit fixed to 'ms' in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag_ms")

    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["waveplot_mono"], extensions=["png"], tolerance=6)
def test_waveplot_mono(y, sr):

    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(y, sr=sr, max_points=None, x_axis="off")

    plt.subplot(3, 1, 2)
    librosa.display.waveplot(y, sr=sr, x_axis="off")

    plt.subplot(3, 1, 3)
    librosa.display.waveplot(y, sr=sr, x_axis="time")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["waveplot_ext_axes"], extensions=["png"], tolerance=6)
def test_waveplot_ext_axes(y):
    plt.figure()
    ax_left = plt.subplot(1, 2, 1)
    ax_right = plt.subplot(1, 2, 2)

    # implicitly ax_right
    librosa.display.waveplot(y, color="blue")
    librosa.display.waveplot(y, color="red", ax=ax_left)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["waveplot_stereo"], extensions=["png"], tolerance=6)
def test_waveplot_stereo(y, sr):

    ys = librosa.util.stack([y, 2 * y])

    plt.figure()
    librosa.display.waveplot(ys, sr=sr)
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_unknown_wavaxis(y, sr):

    plt.figure()
    librosa.display.waveplot(y, sr=sr, x_axis="something not in the axis map")
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_waveplot_bad_maxsr(y, sr):

    plt.figure()
    librosa.display.waveplot(y, sr=sr, max_sr=0)
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_waveplot_bad_maxpoints(y, sr):
    plt.figure()
    librosa.display.waveplot(y, sr=sr, max_points=0)
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("axis", ["x_axis", "y_axis"])
def test_unknown_axis(S_abs, axis):

    kwargs = dict()
    kwargs.setdefault(axis, "something not in the axis map")
    plt.figure()
    librosa.display.specshow(S_abs, **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        np.arange(1, 10.0),  # strictly positive
        -np.arange(1, 10.0),  # strictly negative
        np.arange(-3, 4.0),  # signed,
        np.arange(2, dtype=np.bool),
    ],
)  # binary
def test_cmap_robust(data):

    cmap1 = librosa.display.cmap(data, robust=False)
    cmap2 = librosa.display.cmap(data, robust=True)

    assert type(cmap1) is type(cmap2)

    if isinstance(cmap1, matplotlib.colors.ListedColormap):
        assert np.allclose(cmap1.colors, cmap2.colors)
    elif isinstance(cmap1, matplotlib.colors.LinearSegmentedColormap):
        assert cmap1.name == cmap2.name
    else:
        assert cmap1 == cmap2


@pytest.mark.mpl_image_compare(baseline_images=["coords"], extensions=["png"], tolerance=6)
def test_coords(Csync, beat_t):

    plt.figure()
    librosa.display.specshow(Csync, x_coords=beat_t, x_axis="time", y_axis="cqt_note")
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_bad_coords(S_abs):

    librosa.display.specshow(S_abs, x_coords=np.arange(S_abs.shape[1] // 2))
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["sharex_specshow_ms"], extensions=["png"], tolerance=6)
def test_sharex_specshow_ms(S_abs, y, sr):

    # Correct time range ~= 4.6 s or 4600ms
    # Due to shared x_axis, both plots are plotted in 's'.
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_abs, ref=np.max), x_axis="time")
    plt.xlabel("")  # hide the x label here, which is not propagated automatically
    plt.subplot(2, 1, 2, sharex=ax)
    librosa.display.waveplot(y, sr, x_axis="ms")
    plt.xlabel("")  # hide the x label here, which is not propagated automatically
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["sharex_waveplot_ms"], extensions=["png"], tolerance=6)
def test_sharex_waveplot_ms(y, sr, S_abs):

    # Correct time range ~= 4.6 s or 4600ms
    # Due to shared x_axis, both plots are plotted in 'ms'.
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 1, 1)
    librosa.display.waveplot(y, sr)
    plt.xlabel("")  # hide the x label here, which is not propagated automatically
    plt.subplot(2, 1, 2, sharex=ax)
    librosa.display.specshow(librosa.amplitude_to_db(S_abs, ref=np.max), x_axis="ms")
    plt.xlabel("")  # hide the x label here, which is not propagated automatically
    return plt.gcf()
