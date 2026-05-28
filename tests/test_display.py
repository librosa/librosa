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

import gc
import weakref

from packaging import version

import pytest

import matplotlib

import matplotlib.collections
import matplotlib.pyplot as plt

import librosa
import librosa.display
import numpy as np
from typing import Any, Dict

STYLE = "default"

# Workaround for old freetype builds with our image fixtures
FT_VERSION = version.parse(matplotlib.ft2font.__freetype_version__)
OLD_FT = not (FT_VERSION >= version.parse("2.10"))


@pytest.fixture(scope="module")
def audio():

    __EXAMPLE_FILE = os.path.join("tests", "test_audio.ogg")
    # Force 64-bit here to avoid phase instabilities in display down the road
    y, sr = librosa.load(__EXAMPLE_FILE, dtype=np.float64)
    return y, sr


@pytest.fixture(scope="module")
def y(audio):
    return audio[0]


@pytest.fixture(scope="module")
def sr(audio):
    return audio[1]


@pytest.fixture(scope="module")
def S(y):
    return librosa.stft(y)


@pytest.fixture(scope="module")
def S_abs(S):
    return np.abs(S)


@pytest.fixture(scope="module")
def C(y, sr):
    return np.abs(librosa.cqt(y, sr=sr))


@pytest.fixture(scope="module")
def S_signed(S):
    return np.round(np.abs(S) - np.mean(np.abs(S)), decimals=4)


@pytest.fixture(scope="module")
def rhythm(y, sr):
    return librosa.beat.beat_track(y=y, sr=sr)


@pytest.fixture(scope="module")
def tempo(rhythm):
    return rhythm[0]


@pytest.fixture(scope="module")
def beats(rhythm, C):
    return librosa.util.fix_frames(rhythm[1])


@pytest.fixture(scope="module")
def beat_t(beats, sr):
    return librosa.frames_to_time(beats, sr=sr)


@pytest.fixture(scope="module")
def Csync(C, beats):
    return librosa.util.sync(C, beats, aggregate=np.median)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_unknown_time_unit(y):
    times = np.arange(len(y))
    plt.figure()
    ax = plt.gca()
    ax.plot(times, y)
    ax.xaxis.set_major_formatter(
        librosa.display.TimeFormatter(unit="an unsupported time unit")
    )
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["complex"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_complex_input(S):
    plt.figure()
    with pytest.warns(UserWarning, match="Trying to display complex"):
        librosa.display.specshow(S)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["abs"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_abs_input(S_abs):
    plt.figure()
    librosa.display.specshow(S_abs)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["cqt_note"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_cqt_note(C):
    plt.figure()
    librosa.display.specshow(C, y_axis="cqt_note")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["fft_note"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_fft_note(S_abs):
    plt.figure()
    librosa.display.specshow(S_abs, y_axis="fft_note")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["cqt_hz"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_cqt_hz(C):
    plt.figure()
    librosa.display.specshow(C, y_axis="cqt_hz")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["tempo"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_tempo(y, sr):
    T = librosa.feature.tempogram(y=y, sr=sr, win_length=64)

    plt.figure()
    librosa.display.specshow(T, y_axis="tempo", cmap="magma")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["fourier_tempo"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
@pytest.mark.filterwarnings(
    "ignore:n_fft=.*is too large"
)  # our test signal is short, but this is fine here
def test_fourier_tempo(y, sr):
    T = librosa.feature.fourier_tempogram(y=y, sr=sr, win_length=64)

    plt.figure()
    librosa.display.specshow(np.abs(T), y_axis="fourier_tempo", cmap="magma")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["tonnetz"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_tonnetz(C):
    plt.figure()
    chroma = librosa.feature.chroma_cqt(C=C)
    ton = librosa.feature.tonnetz(chroma=chroma)
    librosa.display.specshow(ton, y_axis="tonnetz")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["chroma"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_chroma(S_abs, sr):
    plt.figure()
    plt.subplot(3, 1, 1)
    chr1 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr)
    librosa.display.specshow(chr1, y_axis="chroma")

    plt.subplot(3, 1, 2)
    chr2 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr, n_chroma=2 * 12)
    librosa.display.specshow(chr2, y_axis="chroma", bins_per_octave=2 * 12)

    plt.subplot(3, 1, 3)
    chr3 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr, n_chroma=3 * 12)
    librosa.display.specshow(chr3, y_axis="chroma", bins_per_octave=3 * 12)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["chroma_svara"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_chroma_svara(C, sr):

    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, threshold=0.9)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(10, 10))

    # Hindustani, no thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax1)

    # Hindustani, kafi thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax2, thaat="kafi")

    # Carnatic, mela 22
    librosa.display.specshow(chroma, y_axis="chroma_c", Sa=5, ax=ax3, mela=22)

    # Carnatic, mela 1
    librosa.display.specshow(chroma, y_axis="chroma_c", Sa=7, ax=ax4, mela=1)

    ax1.label_outer()
    ax2.label_outer()
    ax3.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["double_chroma"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_double_chroma(S_abs, sr):
    plt.figure()

    chr1 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr)
    chr1 = np.vstack((chr1, chr1))
    librosa.display.specshow(chr1, y_axis="chroma", bins_per_octave=12)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_mel"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_x_mel(S_abs):
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M.T, x_axis="mel")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["y_mel"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_y_mel(S_abs):
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M, y_axis="mel")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["y_mel_bounded"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_y_mel_bounded(S_abs):
    plt.figure()

    fmin, fmax = 110, 880
    M = librosa.feature.melspectrogram(S=S_abs**2, fmin=fmin, fmax=fmax)
    librosa.display.specshow(M, y_axis="mel", fmin=fmin, fmax=fmax)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_none_y_linear"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_none_yaxis_linear(S_abs, S_signed):
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs, y_axis="linear")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed, y_axis="fft")

    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_ext_axes"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_specshow_ext_axes(S_abs):
    plt.figure()
    ax_left = plt.subplot(1, 2, 1)
    ax_right = plt.subplot(1, 2, 2)

    # implicitly ax_right
    librosa.display.specshow(S_abs, cmap="gray")
    librosa.display.specshow(S_abs, cmap="magma", ax=ax_left)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_none_y_log"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_none_yaxis_log(S_abs, S_signed):
    plt.figure()

    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs, y_axis="log")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed, y_axis="log")

    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_linear_y_none"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_linear_yaxis_none(S_abs, S_signed):
    plt.figure()

    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="linear")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="fft")

    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_log_y_none"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_log_yaxis_none(S_abs, S_signed):

    plt.figure()

    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="log")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="log")

    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_time_y_none"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_time_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="time")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_none_y_time"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_none_yaxis_time(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="time")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_frames_y_none"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_frames_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="frames")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_none_y_frames"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_none_yaxis_frames(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="frames")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_lag_y_none"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_lag_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="lag")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_none_y_lag"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_time_yaxis_lag(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="lag")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["time_scales_auto"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
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


@pytest.mark.mpl_image_compare(
    baseline_images=["time_unit"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_time_unit(S_abs, sr):

    # sr = 22050, hop_length = 512, S.shape[1] = 198
    # 197 * 512 / 22050 ~= 4.6s
    plt.figure(figsize=(9, 10))
    plt.subplot(5, 1, 1)
    # time scale auto
    librosa.display.specshow(S_abs, sr=sr, x_axis="time")

    plt.subplot(5, 1, 2)
    # time unit fixed to 'h'
    librosa.display.specshow(S_abs, sr=sr, x_axis="h")

    plt.subplot(5, 1, 3)
    # time unit fixed to 'h'
    librosa.display.specshow(S_abs, sr=sr, x_axis="m")

    plt.subplot(5, 1, 4)
    # time unit fixed to 's'
    librosa.display.specshow(S_abs, sr=sr, x_axis="s")

    plt.subplot(5, 1, 5)
    # time unit fixed to 'ms'
    librosa.display.specshow(S_abs, sr=sr, x_axis="ms")

    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["time_unit_lag"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_time_unit_lag(S_abs, sr):

    plt.figure(figsize=(9, 10))
    plt.subplot(5, 1, 1)
    # time scale auto in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag")

    plt.subplot(5, 1, 2)
    # time unit fixed to 'h' in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag_h")

    plt.subplot(5, 1, 3)
    # time unit fixed to 'm' in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag_m")

    plt.subplot(5, 1, 4)
    # time unit fixed to 's' in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag_s")

    plt.subplot(5, 1, 5)
    # time unit fixed to 'ms' in lag mode
    librosa.display.specshow(S_abs, sr=sr, x_axis="lag_ms")

    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_mono(y, sr):

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono_trans"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_mono_trans(y, sr):

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, transpose=True)
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono_zoom"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_mono_zoom(y, sr):

    fig, ax = plt.subplots()
    out = librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2)
    # Zoom into 1/8 of a second, make sure it's out of the initial viewport
    ax.set(xlim=[1, 1.125])
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono_zoom_trans"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_mono_zoom_trans(y, sr):

    fig, ax = plt.subplots()
    out = librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2, transpose=True)
    # Zoom into 1/8 of a second, make sure it's out of the initial viewport
    ax.set(ylim=[1, 1.125])
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono_zoom_out"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_mono_zoom_out(y, sr):

    fig, ax = plt.subplots()
    out = librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2)
    # Zoom into 1/8 of a second, make sure it's out of the initial viewport
    ax.set(xlim=[1, 1.125])
    # Zoom back out to get an envelope view again
    ax.set(xlim=[0, 1])
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_ext_axes"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_ext_axes(y):
    plt.figure()
    ax_left = plt.subplot(1, 2, 1)
    ax_right = plt.subplot(1, 2, 2)

    # implicitly ax_right
    librosa.display.waveshow(y, color="blue")
    librosa.display.waveshow(y, color="red", ax=ax_left)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_inverted"],
    extensions=["png"],
    tolerance=3,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_inverted(y, sr):

    fig, ax = plt.subplots(nrows=3, sharex=True)
    # Original waveshow
    librosa.display.waveshow(y, sr=sr, ax=ax[0], invert=False, label="Regular")
    ax[0].legend(loc="upper right")

    # Inverted with default (axes face) color
    librosa.display.waveshow(
        y, sr=sr, ax=ax[1], invert=True, invert_color=None, label="Inverted"
    )
    ax[1].legend(loc="upper right")

    # Inverted with custom color
    librosa.display.waveshow(
        y, sr=sr, ax=ax[2], invert=True, invert_color="#2d2d2d", label="Inverted custom"
    )
    ax[2].legend(loc="upper right")

    for axi in ax.flat:
        axi.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_stereo"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_waveshow_stereo(y, sr):

    ys = librosa.util.stack([y, 2 * y])

    plt.figure()
    librosa.display.waveshow(ys, sr=sr)
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_unknown_wavaxis(y, sr):

    plt.figure()
    librosa.display.waveshow(y, sr=sr, axis="something not in the axis map")
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_waveshow_unknown_wavaxis(y, sr):

    plt.figure()
    librosa.display.waveshow(y, sr=sr, axis="something not in the axis map")
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_waveshow_bad_maxpoints(y, sr):
    plt.figure()
    librosa.display.waveshow(y, sr=sr, max_points=0)
    return plt.gcf()


def test_waveshow_adaptor_survives_gc_single(y, sr):
    """Regression test for #1970: single waveshow adaptor should survive GC."""
    fig, ax = plt.subplots()

    librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2)

    gc.collect()

    ax.set_xlim(0, 0.025)
    fig.canvas.draw()

    # Data lines only (not ticks)
    visible_data_lines = [l for l in ax.lines if l.get_visible()]

    # Envelope(s) are PolyCollections in ax.collections
    polys = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PolyCollection)
    ]

    assert (
        len(visible_data_lines) >= 1
    ), "Sample/step artist should be visible after zoom"
    assert all(
        not p.get_visible() for p in polys
    ), "Envelope should be hidden after zoom"

    plt.close(fig)


def test_waveshow_adaptor_survives_gc_multi(y, sr):
    """Regression test for #1970: multiple waveshow adaptors should survive GC."""
    fig, ax = plt.subplots()

    librosa.display.waveshow(y, sr=sr, ax=ax, color="blue", max_points=sr // 2)
    librosa.display.waveshow(
        y, sr=sr, ax=ax, color="red", alpha=0.5, zorder=-1, max_points=sr // 2
    )

    gc.collect()

    ax.set_xlim(0, 0.025)
    fig.canvas.draw()

    visible_data_lines = [l for l in ax.lines if l.get_visible()]
    polys = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PolyCollection)
    ]

    assert (
        len(visible_data_lines) >= 2
    ), "Both sample/step artists should be visible after zoom"
    assert all(
        not p.get_visible() for p in polys
    ), "All envelopes should be hidden after zoom"

    plt.close(fig)


def test_waveshow_registry_cleanup_on_axes_gc():
    """Test that the adaptor registry uses WeakKeyDictionary correctly.

    This verifies the fix for #1970: adaptors are stored in a WeakKeyDictionary
    keyed by Axes, so they will be cleaned up when the Axes is garbage collected.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax_id = id(ax)

    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    librosa.display.waveshow(y, sr=22050, ax=ax)

    # Our Axes should be in the registry
    before_keys = set(map(id, librosa.display._WAVESHOW_ADAPTORS.keys()))
    assert ax_id in before_keys, "Axes should be in registry after waveshow"

    # Verify the registry is a WeakKeyDictionary (semantics test)
    assert isinstance(librosa.display._WAVESHOW_ADAPTORS, weakref.WeakKeyDictionary)

    # Verify an adaptor is registered for this axes
    assert ax in librosa.display._WAVESHOW_ADAPTORS
    assert len(librosa.display._WAVESHOW_ADAPTORS[ax]) >= 1


def test_waveshow_update_gc_guard():

    fig, ax = plt.subplots()
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050, endpoint=False))
    adaptor = librosa.display.waveshow(y, sr=22050, ax=ax)

    # Simulate a dead weakref
    class _Tmp:
        pass

    tmp = _Tmp()
    adaptor._steps_ref = weakref.ref(tmp)  # type: ignore[arg-type]
    del tmp
    gc.collect()

    assert adaptor.steps is None

    # Should return early without error (exercises the GC guard)
    adaptor.update(ax)

    plt.close(fig)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("axis", ["x_axis", "y_axis"])
def test_unknown_axis(S_abs, axis: str):

    kwargs: Dict[str, Any] = {axis: "something not in the axis map"}
    plt.figure()
    librosa.display.specshow(S_abs, **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        np.arange(1, 10.0),  # strictly positive
        -np.arange(1, 10.0),  # strictly negative
        np.arange(-3, 4.0),  # signed,
        np.arange(2, dtype=bool),
    ],
)  # binary
def test_infer_cmap_robust(data):

    cmap1 = librosa.display.infer_cmap(data, robust=False)
    cmap2 = librosa.display.infer_cmap(data, robust=True)

    if isinstance(cmap1, matplotlib.colors.ListedColormap) and isinstance(
        cmap2, matplotlib.colors.ListedColormap
    ):
        assert np.allclose(cmap1.colors, cmap2.colors)
    elif isinstance(cmap1, matplotlib.colors.LinearSegmentedColormap) and isinstance(
        cmap2, matplotlib.colors.LinearSegmentedColormap
    ):
        assert cmap1.name == cmap2.name
    else:
        assert cmap1 == cmap2
        assert type(cmap1) is type(cmap2)


@pytest.mark.mpl_image_compare(
    baseline_images=["coords"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_coords(Csync, beat_t):

    plt.figure()
    librosa.display.specshow(Csync, x_coords=beat_t, x_axis="time", y_axis="cqt_note")
    return plt.gcf()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_bad_coords(S_abs):

    librosa.display.specshow(S_abs, x_coords=np.arange(S_abs.shape[1] // 2))
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["sharex_specshow_ms"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_sharex_specshow_ms(S_abs, y, sr):

    # Correct time range ~= 4.6 s or 4600ms
    # Due to shared x_axis, both plots are plotted in 's'.
    fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)
    librosa.display.specshow(
        librosa.amplitude_to_db(S_abs, ref=np.max), x_axis="time", ax=ax
    )
    ax.set(xlabel="")  # hide the x label here, which is not propagated automatically
    ax2.margins(x=0)
    librosa.display.waveshow(y, sr=sr, axis="ms", ax=ax2)
    ax2.set(xlabel="")  # hide the x label here, which is not propagated automatically
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["sharex_waveplot_ms"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_sharex_waveplot_ms(y, sr, S_abs):

    # Correct time range ~= 4.6 s or 4600ms
    # Due to shared x_axis, both plots are plotted in 'ms'.
    fig, (ax, ax2) = plt.subplots(sharex=True, nrows=2, figsize=(8, 8))
    ax.margins(x=0)
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(xlabel="")  # hide the x label here, which is not propagated automatically
    ax2.margins(x=0)
    librosa.display.specshow(
        librosa.amplitude_to_db(S_abs, ref=np.max), x_axis="ms", ax=ax2
    )
    ax2.set(xlabel="")  # hide the x label here, which is not propagated automatically
    return fig


@pytest.mark.parametrize("format_str", ["cqt_hz", "cqt_note", "vqt_hz"])
def test_axis_bound_warning(format_str):

    with pytest.warns(UserWarning):
        # set sr=22050
        # fmin= 11025
        # 72 bins
        # 12 bins per octave

        librosa.display.specshow(
            np.zeros((72, 3)),
            y_axis=format_str,
            fmin=11025,
            sr=22050,
            bins_per_octave=12,
            intervals="ji3",
        )


@pytest.mark.mpl_image_compare(
    baseline_images=["cqt_svara"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_display_cqt_svara(C, sr):

    Camp = librosa.amplitude_to_db(C, ref=np.max)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        nrows=5, sharex=True, figsize=(10, 10)
    )

    librosa.display.specshow(Camp, y_axis="cqt_svara", Sa=261, ax=ax1)
    librosa.display.specshow(Camp, y_axis="cqt_svara", Sa=440, ax=ax2)
    librosa.display.specshow(Camp, y_axis="cqt_svara", Sa=261, ax=ax3)
    librosa.display.specshow(Camp, y_axis="cqt_svara", Sa=261, mela=1, ax=ax4)
    librosa.display.specshow(Camp, y_axis="cqt_svara", Sa=261, mela=1, ax=ax5)

    ax3.set_ylim([440, 880])
    ax5.set_ylim([440, 880])
    ax1.label_outer()
    ax2.label_outer()
    ax3.label_outer()
    ax4.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["fft_svara"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_display_fft_svara(S_abs, sr):

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        nrows=5, sharex=True, figsize=(10, 10)
    )

    librosa.display.specshow(S_abs, y_axis="fft_svara", Sa=261, ax=ax1)
    librosa.display.specshow(S_abs, y_axis="fft_svara", Sa=440, ax=ax2)
    librosa.display.specshow(S_abs, y_axis="fft_svara", Sa=261, ax=ax3)
    librosa.display.specshow(S_abs, y_axis="fft_svara", Sa=261, mela=1, ax=ax4)
    librosa.display.specshow(S_abs, y_axis="fft_svara", Sa=261, mela=1, ax=ax5)

    ax3.set_ylim([440, 880])
    ax5.set_ylim([440, 880])
    ax1.label_outer()
    ax2.label_outer()
    ax3.label_outer()
    ax4.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["nfft_odd"], extensions=["png"], tolerance=6, style=STYLE
)
def test_display_fft_odd():

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 10))
    N1 = 8
    N2 = N1 + 1
    sr = N1
    S = np.tile(np.arange(1 + N1 // 2), (20, 1)).T
    # Use the default inference
    librosa.display.specshow(S, x_axis="time", y_axis="fft", sr=sr, ax=ax1)

    # Force it to match exactly
    librosa.display.specshow(S, x_axis="time", y_axis="fft", sr=sr, n_fft=N1, ax=ax2)

    # Override with an odd number
    librosa.display.specshow(S, x_axis="time", y_axis="fft", sr=sr, n_fft=N2, ax=ax3)

    ax1.label_outer()
    ax2.label_outer()
    ax3.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["nfft_odd_ftempo"], extensions=["png"], tolerance=6, style=STYLE
)
def test_display_fourier_tempo_odd():

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 10))
    N1 = 8
    N2 = N1 + 1
    sr = N1
    S = np.tile(np.arange(1 + N1 // 2), (20, 1)).T
    # Use the default inference
    librosa.display.specshow(S, y_axis="fourier_tempo", sr=sr, ax=ax1)

    # Force it to match exactly
    librosa.display.specshow(S, y_axis="fourier_tempo", sr=sr, win_length=N1, ax=ax2)

    # Override with an odd number
    librosa.display.specshow(S, y_axis="fourier_tempo", sr=sr, win_length=N2, ax=ax3)

    ax1.label_outer()
    ax2.label_outer()
    ax3.label_outer()

    return fig


@pytest.mark.parametrize(
    "x_axis,y_axis,xlim,ylim,out",
    [
        (None, None, (0.0, 1.0), (0.0, 1.0), False),
        ("time", "linear", (0.0, 1.0), (0.0, 1.0), False),
        ("time", "time", (0.0, 1.0), (0.0, 2.0), False),
        ("chroma", "chroma", (0.0, 1.0), (0.0, 1.0), True),
        ("s", "ms", (0.0, 1.0), (0.0, 1.0), True),
    ],
)
def test_same_axes(x_axis, y_axis, xlim, ylim, out):
    assert librosa.display.__same_axes(x_axis, y_axis, xlim, ylim) == out


def test_auto_aspect():

    fig, ax = plt.subplots(nrows=5)

    # Ensure auto aspect by default
    for axi in ax:
        axi.set(aspect="auto")

    X = np.zeros((12, 12))

    # Different axes with incompatible types should retain auto scaling
    librosa.display.specshow(X, x_axis="chroma", y_axis="time", ax=ax[0])
    assert ax[0].get_aspect() == "auto"

    # Same axes and auto_aspect=True should force equal scaling
    librosa.display.specshow(X, x_axis="chroma", y_axis="chroma", ax=ax[1])
    assert ax[1].get_aspect() == 1.0

    # Same axes and auto_aspect=False should retain auto scaling
    librosa.display.specshow(
        X, x_axis="chroma", y_axis="chroma", auto_aspect=False, ax=ax[2]
    )
    assert ax[2].get_aspect() == "auto"

    # Different extents with auto_aspect=True should retain auto scaling
    librosa.display.specshow(
        X[:2, :], x_axis="chroma", y_axis="chroma", auto_aspect=True, ax=ax[3]
    )
    assert ax[3].get_aspect() == "auto"

    # different axes with compatible types and auto_aspect=True should force equal scaling
    librosa.display.specshow(X, x_axis="time", y_axis="ms", ax=ax[4])
    assert ax[4].get_aspect() == 1.0


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_unicode_true"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_specshow_unicode_true(C, sr):

    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, threshold=0.9)

    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

    # Hindustani, no thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax[0], unicode=True)

    # Hindustani, kafi thaat
    librosa.display.specshow(
        chroma, y_axis="chroma_h", Sa=5, ax=ax[1], thaat="kafi", unicode=True
    )

    # Carnatic, mela 22
    librosa.display.specshow(
        chroma, y_axis="chroma_c", Sa=5, ax=ax[2], mela=22, unicode=True
    )

    # Carnatic, mela 1
    librosa.display.specshow(
        chroma, y_axis="chroma_c", Sa=7, ax=ax[3], mela=1, unicode=True
    )

    # Pitches
    librosa.display.specshow(
        chroma, y_axis="chroma", ax=ax[4], key="Eb:maj", unicode=True
    )

    for axi in ax:
        axi.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_unicode_false"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_specshow_unicode_false(C, sr):

    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, threshold=0.9)

    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

    # Hindustani, no thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax[0], unicode=False)

    # Hindustani, kafi thaat
    librosa.display.specshow(
        chroma, y_axis="chroma_h", Sa=5, ax=ax[1], thaat="kafi", unicode=False
    )

    # Carnatic, mela 22
    librosa.display.specshow(
        chroma, y_axis="chroma_c", Sa=5, ax=ax[2], mela=22, unicode=False
    )

    # Carnatic, mela 1
    librosa.display.specshow(
        chroma, y_axis="chroma_c", Sa=7, ax=ax[3], mela=1, unicode=False
    )

    librosa.display.specshow(
        chroma, y_axis="chroma", ax=ax[4], key="Eb:maj", unicode=False
    )

    for axi in ax:
        axi.label_outer()

    return fig


def test_waveshow_disconnect(y, sr):
    fig, ax = plt.subplots()
    ad = librosa.display.waveshow(y=y, sr=sr, ax=ax)

    assert ad.envelope is not None
    assert ad.steps is not None

    # By default, envelope should be visible and steps should not
    assert ad.envelope.get_visible() and not ad.steps.get_visible()

    # Zoom in to a 0.25 second range
    ax.set(xlim=[0, 0.25])

    # Steps should be visible but not envelope
    assert ad.steps.get_visible() and not ad.envelope.get_visible()

    # Zoom back out
    ax.set(xlim=[0, 4])
    assert ad.envelope.get_visible() and not ad.steps.get_visible()

    # Disconnect
    ad.disconnect()

    # Zoom back in to a 0.25 second range
    ax.set(xlim=[0, 0.25])

    # Envelope should now still be visible
    assert ad.envelope.get_visible() and not ad.steps.get_visible()


def test_waveshow_deladaptor(y, sr):
    fig, ax = plt.subplots()
    ad = librosa.display.waveshow(y=y, sr=sr, ax=ax)

    envelope, steps = ad.envelope, ad.steps
    # These should be live for the rest of the tests to function
    assert envelope is not None
    assert steps is not None
    # By default, envelope should be visible and steps should not
    assert envelope.get_visible() and not steps.get_visible()

    # Zoom in to a 0.25 second range
    ax.set(xlim=[0, 0.25])

    # Steps should be visible but not envelope
    assert steps.get_visible() and not envelope.get_visible()

    # Zoom back out
    ax.set(xlim=[0, 4])
    assert envelope.get_visible() and not steps.get_visible()

    # Disconnect
    ad.disconnect(strict=True)

    # Zoom back in to a 0.25 second range
    ax.set(xlim=[0, 0.25])

    # Envelope should now still be visible
    assert envelope.get_visible() and not steps.get_visible()


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_vqt"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_specshow_vqt(C):

    fig, ax = plt.subplots(nrows=4, figsize=(12, 10))

    librosa.display.specshow(C, y_axis="vqt_hz", intervals="ji5", ax=ax[0])
    librosa.display.specshow(C, y_axis="vqt_note", intervals="ji5", ax=ax[1])
    librosa.display.specshow(C, y_axis="vqt_fjs", intervals="ji5", ax=ax[2])
    librosa.display.specshow(
        C, y_axis="vqt_fjs", intervals="ji5", ax=ax[3], unicode=False
    )

    for _ax in ax:
        _ax.set(ylim=[55, 165])
    return fig


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_chromafjs_badintervals():
    formatter = librosa.display.ChromaFJSFormatter(intervals=dict())


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_chromafjs_badbpo():
    formatter = librosa.display.ChromaFJSFormatter(
        intervals="ji3", bins_per_octave=None
    )


@pytest.mark.mpl_image_compare(
    baseline_images=["chroma_fjs"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_specshow_chromafjs(C, sr):

    # This isn't a VQT chroma, but that's not important here
    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, threshold=0.9)

    intervals = librosa.plimit_intervals(primes=[3, 5])

    fig, ax = plt.subplots(nrows=2, figsize=(12, 8))

    librosa.display.specshow(chroma, y_axis="chroma_fjs", intervals="ji5", ax=ax[0])
    librosa.display.specshow(chroma, y_axis="chroma_fjs", intervals=intervals, ax=ax[1])

    return fig


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_vqt_hz_nointervals(C, sr):
    librosa.display.specshow(C, sr=sr, y_axis="vqt_hz")


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("vscale", ["dBFS[1]", "dBFS[power,1]"])
def test_parse_vscale_dbfs_ref(vscale):
    # This should raise an error because a reference value is
    # not allowed with dBFS
    librosa.display.__parse_vscale(vscale)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("vscale", ["bad string", "dB[gibberish]", "dBFS[gibberish]"])
def test_parse_vscale_fail(vscale):
    librosa.display.__parse_vscale(vscale)


@pytest.mark.parametrize(
    "vscale, mode, scale_type, ref",
    [
        ("dBFS", "dBFS", "amplitude", "max"),
        ("dBFS[power]", "dBFS", "power", "max"),
        ("dB", "dB", "amplitude", None),
        ("dB[2]", "dB", "amplitude", 2),
        ("dB[1e-1]", "dB", "amplitude", 0.1),
        ("dB[power]", "dB", "power", None),
        ("dB[power,2]", "dB", "power", 2),
        ("dB[power,1e-1]", "dB", "power", 0.1),
    ],
)
def test_parse_vscale(vscale, mode, scale_type, ref):
    assert librosa.display.__parse_vscale(vscale) == (mode, scale_type, ref)


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_vscale"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_specshow_vscale(S):
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(12, 12))

    # first column is dB, dBFS, dB with ref value
    i1 = librosa.display.specshow(
        S, vscale="dB", y_axis="log", x_axis="time", ax=ax[0, 0]
    )
    fig.colorbar(i1, ax=ax[0, 0])
    ax[0, 0].set_title("dB")
    i2 = librosa.display.specshow(
        S, vscale="dBFS", y_axis="log", x_axis="time", ax=ax[1, 0]
    )
    fig.colorbar(i2, ax=ax[1, 0])
    ax[1, 0].set_title("dBFS")
    i3 = librosa.display.specshow(
        S, vscale="dB[1e-2]", y_axis="log", x_axis="time", ax=ax[2, 0]
    )
    fig.colorbar(i3, ax=ax[2, 0])
    ax[2, 0].set_title("dB, ref=1e-2")

    # second column is dB[power], dBFS[power], dB[power] with ref value
    i4 = librosa.display.specshow(
        S, vscale="dB[power]", y_axis="log", x_axis="time", ax=ax[0, 1]
    )
    fig.colorbar(i4, ax=ax[0, 1])
    ax[0, 1].set_title("dB power")
    i5 = librosa.display.specshow(
        S, vscale="dBFS[power]", y_axis="log", x_axis="time", ax=ax[1, 1]
    )
    fig.colorbar(i5, ax=ax[1, 1])
    ax[1, 1].set_title("dBFS power")
    i6 = librosa.display.specshow(
        S, vscale="dB[power,1e-2]", y_axis="log", x_axis="time", ax=ax[2, 1]
    )
    fig.colorbar(i6, ax=ax[2, 1])
    ax[2, 1].set_title("dB power, ref=1e-2")

    for _ax in ax.flat:
        _ax.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_vscale_phase"],
    extensions=["png"],
    tolerance=3,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_specshow_vscale_phase():
    # Create a chirp
    sr = 22050
    y = librosa.chirp(fmin=110, fmax=880, duration=1, sr=sr, linear=False)
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    alpha = np.abs(S)
    alpha /= np.max(alpha)  # normalize to [0, 1]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=3,
        sharex=False,
        sharey=False,
        gridspec_kw={"hspace": 0.8},
        figsize=(12, 4),
    )

    # phase, phase difference, and phase difference transpose
    # we use alpha channels here to avoid test failures for unstable phase estimates in quiet regions
    i7 = librosa.display.specshow(
        S, vscale="phase", y_axis="log", x_axis="time", ax=ax[0], alpha=alpha
    )
    fig.colorbar(i7, ax=ax[0])
    ax[0].set_title("phase")

    i8 = librosa.display.specshow(
        S, vscale="dphase", y_axis="log", x_axis="time", ax=ax[1], alpha=alpha
    )
    fig.colorbar(i8, ax=ax[1])
    ax[1].set_title("dphase")

    i9 = librosa.display.specshow(
        S.T, vscale="dphase_t", x_axis="log", y_axis="time", ax=ax[2], alpha=alpha.T
    )
    fig.colorbar(i9, ax=ax[2])
    ax[2].set_title("dphase_t")

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["colorbar_db"],
    extensions=["png"],
    tolerance=3,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_colorbar_db(S):
    fig, ax = plt.subplots(nrows=2)
    i1 = librosa.display.specshow(S, vscale="dB", y_axis="log", x_axis="time", ax=ax[0])
    librosa.display.colorbar_db(i1, ax=ax[0])
    i2 = librosa.display.specshow(
        S, vscale="dBFS", y_axis="log", x_axis="time", ax=ax[1]
    )
    librosa.display.colorbar_db(i2, ax=ax[1], label="dBFS")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["colorbar_phase"],
    extensions=["png"],
    tolerance=3,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_colorbar_phase(S):
    alpha = np.abs(S)
    alpha /= np.max(alpha)
    fig, ax = plt.subplots(nrows=2)
    i1 = librosa.display.specshow(
        S, vscale="phase", y_axis="log", x_axis="time", alpha=alpha, ax=ax[0]
    )
    librosa.display.colorbar_phase(i1, ax=ax[0])
    i2 = librosa.display.specshow(
        S, vscale="dphase", y_axis="log", x_axis="time", alpha=alpha, ax=ax[1]
    )
    librosa.display.colorbar_phase(i2, ax=ax[1], label="Δ radians")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["diverging_slopes"],
    extensions=["png"],
    tolerance=5,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_diverging_scales(S_signed):

    # Cases to test:
    # 0. Diverging scale with default cmap_div (auto norm)
    # 1. Diverging scale with explicit cmap override (no norm)
    # 2. Diverging scale with specified cmap_div (auto norm)
    # 3. Inferred diverging scale with default cmap_div and vmin/vmax (auto norm, truncated)
    # 4. Explicit diverging scale with vmin/vmax (no norm, truncated)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharex=True, sharey=True)

    # Diverging scale with default cmap_div (auto norm)
    i1 = librosa.display.specshow(S_signed, y_axis="log", x_axis="time", ax=ax[0, 0])
    ax[0, 0].set_title("Default cmap_div (auto norm)")
    fig.colorbar(i1, ax=ax[0, 0])

    # Diverging scale with explicit cmap override (no norm)
    i2 = librosa.display.specshow(
        S_signed,
        y_axis="log",
        x_axis="time",
        ax=ax[0, 1],
        cmap="PuOr_r",
    )
    ax[0, 1].set_title("Explicit cmap override (no norm)")
    fig.colorbar(i2, ax=ax[0, 1])

    # Diverging scale with specified cmap_div (auto norm)
    i3 = librosa.display.specshow(
        S_signed, y_axis="log", x_axis="time", ax=ax[1, 0], cmap_div="Spectral_r"
    )
    ax[1, 0].set_title("Specified cmap_div (auto norm)")
    fig.colorbar(i3, ax=ax[1, 0])

    # Inferred diverging scale with default cmap_div and vmin/vmax (auto norm, truncated)
    vmin = -10
    vmax = 30
    i4 = librosa.display.specshow(
        S_signed,
        y_axis="log",
        x_axis="time",
        ax=ax[2, 0],
        vmin=vmin,
        vmax=vmax,
        cmap_div=matplotlib.colormaps["coolwarm"],
    )
    ax[2, 0].set_title("Inferred cmap_div with vmin/vmax (auto norm, truncated)")
    fig.colorbar(i4, ax=ax[2, 0])

    # Explicit diverging scale with vmin/vmax (no norm, truncated)
    i5 = librosa.display.specshow(
        S_signed,
        y_axis="log",
        x_axis="time",
        ax=ax[1, 1],
        vmin=vmin,
        vmax=vmax,
        cmap="Spectral_r",
    )
    ax[1, 1].set_title("Explicit cmap_div with vmin/vmax (no norm, truncated)")
    fig.colorbar(i5, ax=ax[1, 1])

    # Hide the last axis
    ax[2, 1].axis("off")

    for axi in ax.flat:
        axi.label_outer()
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["oct3"],
    extensions=["png"],
    tolerance=5,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_oct3(S_abs, C):

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # STFT, Mel
    # CQT, VQT

    librosa.display.specshow(S_abs, vscale="dBFS", y_axis="log_oct3", ax=ax[0, 0])

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M, vscale="dBFS[power]", y_axis="mel_oct3", ax=ax[0, 1])

    librosa.display.specshow(C, vscale="dBFS", y_axis="cqt_oct3", ax=ax[1, 0])

    librosa.display.specshow(
        C, vscale="dBFS", y_axis="vqt_oct3", ax=ax[1, 1], intervals="equal"
    )

    # Put the ticks on the right just to reduce crowding
    ax[0, 1].yaxis.tick_right()
    ax[1, 1].yaxis.tick_right()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["wavebars"],
    extensions=["png"],
    tolerance=5,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_wavebars(y, sr):

    fig, ax = plt.subplots(nrows=6, layout="tight", figsize=(8, 12))

    librosa.display.wavebars(y, sr=sr, ax=ax[0], label="100")
    librosa.display.wavebars(
        y, sr=sr, color="C1", n_bars=150, rounding_ratio=0, ax=ax[1], label="150 square"
    )
    librosa.display.wavebars(
        y, sr=sr, color="C2", n_bars=50, ax=ax[2], invert=True, label="50 inverted"
    )
    librosa.display.wavebars(
        y, sr=sr, color="C3", gap_ratio=0, ax=ax[3], label="no gap"
    )
    librosa.display.wavebars(
        y,
        sr=sr,
        color="C4",
        offset=30,
        rounding_ratio=0.3,
        invert=True,
        invert_color="#2d2d2d",
        ax=ax[4],
        label="offset 30, invert dark",
    )
    librosa.display.wavebars(y, sr=sr, ax=ax[5], color="C5", axis="ms", label="time_ms")

    for axi in ax.flat:
        axi.legend()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["wavebars_transpose"],
    extensions=["png"],
    tolerance=5,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_wavebars_transpose(y, sr):

    fig, ax = plt.subplots(ncols=6, layout="tight", figsize=(12, 8))

    librosa.display.wavebars(y, sr=sr, ax=ax[0], label="100", transpose=True)
    librosa.display.wavebars(
        y,
        sr=sr,
        color="C1",
        n_bars=150,
        rounding_ratio=0,
        transpose=True,
        ax=ax[1],
        label="150 square",
    )
    librosa.display.wavebars(
        y,
        sr=sr,
        color="C2",
        n_bars=50,
        transpose=True,
        ax=ax[2],
        invert=True,
        label="50 inverted",
    )
    librosa.display.wavebars(
        y, sr=sr, color="C3", gap_ratio=0, transpose=True, ax=ax[3], label="no gap"
    )
    librosa.display.wavebars(
        y,
        sr=sr,
        color="C4",
        offset=30,
        transpose=True,
        rounding_ratio=0.3,
        invert=True,
        invert_color="#2d2d2d",
        ax=ax[4],
        label="offset 30, invert dark",
    )
    librosa.display.wavebars(
        y, sr=sr, ax=ax[5], color="C5", axis="ms", transpose=True, label="time_ms"
    )

    for axi in ax.flat:
        axi.legend()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["wavef0"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_wavef0(y, sr):
    fig, ax = plt.subplots(nrows=2, figsize=(8, 6))

    f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")),
                            fmax=float(librosa.note_to_hz("C7")), sr=sr)
    # Waveform with f0
    librosa.display.wavef0(y, sr=sr, f0=f0, ax=ax[0], label='waveshow')
    ax[0].legend(loc='lower right')

    # Waveform with f0 and pitch
    librosa.display.wavef0(y, sr=sr, f0=f0, method='wavebars', freq_axis='cqt_hz',
                           color='C1', ax=ax[1], label='wavebars')
    ax[1].legend(loc='lower right')
    return fig


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_legend_for_axes_no_axes():
    fig = plt.figure()
    librosa.display.legend_for_axes(fig=fig)


def test_legend_for_axes_current():
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(np.arange(4), label="line")

    leg = librosa.display.legend_for_axes()
    assert leg is not None
    assert leg.figure is fig


def test_legend_for_axes_scalar():
    fig, ax = plt.subplots()
    ax.plot(np.arange(4), label="line")
    leg = librosa.display.legend_for_axes(axes=ax)
    assert leg is not None
    assert leg.figure is fig


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_legend_for_axes_toomanydimensions():
    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax_stack = np.stack([ax, ax], axis=0)
    librosa.display.legend_for_axes(axes=ax_stack)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_legend_for_axes_mismatched_figures():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    librosa.display.legend_for_axes([ax1, ax2])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_legend_for_axes_bad_loc():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")

    librosa.display.legend_for_axes([ax], loc="center")


def test_legend_for_axes_explicit_bbox():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")

    leg = librosa.display.legend_for_axes(
        [ax], loc="center", bbox_to_anchor=(0.5, 0.5, 0.2, 0.2)
    )

    assert leg is not None


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_right"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_right():
    fig, axes = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(right=0.8)
    librosa.display.legend_for_axes(axes=axes, loc="center left")

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["wavef0_transpose"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_wavef0_transpose(y, sr):
    fig, ax = plt.subplots(ncols=2, figsize=(6, 8))

    f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")),
                            fmax=float(librosa.note_to_hz("C7")), sr=sr)
    # Waveform with f0
    librosa.display.wavef0(y, sr=sr, f0=f0, ax=ax[0], label='waveshow',
                           transpose=True)
    ax[0].legend(loc='lower right')

    # Waveform with f0 and pitch
    librosa.display.wavef0(y, sr=sr, f0=f0, method='wavebars', freq_axis='cqt_hz',
                           color='C1', ax=ax[1], label='wavebars',
                           transpose=True)
    ax[1].legend(loc='lower right')
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_left"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_left():
    fig, axes = plt.subplots(nrows=2, figsize=(8, 4), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(left=0.25)
    librosa.display.legend_for_axes(axes=axes, loc="center right")

    return fig


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_wavef0_bad_method(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")),
                            fmax=float(librosa.note_to_hz("C7")), sr=sr)
    librosa.display.wavef0(y, f0=f0, sr=sr, method='bad_method')


@pytest.mark.parametrize(
    "transpose,values",
    [
        (
            False,
            np.array([[0.0, -12.0], [1.0, 0.0], [2.0, 7.0], [3.0, 12.0]], dtype=float),
        ),
        (
            True,
            np.array([[-12.0, 0.0], [0.0, 1.0], [7.0, 2.0], [12.0, 3.0]], dtype=float),
        ),
    ],
)
def test_transformf0_roundtrip(transpose, values):
    f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=float)

    trans = librosa.display.Transformf0(
        f0=f0,
        sr=1,
        hop_length=1,
        bins_per_octave=12,
        norm=2.0,
        transpose=transpose,
    )

    forward = trans.transform(values)
    recovered = trans.inverted().transform(forward)

    assert np.allclose(recovered, values, rtol=1e-12, atol=1e-12)


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_above"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_above():
    fig, axes = plt.subplots(ncols=3, figsize=(8, 4), sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(top=0.8)
    librosa.display.legend_for_axes(axes=axes, loc="lower center", ncol=3)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_below"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_below():
    fig, axes = plt.subplots(ncols=3, figsize=(8, 4), sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(bottom=0.25)
    librosa.display.legend_for_axes(axes=axes, loc="upper center", ncol=3)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_1d"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_default_1d():
    fig, axes = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(right=0.8)
    librosa.display.legend_for_axes(axes=axes)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_row"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_default_row():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    axes = np.asarray(axes)[np.newaxis, :]
    fig.subplots_adjust(top=0.8)
    librosa.display.legend_for_axes(axes=axes)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_col"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_default_col():
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    axes = np.asarray(axes)[:, np.newaxis]
    fig.subplots_adjust(right=0.8)
    librosa.display.legend_for_axes(axes=axes)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_grid"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_legend_for_axes_default_grid():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharex=True, sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(right=0.8)
    librosa.display.legend_for_axes(axes=axes)

    return fig


def test_squeeze_shape():
    assert librosa.display._squeeze_shape((3,)) == (3,)
    assert librosa.display._squeeze_shape((1, 3)) == (3,)
    assert librosa.display._squeeze_shape((3, 1)) == (3,)
    assert librosa.display._squeeze_shape((1, 3, 1, 2, 1)) == (3, 2)
    assert librosa.display._squeeze_shape((1, 1, 1)) == ()


@pytest.mark.parametrize(
    "func,dims,badprops",
    [
        ("waveshow", 1, []),
        ("wavebars", 1, []),
        ("specshow", 2, ["color"]),
    ],
)
def test_resolve_multiplot(func, dims, badprops):
    function, dims_out, badprops_out = librosa.display._resolve_multiplot(func)

    assert callable(function)
    assert dims_out == dims
    assert badprops_out == badprops


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_resolve_multiplot_bad():
    librosa.display._resolve_multiplot("bogus")  # type: ignore


@pytest.mark.parametrize(
    "data,dims,orient,axshape,nrows,ncols,multi_input",
    [
        ((np.zeros((3, 10)),), 1, "v", (3,), 3, 1, False),
        ((np.zeros((3, 10)),), 1, "h", (3,), 1, 3, False),
        ((np.zeros((2, 3, 10)),), 1, "v", (2, 3), 2, 3, False),
        ((np.zeros((2, 3, 4, 5)),), 2, "v", (2, 3), 2, 3, False),
        ((np.zeros(10), np.zeros(10), np.zeros(10)), 1, "v", (3,), 3, 1, True),
        ((np.zeros(10), np.zeros(10), np.zeros(10)), 1, "h", (3,), 1, 3, True),
        ((np.zeros((4, 5)), np.zeros((4, 5))), 2, "v", (2,), 2, 1, True),
    ],
)
def test_mp_get_layout(data, dims, orient, axshape, nrows, ncols, multi_input):
    axshape_out, nrows_out, ncols_out, multi_input_out = librosa.display._mp_get_layout(
        data, dims, orient
    )

    assert axshape_out == axshape
    assert nrows_out == nrows
    assert ncols_out == ncols
    assert multi_input_out is multi_input


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mp_get_layout_bad_orient():
    librosa.display._mp_get_layout((np.zeros((3, 10)),), dims=1, orient="q")  # type: ignore


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mp_get_layout_no_data():
    librosa.display._mp_get_layout(tuple(), dims=1, orient="v")


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mp_get_layout_bad_axshape():
    librosa.display._mp_get_layout((np.zeros((2, 3, 4, 10)),), dims=1, orient="v")


@pytest.mark.parametrize(
    "shape,orient,output_shape,axes_shape",
    [
        ((3,), "v", (3,), (3, 1)),
        ((3,), "h", (3,), (1, 3)),
        ((2, 2), "v", (2, 2), (2, 2)),
        ((2, 2), "h", (2, 2), (2, 2)),
    ],
)
def test_mp_setup_axes_create(shape, orient, output_shape, axes_shape):
    if len(shape) == 1:
        if orient == "v":
            nrows, ncols = (shape[0], 1)
        elif orient == "h":
            nrows, ncols = (1, shape[0])
    else:
        nrows, ncols = shape

    fig, axes, out_shape = librosa.display._mp_setup_axes(
        axes=None,
        fig=None,
        nrows=nrows,
        ncols=ncols,
        axshape=shape,
        orient=orient,
        sharex=True,
        sharey=True,
    )

    assert fig is not None
    assert axes.shape == axes_shape
    assert out_shape == output_shape


def test_mp_setup_axes_with_fig():
    fig = plt.figure()

    fig_out, axes, out_shape = librosa.display._mp_setup_axes(
        axes=None,
        fig=fig,
        fig_kw=None,
        nrows=2,
        ncols=1,
        axshape=(2,),
        orient="v",
        sharex=True,
        sharey=True,
    )

    assert fig_out is fig
    assert axes.shape == (2, 1)
    assert out_shape == (2,)


@pytest.mark.parametrize(
    "orient,axes_in_shape,axes_out_shape,output_shape",
    [
        ("v", (3,), (3, 1), (3,)),
        ("h", (3,), (1, 3), (3,)),
        ("v", (2, 2), (2, 2), (2, 2)),
    ],
)
def test_mp_setup_axes_array_input(orient, axes_in_shape, axes_out_shape, output_shape):
    fig, axes_in = (
        plt.subplots(*axes_in_shape)
        if len(axes_in_shape) == 2
        else plt.subplots(axes_in_shape[0])
    )

    fig_out, axes_out, out_shape = librosa.display._mp_setup_axes(
        axes=np.asarray(axes_in),
        fig=None,
        fig_kw=None,
        nrows=axes_out_shape[0],
        ncols=axes_out_shape[1],
        axshape=output_shape,
        orient=orient,
        sharex=True,
        sharey=True,
    )

    assert fig_out is fig
    assert axes_out.shape == axes_out_shape
    assert out_shape == output_shape


def test_mp_setup_axes_scalar_input():
    fig, ax = plt.subplots()

    fig_out, axes_out, out_shape = librosa.display._mp_setup_axes(
        axes=ax,
        fig=None,
        fig_kw=None,
        nrows=1,
        ncols=1,
        axshape=tuple(),
        orient="v",
        sharex=True,
        sharey=True,
    )

    assert fig_out is fig
    assert axes_out.shape == (1, 1)
    assert out_shape == tuple()


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mp_setup_axes_bad_shape():
    fig, axes = plt.subplots(nrows=2)

    librosa.display._mp_setup_axes(
        axes=np.asarray(axes),
        fig=None,
        fig_kw=None,
        nrows=3,
        ncols=1,
        axshape=(3,),
        orient="v",
        sharex=True,
        sharey=True,
    )


def test_mp_setup_axes_single():
    fig, ax = plt.subplots()

    fig_out, axes_out, out_shape = librosa.display._mp_setup_axes(
        axes=ax,
        fig=None,
        fig_kw=None,
        nrows=1,
        ncols=1,
        axshape=tuple(),
        orient="v",
        sharex=True,
        sharey=True,
    )

    assert fig_out is fig
    assert axes_out.shape == (1, 1)
    assert out_shape == tuple()


def test_mp_setup_labels_none():
    labels = librosa.display._mp_setup_labels(None, (2, 2))

    assert labels.shape == (2, 2)
    assert labels.dtype == object
    # Comparison to None here is vectorized, so == is correct, not `is`.
    assert np.all(labels == None)


def test_mp_setup_labels_values():
    labels = librosa.display._mp_setup_labels(["a", "b", "c"], (3,))

    assert np.array_equal(labels, np.asarray(["a", "b", "c"], dtype=object))


@pytest.mark.xfail(raises=ValueError)
def test_mp_setup_labels_bad_shape():
    librosa.display._mp_setup_labels(["a", "b"], (2, 2))


def test_mp_setup_prop_group_none():
    groups = librosa.display._mp_setup_prop_group(None, (2, 2))
    assert np.array_equal(groups, np.array([[0, 1], [2, 3]]))


def test_mp_setup_prop_group_false():
    groups = librosa.display._mp_setup_prop_group(False, (2, 2))
    assert np.array_equal(groups, np.array([[0, 1], [2, 3]]))


def test_mp_setup_prop_group_true():
    groups = librosa.display._mp_setup_prop_group(True, (2, 2))
    assert np.array_equal(groups, np.ones((2, 2), dtype=int))


def test_mp_setup_prop_group_row():
    groups = librosa.display._mp_setup_prop_group("row", (2, 3))
    assert np.array_equal(groups, np.array([[0, 0, 0], [1, 1, 1]]))


def test_mp_setup_prop_group_col():
    groups = librosa.display._mp_setup_prop_group("col", (2, 3))
    assert np.array_equal(groups, np.array([[0, 1, 2], [0, 1, 2]]))


def test_mp_setup_prop_group_custom():
    groups = librosa.display._mp_setup_prop_group([0, 1, 0, 1], (2, 2))
    assert np.array_equal(groups, np.array([[0, 1], [0, 1]]))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mp_setup_prop_group_bad_shape():
    librosa.display._mp_setup_prop_group([0, 1, 2], (2, 2))


def test_mp_setup_properties_shared():
    prop_group = np.array([[0, 1], [0, 1]])
    properties = librosa.display._mp_setup_properties(prop_group, [], None)

    assert properties.shape == prop_group.shape
    assert properties[0, 0] == properties[1, 0]
    assert properties[0, 1] == properties[1, 1]
    assert properties[0, 0] is not properties[0, 1]


def test_mp_setup_properties_badprops():
    prop_group = np.array([[0, 1]])
    properties = librosa.display._mp_setup_properties(prop_group, ["color"], None)

    for prop in properties.flat:
        assert "color" not in prop


@pytest.mark.mpl_image_compare(
    baseline_images=["multiplot_wave_constructed_axes_stacked"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_multiplot_wave_constructed_axes_stacked(y, sr):
    y_stack = np.stack([y, y[::-1]])

    librosa.display.multiplot(
        "waveshow",
        y_stack,
        sr=sr,
        sharex=True,
        sharey=True,
        share_properties="row",
        labels=["forward", "backward"],
    )

    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["multiplot_wave_constructed_axes_variadic"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_multiplot_wave_constructed_axes_variadic(y, sr):
    librosa.display.multiplot(
        "waveshow",
        y,
        y[::-1],
        sr=sr,
        sharex=True,
        sharey=True,
        share_properties="row",
        labels=["forward", "backward"],
    )

    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["multiplot_wave_existing_axes_stacked"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_multiplot_wave_existing_axes_stacked(y, sr):
    y_stack = np.stack([y, y[::-1]])
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 6))

    out = librosa.display.multiplot(
        "waveshow",
        y_stack,
        axes=ax,
        sr=sr,
        share_properties="row",
        labels=["forward", "backward"],
    )

    assert out.shape == ax.shape

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["multiplot_img_existing_axes_variadic"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_multiplot_img_existing_axes_variadic(y, sr):
    y_stack = np.stack([y, y[::-1]])
    D = np.abs(librosa.stft(y_stack))
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 6))

    out = librosa.display.multiplot(
        "specshow",
        D[0],
        D[1],
        axes=ax,
        x_axis="time",
        y_axis="log",
        share_properties=False,
        titles=["forward", "backward"],
    )

    assert out.shape == ax.shape
    librosa.display.colorbar_db(out[0], ax=ax)
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["multiplot_axes_slices_mixed"],
    extensions=["png"],
    tolerance=6,
    style=STYLE,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_multiplot_axes_slices_mixed(y, sr):
    y_stack = np.stack([y, y[::-1]])
    D = np.abs(librosa.stft(y_stack))

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex="row", figsize=(10, 8))

    out_wave = librosa.display.multiplot(
        "waveshow",
        y_stack,
        axes=ax[0],
        sr=sr,
        sharex=True,
        sharey=True,
        share_properties="row",
        labels=["forward", None],
    )

    out_bars = librosa.display.multiplot(
        "wavebars",
        y_stack,
        axes=ax[1],
        sr=sr,
        sharex=True,
        sharey=True,
        share_properties="row",
        labels=[None, "backward"],
    )

    out_img = librosa.display.multiplot(
        "specshow",
        D,
        axes=ax[2],
        sr=sr,
        x_axis="time",
        y_axis="log",
        sharex=True,
        sharey=True,
        share_properties=False,
        titles=["forward", None],
    )

    assert out_wave.shape == ax[0].shape
    assert out_bars.shape == ax[1].shape
    assert out_img.shape == ax[2].shape

    for row in ax:
        for axi in np.ravel(row):
            axi.label_outer()

    return fig
