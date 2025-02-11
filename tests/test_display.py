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


from packaging import version

import pytest

import matplotlib

import matplotlib.pyplot as plt

import librosa
import librosa.display
import numpy as np
from typing import Any, Dict


STYLE = "default"

# Workaround for old freetype builds with our image fixtures
FT_VERSION = version.parse(matplotlib.ft2font.__freetype_version__)
OLD_FT = not (FT_VERSION >= version.parse("2.10"))


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
    return librosa.util.fix_frames(rhythm[1])


@pytest.fixture
def beat_t(beats, sr):
    return librosa.frames_to_time(beats, sr=sr)


@pytest.fixture
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
    T = librosa.feature.tempogram(y=y, sr=sr)

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
    T = librosa.feature.fourier_tempogram(y=y, sr=sr)

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
def test_xaxis_none_yaxis_linear(S_abs, S_signed, S_bin):
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs, y_axis="linear")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed, y_axis="fft")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin, y_axis="hz")
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
def test_xaxis_none_yaxis_log(S_abs, S_signed, S_bin):
    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs, y_axis="log")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed, y_axis="log")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin, y_axis="log")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_linear_y_none"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_linear_yaxis_none(S_abs, S_signed, S_bin):
    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="linear")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="fft")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin.T, x_axis="hz")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["x_log_y_none"], extensions=["png"], tolerance=6, style=STYLE
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_xaxis_log_yaxis_none(S_abs, S_signed, S_bin):

    plt.figure()

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="log")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="log")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_bin.T, x_axis="log")
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
    del ad

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
