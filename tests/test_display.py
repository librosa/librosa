#!/usr/bin/env python
# CREATED:2015-02-14 22:51:01 by Brian McFee <brian.mcfee@nyu.edu>
"""Unit tests for display module"""

# Disable cache
import contextlib
import os

with contextlib.suppress(KeyError):
    os.environ.pop("LIBROSA_CACHE_DIR")

import gc
import weakref
from typing import Any, cast

import matplotlib
import matplotlib.collections
import matplotlib.collections as mcoll
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import matplotlib.transforms as mtransforms
import numpy as np
import pytest
from matplotlib import colormaps as mcm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from matplotlib.testing.conftest import text_placeholders  # noqa: F401

import librosa
import librosa.display
import librosa.display.formatting
import librosa.display.signal

STYLE = "default"


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
def beats(rhythm):
    return librosa.util.fix_frames(rhythm[1])


@pytest.fixture(scope="module")
def beat_t(beats, sr):
    return librosa.frames_to_time(beats, sr=sr)


@pytest.fixture(scope="module")
def Csync(C, beats):
    return librosa.util.sync(C, beats, aggregate=np.median)


def test_axes_type_validation():
    with pytest.raises(librosa.ParameterTypeError):
        librosa.display.formatting._check_axes("not_an_axes")  # type: ignore


def test_unknown_time_unit(y):
    times = np.arange(len(y))
    plt.figure()
    ax = plt.gca()
    ax.plot(times, y)
    with pytest.raises(librosa.ParameterValueError):
        ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit="an unsupported time unit"))


@pytest.mark.mpl_image_compare(baseline_images=["complex"], extensions=["png"], style=STYLE)
@pytest.mark.usefixtures("text_placeholders")
def test_complex_input(S):
    plt.figure()
    with pytest.warns(UserWarning, match="Trying to display complex"):
        librosa.display.specshow(S)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["abs"], extensions=["png"], style=STYLE)
@pytest.mark.usefixtures("text_placeholders")
def test_abs_input(S_abs):
    plt.figure()
    librosa.display.specshow(S_abs)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["cqt_note"], extensions=["png"], style=STYLE)
def test_cqt_note(C):
    plt.figure()
    librosa.display.specshow(C, y_axis="cqt_note")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["fft_note"], extensions=["png"], style=STYLE)
def test_fft_note(S_abs):
    plt.figure()
    librosa.display.specshow(S_abs, y_axis="fft_note")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["cqt_hz"], extensions=["png"], style=STYLE)
def test_cqt_hz(C):
    plt.figure()
    librosa.display.specshow(C, y_axis="cqt_hz")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["tempo"], extensions=["png"], style=STYLE)
def test_tempo(y, sr):
    T = librosa.feature.tempogram(y=y, sr=sr, win_length=64)

    plt.figure()
    librosa.display.specshow(T, y_axis="tempo", cmap="magma")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["fourier_tempo"], extensions=["png"], style=STYLE)
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")  # our test signal is short, but this is fine here
def test_fourier_tempo(y, sr):
    T = librosa.feature.fourier_tempogram(y=y, sr=sr, win_length=64)

    plt.figure()
    librosa.display.specshow(np.abs(T), y_axis="fourier_tempo", cmap="magma")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["tonnetz"], extensions=["png"], style=STYLE)
def test_tonnetz(C):
    plt.figure()
    chroma = librosa.feature.chroma_cqt(C=C)
    ton = librosa.feature.tonnetz(chroma=chroma)
    librosa.display.specshow(ton, y_axis="tonnetz")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["chroma"], extensions=["png"], style=STYLE)
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


def test_svara_parameter_error():
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.formatting.SvaraFormatter(Sa=None)  # type: ignore


def test_chroma_fallback_defaults():
    fig, ax = plt.subplots()
    librosa.display.formatting._decorate_axis(ax.yaxis, ax_type="chroma_h", Sa=None)
    librosa.display.formatting._decorate_axis(ax.yaxis, ax_type="chroma_c", Sa=None, mela=1)
    plt.close(fig)


@pytest.mark.mpl_image_compare(baseline_images=["chroma_svara"], extensions=["png"], style=STYLE)
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


@pytest.mark.mpl_image_compare(baseline_images=["double_chroma"], extensions=["png"], style=STYLE)
def test_double_chroma(S_abs, sr):
    plt.figure()

    chr1 = librosa.feature.chroma_stft(S=S_abs**2, sr=sr)
    chr1 = np.vstack((chr1, chr1))
    librosa.display.specshow(chr1, y_axis="chroma", bins_per_octave=12)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_mel"], extensions=["png"], style=STYLE)
def test_x_mel(S_abs):
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M.T, x_axis="mel")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["y_mel"], extensions=["png"], style=STYLE)
def test_y_mel(S_abs):
    plt.figure()

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M, y_axis="mel")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["y_mel_bounded"], extensions=["png"], style=STYLE)
def test_y_mel_bounded(S_abs):
    plt.figure()

    fmin, fmax = 110, 880
    M = librosa.feature.melspectrogram(S=S_abs**2, fmin=fmin, fmax=fmax)
    librosa.display.specshow(M, y_axis="mel", fmin=fmin, fmax=fmax)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_linear"], extensions=["png"], style=STYLE)
def test_xaxis_none_yaxis_linear(S_abs, S_signed):
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs, y_axis="linear")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed, y_axis="fft")

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["specshow_ext_axes"], extensions=["png"], style=STYLE)
@pytest.mark.usefixtures("text_placeholders")
def test_specshow_ext_axes(S_abs):
    plt.figure()
    ax_left = plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)

    # implicitly ax_right
    librosa.display.specshow(S_abs, cmap="gray")
    librosa.display.specshow(S_abs, cmap="magma", ax=ax_left)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_log"], extensions=["png"], style=STYLE)
def test_xaxis_none_yaxis_log(S_abs, S_signed):
    plt.figure()

    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs, y_axis="log")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed, y_axis="log")

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_linear_y_none"], extensions=["png"], style=STYLE)
def test_xaxis_linear_yaxis_none(S_abs, S_signed):
    plt.figure()

    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="linear")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="fft")

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_log_y_none"], extensions=["png"], style=STYLE)
def test_xaxis_log_yaxis_none(S_abs, S_signed):

    plt.figure()

    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_abs.T, x_axis="log")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_signed.T, x_axis="log")

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_time_y_none"], extensions=["png"], style=STYLE)
def test_xaxis_time_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="time")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_time"], extensions=["png"], style=STYLE)
def test_xaxis_none_yaxis_time(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="time")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_frames_y_none"], extensions=["png"], style=STYLE)
def test_xaxis_frames_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="frames")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_frames"], extensions=["png"], style=STYLE)
def test_xaxis_none_yaxis_frames(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="frames")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_lag_y_none"], extensions=["png"], style=STYLE)
def test_xaxis_lag_yaxis_none(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs, x_axis="lag")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["x_none_y_lag"], extensions=["png"], style=STYLE)
def test_xaxis_time_yaxis_lag(S_abs):

    plt.figure()
    librosa.display.specshow(S_abs.T, y_axis="lag")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_images=["time_scales_auto"], extensions=["png"], style=STYLE)
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


@pytest.mark.mpl_image_compare(baseline_images=["time_unit"], extensions=["png"], style=STYLE)
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


@pytest.mark.mpl_image_compare(baseline_images=["time_unit_lag"], extensions=["png"], style=STYLE)
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


@pytest.mark.mpl_image_compare(baseline_images=["waveshow_mono"], extensions=["png"], style=STYLE)
@pytest.mark.usefixtures("text_placeholders")
def test_waveshow_mono(y, sr):

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono_trans"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_waveshow_mono_trans(y, sr):

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, transpose=True)
    return fig


@pytest.mark.mpl_image_compare(baseline_images=["waveshow_mono_zoom"], extensions=["png"], style=STYLE)
@pytest.mark.usefixtures("text_placeholders")
def test_waveshow_mono_zoom(y, sr):

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2)
    # Zoom into 1/8 of a second, make sure it's out of the initial viewport
    ax.set(xlim=[1, 1.125])
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono_zoom_trans"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_waveshow_mono_zoom_trans(y, sr):

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2, transpose=True)
    # Zoom into 1/8 of a second, make sure it's out of the initial viewport
    ax.set(ylim=[1, 1.125])
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_mono_zoom_out"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_waveshow_mono_zoom_out(y, sr):

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2)
    # Zoom into 1/8 of a second, make sure it's out of the initial viewport
    ax.set(xlim=[1, 1.125])
    # Zoom back out to get an envelope view again
    ax.set(xlim=[0, 1])
    return fig


@pytest.mark.mpl_image_compare(baseline_images=["waveshow_ext_axes"], extensions=["png"], style=STYLE)
@pytest.mark.usefixtures("text_placeholders")
def test_waveshow_ext_axes(y):
    plt.figure()
    ax_left = plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)

    # implicitly ax_right
    librosa.display.waveshow(y, color="blue")
    librosa.display.waveshow(y, color="red", ax=ax_left)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["waveshow_inverted"],
    extensions=["png"],
    style=STYLE,
)
def test_waveshow_inverted(y, sr):

    fig, ax = plt.subplots(nrows=3, sharex=True)
    # Original waveshow
    librosa.display.waveshow(y, sr=sr, ax=ax[0], invert=False, label="Regular")
    ax[0].legend(loc="upper right")

    # Inverted with default (axes face) color
    librosa.display.waveshow(y, sr=sr, ax=ax[1], invert=True, invert_color=None, label="Inverted")
    ax[1].legend(loc="upper right")

    # Inverted with custom color
    librosa.display.waveshow(y, sr=sr, ax=ax[2], invert=True, invert_color="#2d2d2d", label="Inverted custom")
    ax[2].legend(loc="upper right")

    for axi in ax.flat:
        axi.label_outer()

    return fig


@pytest.mark.mpl_image_compare(baseline_images=["waveshow_stereo"], extensions=["png"], style=STYLE)
@pytest.mark.usefixtures("text_placeholders")
def test_waveshow_stereo(y, sr):

    ys = librosa.util.stack([y, 2 * y])

    plt.figure()
    librosa.display.waveshow(ys, sr=sr)
    return plt.gcf()


def test_unknown_wavaxis(y, sr):

    plt.figure()
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.waveshow(y, sr=sr, axis="something not in the axis map")


def test_waveshow_unknown_wavaxis(y, sr):

    plt.figure()
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.waveshow(y, sr=sr, axis="something not in the axis map")


def test_waveshow_bad_maxpoints(y, sr):
    plt.figure()
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.waveshow(y, sr=sr, max_points=0)


def test_waveshow_adaptor_survives_gc_single(y, sr):
    """Regression test for #1970: single waveshow adaptor should survive GC."""
    fig, ax = plt.subplots()

    librosa.display.waveshow(y, sr=sr, ax=ax, max_points=sr // 2)

    gc.collect()

    ax.set_xlim(0, 0.025)
    fig.canvas.draw()

    # Data lines only (not ticks)
    visible_data_lines = [line for line in ax.lines if line.get_visible()]

    # Envelope(s) are PolyCollections in ax.collections
    polys = [c for c in ax.collections if isinstance(c, matplotlib.collections.PolyCollection)]

    assert len(visible_data_lines) >= 1, "Sample/step artist should be visible after zoom"
    assert all(not p.get_visible() for p in polys), "Envelope should be hidden after zoom"

    plt.close(fig)


def test_waveshow_adaptor_survives_gc_multi(y, sr):
    """Regression test for #1970: multiple waveshow adaptors should survive GC."""
    fig, ax = plt.subplots()

    librosa.display.waveshow(y, sr=sr, ax=ax, color="blue", max_points=sr // 2)
    librosa.display.waveshow(y, sr=sr, ax=ax, color="red", alpha=0.5, zorder=-1, max_points=sr // 2)

    gc.collect()

    ax.set_xlim(0, 0.025)
    fig.canvas.draw()

    visible_data_lines = [line for line in ax.lines if line.get_visible()]
    polys = [c for c in ax.collections if isinstance(c, matplotlib.collections.PolyCollection)]

    assert len(visible_data_lines) >= 2, "Both sample/step artists should be visible after zoom"
    assert all(not p.get_visible() for p in polys), "All envelopes should be hidden after zoom"

    plt.close(fig)


def test_waveshow_registry_cleanup_on_axes_gc():
    """Test that the adaptor registry uses WeakKeyDictionary correctly.

    This verifies the fix for #1970: adaptors are stored in a WeakKeyDictionary
    keyed by Axes, so they will be cleaned up when the Axes is garbage collected.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

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


@pytest.mark.parametrize("axis", ["x_axis", "y_axis"])
def test_unknown_axis(S_abs, axis: str):

    kwargs: dict[str, Any] = {axis: "something not in the axis map"}
    plt.figure()
    with pytest.raises(librosa.ParameterValueError):
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

    if isinstance(cmap1, matplotlib.colors.ListedColormap) and isinstance(cmap2, matplotlib.colors.ListedColormap):
        assert np.allclose(cmap1.colors, cmap2.colors)
    elif isinstance(cmap1, matplotlib.colors.LinearSegmentedColormap) and isinstance(
        cmap2, matplotlib.colors.LinearSegmentedColormap
    ):
        assert cmap1.name == cmap2.name
    else:
        assert cmap1 == cmap2
        assert type(cmap1) is type(cmap2)


@pytest.mark.mpl_image_compare(baseline_images=["coords"], extensions=["png"], style=STYLE)
def test_coords(Csync, beat_t):

    plt.figure()
    librosa.display.specshow(Csync, x_coords=beat_t, x_axis="time", y_axis="cqt_note")
    return plt.gcf()


def test_bad_coords(S_abs):

    with pytest.raises(librosa.ParameterShapeError):
        librosa.display.specshow(S_abs, x_coords=np.arange(S_abs.shape[1] // 2))


@pytest.mark.mpl_image_compare(baseline_images=["sharex_specshow_ms"], extensions=["png"], style=STYLE)
def test_sharex_specshow_ms(S_abs, y, sr):

    # Correct time range ~= 4.6 s or 4600ms
    # Due to shared x_axis, both plots are plotted in 's'.
    fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(S_abs, ref=np.max), x_axis="time", ax=ax)
    ax.set(xlabel="")  # hide the x label here, which is not propagated automatically
    ax2.margins(x=0)
    librosa.display.waveshow(y, sr=sr, axis="ms", ax=ax2)
    ax2.set(xlabel="")  # hide the x label here, which is not propagated automatically
    return fig


@pytest.mark.mpl_image_compare(baseline_images=["sharex_waveplot_ms"], extensions=["png"], style=STYLE)
def test_sharex_waveplot_ms(y, sr, S_abs):

    # Correct time range ~= 4.6 s or 4600ms
    # Due to shared x_axis, both plots are plotted in 'ms'.
    fig, (ax, ax2) = plt.subplots(sharex=True, nrows=2, figsize=(8, 8))
    ax.margins(x=0)
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(xlabel="")  # hide the x label here, which is not propagated automatically
    ax2.margins(x=0)
    librosa.display.specshow(librosa.amplitude_to_db(S_abs, ref=np.max), x_axis="ms", ax=ax2)
    ax2.set(xlabel="")  # hide the x label here, which is not propagated automatically
    return fig


@pytest.mark.parametrize(
    "format_str",
    [
        "cqt_hz",
        "cqt_note",
        "vqt_hz",
        "logfreq_hz",
        "logfreq_note",
        "pitch_hz",
        "pitch_note",
    ],
)
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


@pytest.mark.parametrize(
    "y_axis,expected_formatter",
    [
        ("logfreq_hz", librosa.display.LogHzFormatter),
        ("logfreq_note", librosa.display.NoteFormatter),
        ("logfreq_svara", librosa.display.SvaraFormatter),
        ("logfreq_oct3", librosa.display.formatting.AdaptiveEngFormatter),
        ("pitch_hz", librosa.display.LogHzFormatter),
        ("pitch_note", librosa.display.NoteFormatter),
        ("pitch_svara", librosa.display.SvaraFormatter),
        ("pitch_oct3", librosa.display.formatting.AdaptiveEngFormatter),
        ("pitch_fjs", librosa.display.FJSFormatter),
    ],
)
def test_specshow_logfreq_pitch_axes(y_axis, expected_formatter):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    D = np.ones((84, 10))
    librosa.display.specshow(
        D,
        y_axis=y_axis,
        fmin=32.7,
        bins_per_octave=12,
        Sa=261.63,
        intervals="ji3" if y_axis == "pitch_fjs" else None,
        ax=ax,
    )
    assert isinstance(ax.yaxis.get_major_formatter(), expected_formatter)
    plt.close(fig)


@pytest.mark.parametrize("y_axis", ["pitch_hz", "pitch_note", "pitch_fjs"])
@pytest.mark.parametrize(
    "intervals",
    [
        "ji3",
        "pythagorean",
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.95, 1.98],
    ],
)
def test_specshow_pitch_temperaments(y_axis, intervals):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    D = np.ones((60, 10))
    librosa.display.specshow(D, y_axis=y_axis, fmin=32.7, bins_per_octave=12, intervals=intervals, ax=ax)
    plt.close(fig)


def test_specshow_svara_validation():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    D = np.ones((60, 10))
    with pytest.raises(librosa.ParameterValueError, match="Sa frequency is required"):
        librosa.display.specshow(D, y_axis="logfreq_svara", ax=ax)
    with pytest.raises(librosa.ParameterValueError, match="Sa frequency is required"):
        librosa.display.specshow(D, y_axis="pitch_svara", ax=ax)
    with pytest.raises(librosa.ParameterValueError, match="Svara display is not currently supported"):
        librosa.display.specshow(D, y_axis="pitch_svara", Sa=261.63, intervals="ji3", ax=ax)
    plt.close(fig)


def test_specshow_logfreq_pitch_same_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    R = np.ones((84, 84))
    librosa.display.specshow(
        R,
        x_axis="logfreq_hz",
        y_axis="logfreq_note",
        fmin=32.7,
        bins_per_octave=12,
        ax=ax,
    )
    assert ax.get_aspect() == 1.0
    plt.close(fig)

    fig, ax = plt.subplots()
    librosa.display.specshow(
        R,
        x_axis="pitch_hz",
        y_axis="pitch_note",
        fmin=32.7,
        bins_per_octave=12,
        ax=ax,
    )
    assert ax.get_aspect() == 1.0
    plt.close(fig)


@pytest.mark.mpl_image_compare(baseline_images=["cqt_svara"], extensions=["png"], style=STYLE)
def test_display_cqt_svara(C):

    Camp = librosa.amplitude_to_db(C, ref=np.max)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

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


@pytest.mark.mpl_image_compare(baseline_images=["fft_svara"], extensions=["png"], style=STYLE)
def test_display_fft_svara(S_abs):

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

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


@pytest.mark.mpl_image_compare(baseline_images=["nfft_odd"], extensions=["png"], style=STYLE)
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


@pytest.mark.mpl_image_compare(baseline_images=["nfft_odd_ftempo"], extensions=["png"], style=STYLE)
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
    assert librosa.display._same_axes(x_axis, y_axis, xlim, ylim) == out


def test_auto_aspect():

    _fig, ax = plt.subplots(nrows=5)

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
    librosa.display.specshow(X, x_axis="chroma", y_axis="chroma", auto_aspect=False, ax=ax[2])
    assert ax[2].get_aspect() == "auto"

    # Different extents with auto_aspect=True should retain auto scaling
    librosa.display.specshow(X[:2, :], x_axis="chroma", y_axis="chroma", auto_aspect=True, ax=ax[3])
    assert ax[3].get_aspect() == "auto"

    # different axes with compatible types and auto_aspect=True should force equal scaling
    librosa.display.specshow(X, x_axis="time", y_axis="ms", ax=ax[4])
    assert ax[4].get_aspect() == 1.0


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_unicode_true"],
    extensions=["png"],
    style=STYLE,
)
def test_specshow_unicode_true(C, sr):

    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, threshold=0.9)

    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

    # Hindustani, no thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax[0], unicode=True)

    # Hindustani, kafi thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax[1], thaat="kafi", unicode=True)

    # Carnatic, mela 22
    librosa.display.specshow(chroma, y_axis="chroma_c", Sa=5, ax=ax[2], mela=22, unicode=True)

    # Carnatic, mela 1
    librosa.display.specshow(chroma, y_axis="chroma_c", Sa=7, ax=ax[3], mela=1, unicode=True)

    # Pitches
    librosa.display.specshow(chroma, y_axis="chroma", ax=ax[4], key="Eb:maj", unicode=True)

    for axi in ax:
        axi.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_unicode_false"],
    extensions=["png"],
    style=STYLE,
)
def test_specshow_unicode_false(C, sr):

    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, threshold=0.9)

    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

    # Hindustani, no thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax[0], unicode=False)

    # Hindustani, kafi thaat
    librosa.display.specshow(chroma, y_axis="chroma_h", Sa=5, ax=ax[1], thaat="kafi", unicode=False)

    # Carnatic, mela 22
    librosa.display.specshow(chroma, y_axis="chroma_c", Sa=5, ax=ax[2], mela=22, unicode=False)

    # Carnatic, mela 1
    librosa.display.specshow(chroma, y_axis="chroma_c", Sa=7, ax=ax[3], mela=1, unicode=False)

    librosa.display.specshow(chroma, y_axis="chroma", ax=ax[4], key="Eb:maj", unicode=False)

    for axi in ax:
        axi.label_outer()

    return fig


def test_waveshow_disconnect(y, sr):
    _fig, ax = plt.subplots()
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


def test_adaptor_destructor():
    def _run_test():
        fig, ax = plt.subplots()
        try:
            mock_line = mlines.Line2D([0], [0])
            mock_poly = mcoll.PolyCollection([])

            adaptor = librosa.display.signal.AdaptiveWaveplot(
                times=np.array([0]), y=np.array([0]), steps=mock_line, envelope=mock_poly
            )
            adaptor.connect(ax)

            assert adaptor.ax is ax
            assert adaptor.cid is not None

            ref = weakref.ref(adaptor)
            assert ref() is not None

            adaptor.__del__()
            assert adaptor.ax is None
            assert adaptor.cid is None

            return ref
        finally:
            plt.close(fig)

    ref = _run_test()
    gc.collect()
    assert ref() is None


def test_waveshow_deladaptor(y, sr):
    _fig, ax = plt.subplots()
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


def test_waveshow_cartesian_transform(y, sr):
    fig, ax = plt.subplots()
    try:
        # Scale time by 2.0 and offset by 1.0; scale amplitude by 3.0 and offset by 0.5
        pre = matplotlib.transforms.Affine2D().scale(2.0, 3.0).translate(1.0, 0.5)
        ad = librosa.display.waveshow(y, sr=sr, ax=ax, transform=pre + ax.transData)

        # Verify pre_transform extraction
        assert ad.pre_transform == pre

        # Verify axis decoration was skipped (no TimeFormatter on xaxis)
        assert not isinstance(ax.xaxis.get_major_formatter(), librosa.display.TimeFormatter)
        assert ax.xaxis.get_label_text() != "Time"

        # Zoom into a narrow time window in transformed space:
        # View signal around t = 0.5s with duration = 0.01s (well below max_samples/sr)
        # In transformed space: x = 2 * t + 1
        t_center = 0.5
        t_span = 0.01
        x0 = 2.0 * (t_center - t_span / 2) + 1.0
        x1 = 2.0 * (t_center + t_span / 2) + 1.0
        ax.set(xlim=[x0, x1])

        # Steps should become visible and envelope hidden
        assert ad.steps is not None and ad.envelope is not None
        assert ad.steps.get_visible() and not ad.envelope.get_visible()

        # The sliced steps data should be in signal time coordinates around t_center
        steps_time = np.asarray(ad.steps.get_xdata())
        assert np.isclose(steps_time[len(steps_time) // 2], t_center, atol=0.05)

        # Zoom back out to full signal in transformed space
        t_max = len(y) / sr
        ax.set(xlim=[1.0, 2.0 * t_max + 1.0])
        assert ad.envelope.get_visible() and not ad.steps.get_visible()
    finally:
        plt.close(fig)


def test_waveshow_polar_transform(y, sr):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    assert isinstance(ax, PolarAxes)
    try:
        # Check initial polar limits
        assert ax.name == "polar"
        assert np.isclose(ax.get_thetamin(), 0.0)
        assert np.isclose(ax.get_thetamax(), 360.0)

        duration = len(y) / sr
        # Map time [0, duration] -> theta [0, 2*pi]; amplitude [-1, 1] -> radius [0.5, 1.5]
        pre = matplotlib.transforms.Affine2D().scale(2 * np.pi / duration, 0.5).translate(0.0, 1.0)
        ad = librosa.display.waveshow(y, sr=sr, ax=ax, transform=pre + ax.transData)

        assert ad.pre_transform == pre

        # Polar axis decorations must not be overwritten with TimeFormatter
        assert not isinstance(ax.xaxis.get_major_formatter(), librosa.display.TimeFormatter)
        assert ax.xaxis.get_label_text() != "Time"

        # thetamin/thetamax should NOT be corrupted
        assert np.isclose(ax.get_thetamin(), 0.0)
        assert np.isclose(ax.get_thetamax(), 360.0)

        # Zoom into a small angular slice around pi radians (~ 0.5 * duration)
        theta_center = np.pi
        theta_span = 0.01  # small enough to trigger steps view
        ax.set(xlim=[theta_center - theta_span / 2, theta_center + theta_span / 2])

        assert ad.steps is not None and ad.envelope is not None
        assert ad.steps.get_visible() and not ad.envelope.get_visible()

        # Verify steps data is centered around 0.5 * duration in signal seconds
        steps_time = np.asarray(ad.steps.get_xdata())
        expected_t = 0.5 * duration
        assert np.isclose(steps_time[len(steps_time) // 2], expected_t, atol=0.05)

        # Zoom back out to [0, 2*pi]
        ax.set(xlim=[0.0, 2 * np.pi])
        assert ad.envelope.get_visible() and not ad.steps.get_visible()
        assert np.isclose(ax.get_thetamin(), 0.0)
        assert np.isclose(ax.get_thetamax(), 360.0)
    finally:
        plt.close(fig)


def test_waveshow_pre_transform_fallback(y, sr):
    # 1. Singular pre-transform (LinAlgError on inverted()) falls back to full signal bounds
    _, ax = plt.subplots()
    pre = mtransforms.Affine2D().scale(0.0, 1.0)
    ad = librosa.display.waveshow(y, sr=sr, ax=ax, transform=pre + ax.transData)
    ax.set(xlim=[-10, 10])
    ad.update(ax)
    assert ad.envelope is not None and ad.envelope.get_visible()

    # 2. Non-data transform (ax.transAxes) sets use_view_limits=False for waveshow and passes through for wavebars
    _, ax2 = plt.subplots()
    ad2 = librosa.display.waveshow(y, sr=sr, ax=ax2, transform=ax2.transAxes)
    assert ad2.use_view_limits is False
    assert isinstance(ad2.pre_transform, mtransforms.IdentityTransform)
    assert ad2.steps is not None and ad2.envelope is not None
    assert ad2.steps.get_transform() is ax2.transAxes
    assert ad2.envelope.get_visible()
    coll2 = librosa.display.wavebars(y, sr=sr, ax=ax2, transform=ax2.transAxes)
    assert coll2.get_transform() is ax2.transAxes

    # 3. Post-transform (axes.transData + ScaledTranslation) preserves dataLim and Time decoration
    fig3, ax3 = plt.subplots()
    post = ax3.transData + mtransforms.ScaledTranslation(5 / 72, 5 / 72, fig3.dpi_scale_trans)
    coll3 = librosa.display.wavebars(y, sr=sr, ax=ax3, transform=post)
    assert isinstance(coll3, mcoll.PatchCollection)
    assert ax3.get_xlim()[1] > 1.0
    assert ax3.xaxis.get_label_text() == "Time"

    ax3.clear()
    ad3 = librosa.display.waveshow(y, sr=sr, ax=ax3, transform=post)
    assert ad3.use_view_limits is True
    assert isinstance(ad3.pre_transform, mtransforms.IdentityTransform)
    assert ax3.xaxis.get_label_text() == "Time"

    # 4. Non-separable 2D pre-transform uses all 4 viewport corners in AdaptiveWaveplot.update
    _, ax4 = plt.subplots()
    rot = mtransforms.Affine2D().rotate(np.pi / 4)
    ad4 = librosa.display.waveshow(y, sr=sr, ax=ax4, transform=rot + ax4.transData)
    ax4.set_xlim(-0.2, 0.2)
    ax4.set_ylim(0.0, 0.2)
    ad4.update(ax4)
    assert ad4.steps is not None and ad4.steps.get_visible()


@pytest.mark.mpl_image_compare(baseline_images=["specshow_vqt"], extensions=["png"], style=STYLE)
def test_specshow_vqt(C):

    fig, ax = plt.subplots(nrows=4, figsize=(12, 10))

    librosa.display.specshow(C, y_axis="vqt_hz", intervals="ji5", ax=ax[0])
    librosa.display.specshow(C, y_axis="vqt_note", intervals="ji5", ax=ax[1])
    librosa.display.specshow(C, y_axis="vqt_fjs", intervals="ji5", ax=ax[2])
    librosa.display.specshow(C, y_axis="vqt_fjs", intervals="ji5", ax=ax[3], unicode=False)

    for _ax in ax:
        _ax.set(ylim=[55, 165])
    return fig


def test_chromafjs_badintervals():
    with pytest.raises(librosa.ParameterTypeError):
        librosa.display.ChromaFJSFormatter(intervals=dict())


def test_chromafjs_badbpo():
    with pytest.raises(librosa.ParameterTypeError):
        librosa.display.ChromaFJSFormatter(intervals="ji3", bins_per_octave=None)


@pytest.mark.mpl_image_compare(baseline_images=["chroma_fjs"], extensions=["png"], style=STYLE)
def test_specshow_chromafjs(C, sr):

    # This isn't a VQT chroma, but that's not important here
    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, threshold=0.9)

    intervals = librosa.plimit_intervals(primes=[3, 5])

    fig, ax = plt.subplots(nrows=2, figsize=(12, 8))

    librosa.display.specshow(chroma, y_axis="chroma_fjs", intervals="ji5", ax=ax[0])
    librosa.display.specshow(chroma, y_axis="chroma_fjs", intervals=intervals, ax=ax[1])

    return fig


def test_vqt_hz_nointervals(C, sr):
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.specshow(C, sr=sr, y_axis="vqt_hz")


@pytest.mark.parametrize("vscale", ["dBFS[1]", "dBFS[power,1]"])
def test_parse_vscale_dbfs_ref(vscale):
    # This should raise an error because a reference value is
    # not allowed with dBFS
    with pytest.raises(librosa.ParameterValueError):
        librosa.display._parse_vscale(vscale)


@pytest.mark.parametrize("vscale", ["bad string", "dB[gibberish]", "dBFS[gibberish]"])
def test_parse_vscale_fail(vscale):
    with pytest.raises(librosa.ParameterValueError):
        librosa.display._parse_vscale(vscale)


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
    assert librosa.display._parse_vscale(vscale) == (mode, scale_type, ref)


@pytest.mark.mpl_image_compare(baseline_images=["specshow_vscale"], extensions=["png"], style=STYLE)
def test_specshow_vscale(S):
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(12, 12))

    # first column is dB, dBFS, dB with ref value
    i1 = librosa.display.specshow(S, vscale="dB", y_axis="log", x_axis="time", ax=ax[0, 0])
    fig.colorbar(i1, ax=ax[0, 0])
    ax[0, 0].set_title("dB")
    i2 = librosa.display.specshow(S, vscale="dBFS", y_axis="log", x_axis="time", ax=ax[1, 0])
    fig.colorbar(i2, ax=ax[1, 0])
    ax[1, 0].set_title("dBFS")
    i3 = librosa.display.specshow(S, vscale="dB[1e-2]", y_axis="log", x_axis="time", ax=ax[2, 0])
    fig.colorbar(i3, ax=ax[2, 0])
    ax[2, 0].set_title("dB, ref=1e-2")

    # second column is dB[power], dBFS[power], dB[power] with ref value
    i4 = librosa.display.specshow(S, vscale="dB[power]", y_axis="log", x_axis="time", ax=ax[0, 1])
    fig.colorbar(i4, ax=ax[0, 1])
    ax[0, 1].set_title("dB power")
    i5 = librosa.display.specshow(S, vscale="dBFS[power]", y_axis="log", x_axis="time", ax=ax[1, 1])
    fig.colorbar(i5, ax=ax[1, 1])
    ax[1, 1].set_title("dBFS power")
    i6 = librosa.display.specshow(S, vscale="dB[power,1e-2]", y_axis="log", x_axis="time", ax=ax[2, 1])
    fig.colorbar(i6, ax=ax[2, 1])
    ax[2, 1].set_title("dB power, ref=1e-2")

    for _ax in ax.flat:
        _ax.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["specshow_vscale_phase"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
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
    i7 = librosa.display.specshow(S, vscale="phase", y_axis="log", x_axis="time", ax=ax[0], alpha=alpha)
    fig.colorbar(i7, ax=ax[0])
    ax[0].set_title("phase")

    i8 = librosa.display.specshow(S, vscale="dphase", y_axis="log", x_axis="time", ax=ax[1], alpha=alpha)
    fig.colorbar(i8, ax=ax[1])
    ax[1].set_title("dphase")

    i9 = librosa.display.specshow(S.T, vscale="dphase_t", x_axis="log", y_axis="time", ax=ax[2], alpha=alpha.T)
    fig.colorbar(i9, ax=ax[2])
    ax[2].set_title("dphase_t")

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["colorbar_db"],
    extensions=["png"],
    style=STYLE,
)
def test_colorbar_db(S):
    fig, ax = plt.subplots(nrows=2)
    i1 = librosa.display.specshow(S, vscale="dB", y_axis="log", x_axis="time", ax=ax[0])
    librosa.display.colorbar_db(i1, ax=ax[0])
    i2 = librosa.display.specshow(S, vscale="dBFS", y_axis="log", x_axis="time", ax=ax[1])
    librosa.display.colorbar_db(i2, ax=ax[1], label="dBFS")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["colorbar_phase"],
    extensions=["png"],
    style=STYLE,
)
def test_colorbar_phase(S):
    alpha = np.abs(S)
    alpha /= np.max(alpha)
    fig, ax = plt.subplots(nrows=2)
    i1 = librosa.display.specshow(S, vscale="phase", y_axis="log", x_axis="time", alpha=alpha, ax=ax[0])
    librosa.display.colorbar_phase(i1, ax=ax[0])
    i2 = librosa.display.specshow(S, vscale="dphase", y_axis="log", x_axis="time", alpha=alpha, ax=ax[1])
    librosa.display.colorbar_phase(i2, ax=ax[1], label="Δ radians")
    return fig


def test_axis_fallbacks():
    fig, ax = plt.subplots()
    im_real = ax.imshow(np.array([[0.0, 1.0], [1.0, 0.0]]))

    librosa.display.colorbar_db(im_real, ax=None)
    librosa.display.colorbar_phase(im_real, ax=None)

    mock_line = mlines.Line2D([0], [0])
    mock_poly = mcoll.PolyCollection([])
    adaptor = librosa.display.signal.AdaptiveWaveplot(
        times=np.array([0]), y=np.array([0]), steps=mock_line, envelope=mock_poly
    )
    proxy = adaptor.label_proxy_
    handler = librosa.display.signal._AdaptiveWaveplotHandler()

    # plot something so legend isn't totally empty, just in case
    ax.plot([0, 1], [0, 1], label="test")
    leg = ax.legend()

    handler.create_artists(
        leg,
        orig_handle=proxy,
        xdescent=0,
        ydescent=0,
        width=10,
        height=10,
        fontsize=10,
        trans=mtransforms.IdentityTransform(),
    )
    plt.close(fig)


@pytest.mark.mpl_image_compare(
    baseline_images=["diverging_slopes"],
    extensions=["png"],
    style=STYLE,
)
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
    i3 = librosa.display.specshow(S_signed, y_axis="log", x_axis="time", ax=ax[1, 0], cmap_div="Spectral_r")
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
    style=STYLE,
)
def test_oct3(S_abs, C):

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # STFT, Mel
    # CQT, VQT

    librosa.display.specshow(S_abs, vscale="dBFS", y_axis="log_oct3", ax=ax[0, 0])

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M, vscale="dBFS[power]", y_axis="mel_oct3", ax=ax[0, 1])

    librosa.display.specshow(C, vscale="dBFS", y_axis="cqt_oct3", ax=ax[1, 0])

    librosa.display.specshow(C, vscale="dBFS", y_axis="vqt_oct3", ax=ax[1, 1], intervals="equal")

    # Put the ticks on the right just to reduce crowding
    ax[0, 1].yaxis.tick_right()
    ax[1, 1].yaxis.tick_right()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["oct3_minor"],
    extensions=["png"],
    style=STYLE,
)
def test_oct3_minor(S_abs, C):

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # STFT, Mel
    # CQT, VQT

    librosa.display.specshow(S_abs, vscale="dBFS", y_axis="log_oct3", ax=ax[0, 0])

    M = librosa.feature.melspectrogram(S=S_abs**2)
    librosa.display.specshow(M, vscale="dBFS[power]", y_axis="mel_oct3", ax=ax[0, 1])

    librosa.display.specshow(C, vscale="dBFS", y_axis="cqt_oct3", ax=ax[1, 0])

    librosa.display.specshow(C, vscale="dBFS", y_axis="vqt_oct3", ax=ax[1, 1], intervals="equal")

    # Put the ticks on the right just to reduce crowding
    ax[0, 1].yaxis.tick_right()
    ax[1, 1].yaxis.tick_right()

    for axi in ax.flat:
        axi.set_ylim([200, 600])

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["wavebars"],
    extensions=["png"],
    style=STYLE,
)
def test_wavebars(y, sr):

    fig, ax = plt.subplots(nrows=6, layout="tight", figsize=(8, 12))

    librosa.display.wavebars(y, sr=sr, ax=ax[0], label="100")
    librosa.display.wavebars(y, sr=sr, color="C1", n_bars=150, rounding_ratio=0, ax=ax[1], label="150 square")
    librosa.display.wavebars(y, sr=sr, color="C2", n_bars=50, ax=ax[2], invert=True, label="50 inverted")
    librosa.display.wavebars(y, sr=sr, color="C3", gap_ratio=0, ax=ax[3], label="no gap")
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
    style=STYLE,
)
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
    librosa.display.wavebars(y, sr=sr, color="C3", gap_ratio=0, transpose=True, ax=ax[3], label="no gap")
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
    librosa.display.wavebars(y, sr=sr, ax=ax[5], color="C5", axis="ms", transpose=True, label="time_ms")

    for axi in ax.flat:
        axi.legend()

    return fig


def test_wavebars_nbars_validation():
    fig, ax = plt.subplots()
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.wavebars(np.array([0.0, 1.0]), n_bars=0, ax=ax)
    plt.close(fig)


def test_wavebars_cartesian_transform(y, sr):
    fig, ax = plt.subplots()
    try:
        # Scale time by 2.0 and offset by 1.0; scale amplitude by 3.0 and offset by 0.5
        pre = matplotlib.transforms.Affine2D().scale(2.0, 3.0).translate(1.0, 0.5)
        coll = librosa.display.wavebars(y, sr=sr, ax=ax, transform=pre + ax.transData, label="bars")
        assert isinstance(coll, matplotlib.collections.PatchCollection)
        ax.legend()

        # Verify axis decoration was skipped (no TimeFormatter on xaxis)
        assert not isinstance(ax.xaxis.get_major_formatter(), librosa.display.TimeFormatter)
        assert ax.xaxis.get_label_text() != "Time"

        # Verify data limits are mapped through pre-transform
        duration = len(y) / sr
        assert ax.dataLim.x0 >= 1.0 - 0.1
        assert ax.dataLim.x1 <= 2.0 * duration + 1.0 + 0.1
    finally:
        plt.close(fig)


def test_wavebars_polar_transform(y, sr):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    assert isinstance(ax, PolarAxes)
    try:
        assert ax.name == "polar"
        assert np.isclose(ax.get_thetamin(), 0.0)
        assert np.isclose(ax.get_thetamax(), 360.0)

        duration = len(y) / sr
        # Map time [0, duration] -> theta [0, 2*pi]; amplitude [-1, 1] -> radius [0.5, 1.5]
        pre = matplotlib.transforms.Affine2D().scale(2 * np.pi / duration, 0.5).translate(0.0, 1.0)
        coll = librosa.display.wavebars(y, sr=sr, ax=ax, transform=pre + ax.transData, label="polar_bars")
        assert isinstance(coll, matplotlib.collections.PatchCollection)
        leg = ax.legend()
        assert "polar_bars" in [t.get_text() for t in leg.get_texts()]

        # Polar axis decorations must not be overwritten with TimeFormatter
        assert not isinstance(ax.xaxis.get_major_formatter(), librosa.display.TimeFormatter)
        assert ax.xaxis.get_label_text() != "Time"

        # thetamin/thetamax should NOT be corrupted
        assert np.isclose(ax.get_thetamin(), 0.0)
        assert np.isclose(ax.get_thetamax(), 360.0)

        # Verify data limits are in (theta, r) coordinates
        assert 0.0 <= ax.dataLim.x0
        assert ax.dataLim.x1 <= 2 * np.pi + 1e-3
        assert 0.0 <= ax.dataLim.y0
        assert ax.dataLim.y1 <= 2.0
    finally:
        plt.close(fig)


@pytest.mark.mpl_image_compare(
    baseline_images=["waveloop_waveshow"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_waveloop_waveshow(y, sr):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), layout="tight")
    librosa.display.waveloop(y, sr=sr, ax=ax[0])
    ax[0].set_title("Mono")

    y_stereo = np.vstack([y, y[::-1]])
    librosa.display.waveloop(y_stereo, sr=sr, r_range=(0.4, 0.95), ax=ax[1])
    ax[1].set_title("Stereo")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveloop_wavebars"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_waveloop_wavebars(y, sr):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw={"projection": "polar"}, layout="tight")
    librosa.display.waveloop(y, sr=sr, method="wavebars", n_bars=64, r_range=(0.5, 1.0), ax=ax[0])
    ax[0].set_title("Wavebars (64)")

    librosa.display.waveloop(
        y,
        sr=sr,
        method="wavebars",
        n_bars=128,
        gap_ratio=0.1,
        rounding_ratio=0.2,
        color="C1",
        time_axis=8,
        ax=ax[1],
    )
    ax[1].set_title("Wavebars (128, 8 bins)")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveloop_concentric_tracks"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_waveloop_concentric_tracks(y, sr):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"}, layout="tight")
    librosa.display.waveloop(y, sr=sr, r_range=(0.6, 1.0), method="waveshow", color="C0", label="Harmonic", ax=ax)
    librosa.display.waveloop(
        y[::-1],
        sr=sr,
        r_range=(0.2, 0.5),
        method="wavebars",
        n_bars=64,
        color="C1",
        decorate=False,
        label="Percussive",
        ax=ax,
    )
    ax.legend(loc="upper right")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["waveloop_options"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_waveloop_options(y, sr):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw={"projection": "polar"}, layout="tight")
    dur = float(librosa.get_duration(y=y, sr=sr)) + 1.0
    librosa.display.waveloop(
        y,
        sr=sr,
        offset=1.0,
        duration=dur,
        direction="counterclockwise",
        theta_offset=0.0,
        time_axis=12,
        ax=ax[0],
    )
    ax[0].set_title("CCW, theta_0=0, offset=1s")

    librosa.display.waveloop(
        y,
        sr=sr,
        time_axis=None,
        r_range=(0.2, 0.8),
        offset=1.0,
        ax=ax[1],
    )
    ax[1].set_title("No ticks, offset=1s")
    return fig


@pytest.mark.parametrize(
    "method,expected_type",
    [("waveshow", librosa.display.AdaptiveWaveplot), ("wavebars", mcoll.PatchCollection)],
)
def test_waveloop_defaults(y, sr, method, expected_type):
    kw: dict[str, Any] = {"n_bars": 64} if method == "wavebars" else {}
    out = librosa.display.waveloop(y, sr=sr, method=method, **kw)
    assert isinstance(out, expected_type)
    ax = out.axes if isinstance(out, mcoll.PatchCollection) else out.ax
    assert isinstance(ax, PolarAxes)
    assert ax.name == "polar"
    assert ax.get_theta_direction() == -1
    assert np.isclose(ax.get_theta_offset(), np.pi / 2)
    assert len(ax.get_yticks()) == 0
    assert ax.get_ylim() == (0.0, 1.0)


def test_waveloop_duration_fractional(y, sr):
    sig_dur = len(y) / sr
    target_dur = sig_dur * 2.0
    adaptor = librosa.display.waveloop(y, sr=sr, duration=target_dur)
    assert isinstance(adaptor, librosa.display.AdaptiveWaveplot)
    assert adaptor.steps is not None and adaptor.steps.figure is not None
    fig = adaptor.steps.figure
    assert isinstance(fig, Figure)
    try:
        ax = adaptor.steps.axes
        assert isinstance(ax, PolarAxes)
        # Signal fills [0, pi] within [0, 2*pi]
        assert ax.dataLim.x0 >= 0.0
        assert ax.dataLim.x1 <= np.pi + 0.1
    finally:
        plt.close(fig)


def test_waveloop_peak_norm(y, sr):
    y_loud = 5.0 * y
    adaptor = librosa.display.waveloop(y_loud, sr=sr, peak_norm=True, r_range=(0.2, 0.8))
    assert isinstance(adaptor, librosa.display.AdaptiveWaveplot)
    assert adaptor.steps is not None and adaptor.steps.figure is not None
    fig = adaptor.steps.figure
    assert isinstance(fig, Figure)
    try:
        ax = adaptor.steps.axes
        assert isinstance(ax, PolarAxes)
        assert ax.get_ylim()[0] == 0.0
        assert ax.get_ylim()[1] == 0.8
        # Make sure no radius exceeds 0.8 or falls below 0.2
        assert ax.dataLim.y0 >= 0.2 - 0.05
        assert ax.dataLim.y1 <= 0.8 + 0.05
    finally:
        plt.close(fig)

    # Test peak_norm=False with standard amplitude
    adaptor_nonorm = librosa.display.waveloop(y, sr=sr, peak_norm=False, r_range=(0.2, 0.8))
    assert isinstance(adaptor_nonorm, librosa.display.AdaptiveWaveplot)
    assert adaptor_nonorm.steps is not None and adaptor_nonorm.steps.figure is not None
    fig_nonorm = adaptor_nonorm.steps.figure
    assert isinstance(fig_nonorm, Figure)
    plt.close(fig_nonorm)


def test_waveloop_concentric(y, sr):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    assert isinstance(ax, PolarAxes)
    try:
        # First track (inner ring)
        librosa.display.waveloop(y, sr=sr, r_range=(0.2, 0.5), ax=ax)
        assert ax.get_ylim() == (0.0, 0.5)
        assert ax.get_autoscaley_on()
        # Second track (outer ring)
        librosa.display.waveloop(y, sr=sr, r_range=(0.6, 1.0), ax=ax)
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.get_autoscaley_on()
        assert ax.get_theta_direction() == -1
        assert np.isclose(ax.get_theta_offset(), np.pi / 2)
        assert len(ax.get_yticks()) == 0
    finally:
        plt.close(fig)

    # Outer ring first, then inner ring: cumulative ylim must stay (0.0, 1.0)
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        librosa.display.waveloop(y, sr=sr, r_range=(0.6, 1.0), ax=ax2)
        librosa.display.waveloop(y, sr=sr, r_range=(0.2, 0.5), method="wavebars", n_bars=32, ax=ax2)
        assert ax2.get_ylim() == (0.0, 1.0)
        assert ax2.dataLim.x0 >= 0.0
    finally:
        plt.close(fig2)

    # Silent signal should still span r_range cleanly
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        librosa.display.waveloop(np.zeros_like(y), sr=sr, r_range=(0.5, 1.0), ax=ax3)
        assert ax3.get_ylim() == (0.0, 1.0)
    finally:
        plt.close(fig3)

    # Explicit user ylim is preserved by autoscale_view
    fig4, ax4 = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        ax4.set_ylim(0.0, 2.0)
        librosa.display.waveloop(y, sr=sr, r_range=(0.5, 1.0), ax=ax4)
        assert ax4.get_ylim() == (0.0, 2.0)
    finally:
        plt.close(fig4)


def test_waveloop_orientation(sr):
    # Verify exact angular positioning and clockwise progression
    duration = 1.0
    y = np.zeros(sr)
    # Pulses at 4 cardinal quarter intervals
    y[0] = 1.0
    y[int(0.25 * sr)] = 1.0
    y[int(0.50 * sr)] = 1.0
    y[int(0.75 * sr)] = 1.0

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    assert isinstance(ax, PolarAxes)
    try:
        librosa.display.waveloop(y, sr=sr, duration=duration, ax=ax)
        fig.canvas.draw()
        center = ax.transData.transform([0, 0])

        for frac, exp_cw_deg in [(0.0, 0.0), (0.25, 90.0), (0.50, 180.0), (0.75, 270.0)]:
            t_val = frac * duration
            theta = t_val * 2 * np.pi
            r = 0.75 + 0.25 * 1.0
            disp = ax.transData.transform([theta, r])
            dx = disp[0] - center[0]
            dy = disp[1] - center[1]
            angle_cw = np.degrees(np.arctan2(dx, dy))
            if angle_cw < 0:
                angle_cw += 360
            assert np.isclose(angle_cw, exp_cw_deg, atol=1e-3)
    finally:
        plt.close(fig)


def test_waveloop_decorate_false(y, sr):
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    assert isinstance(ax, PolarAxes)
    try:
        # Default polar axes has direction=1 (CCW) and offset=0 (East)
        librosa.display.waveloop(y, sr=sr, ax=ax, decorate=False)
        assert ax.get_theta_direction() == 1
        assert np.isclose(ax.get_theta_offset(), 0.0)
    finally:
        plt.close(fig)


def test_waveloop_rectilinear_ax_override(y, sr):
    # 1. Explicit rectilinear axes with pre-existing title -> converted in-place + concentric overlay
    fig, ax = plt.subplots()
    assert ax.name == "rectilinear"
    ax.set_title("Pre-existing Title")
    adaptor = librosa.display.waveloop(y, sr=sr, ax=ax)
    assert isinstance(adaptor, librosa.display.AdaptiveWaveplot)
    assert isinstance(ax, PolarAxes)
    assert ax.name == "polar" and ax is adaptor.ax
    assert ax.get_title() == "Pre-existing Title"
    assert ax.get_theta_direction() == -1
    assert np.isclose(ax.get_theta_offset(), np.pi / 2)
    assert len(ax.get_yticks()) == 0

    librosa.display.waveloop(y, sr=sr, r_range=(0.2, 0.5), method="wavebars", n_bars=32, ax=ax)
    fig.canvas.draw()
    assert ax.name == "polar"

    # 2. No ax passed on an empty figure -> creates polar subplot on that figure
    fig2 = plt.figure()
    adaptor2 = librosa.display.waveloop(y, sr=sr)
    assert isinstance(adaptor2, librosa.display.AdaptiveWaveplot)
    assert isinstance(adaptor2.ax, PolarAxes)
    assert adaptor2.ax.figure is fig2
    assert adaptor2.ax in fig2.axes
    fig2.canvas.draw()

    # 3. No ax passed when rectilinear gca() exists -> converts gca() in-place
    fig3, ax3 = plt.subplots()
    assert ax3.name == "rectilinear"
    adaptor3 = librosa.display.waveloop(y, sr=sr)
    assert isinstance(adaptor3, librosa.display.AdaptiveWaveplot)
    assert adaptor3.ax is ax3
    assert adaptor3.ax.figure is fig3
    assert ax3.name == "polar"

    # 4. Shared Cartesian axes are uncoupled when converting
    fig4, (ax_s1, ax_s2) = plt.subplots(1, 2, sharex=True)
    librosa.display.waveloop(y, sr=sr, ax=ax_s1)
    assert isinstance(ax_s1, PolarAxes)
    assert ax_s2.name == "rectilinear"
    fig4.canvas.draw()


def test_waveloop_validation_errors(y, sr):
    sig_dur = len(y) / sr

    # Non-empty rectilinear axes (data, text, inset axes, legend)
    for setup_fn in (
        lambda a: a.plot([1, 2, 3]),
        lambda a: a.text(0.5, 0.5, "annotation"),
        lambda a: a.inset_axes([0.1, 0.1, 0.2, 0.2]),
        lambda a: a.legend([mlines.Line2D([], [])], ["leg"]),
    ):
        fig_e, ax_e = plt.subplots()
        try:
            setup_fn(ax_e)
            with pytest.raises(librosa.ParameterValueError, match="Cannot convert a non-empty axes"):
                librosa.display.waveloop(y, sr=sr, ax=ax_e)
        finally:
            plt.close(fig_e)

    # Non-rectilinear projection (3D) cannot be converted to polar
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection="3d")
    with pytest.raises(librosa.ParameterValueError, match="Cannot convert axes with projection '3d'"):
        librosa.display.waveloop(y, sr=sr, ax=ax_3d)
    plt.close(fig_3d)

    # Empty audio and duration errors
    with pytest.raises(librosa.ParameterValueError, match="empty audio"):
        librosa.display.waveloop(np.array([], dtype=float), sr=sr, duration=1.0)
    with pytest.raises(librosa.ParameterValueError, match="exceeds duration"):
        librosa.display.waveloop(y, sr=sr, duration=sig_dur / 2)
    for bad_dur in (0, -1.5, np.nan, np.inf, -np.inf, True):
        with pytest.raises(librosa.ParameterValueError, match="duration must be finite and strictly positive"):
            librosa.display.waveloop(y, sr=sr, duration=bad_dur)

    # Non-finite / boolean angular & offset scalars
    for bad_val in (np.nan, np.inf, True):
        with pytest.raises(librosa.ParameterValueError, match="offset must be finite"):
            librosa.display.waveloop(y, sr=sr, offset=bad_val)
        with pytest.raises(librosa.ParameterValueError, match="theta_offset must be finite"):
            librosa.display.waveloop(y, sr=sr, theta_offset=bad_val)

    # Invalid r_range
    for bad_r in ("invalid", ("a", "b"), (False, True)):
        with pytest.raises(librosa.ParameterValueError, match="r_range must be a tuple"):
            librosa.display.waveloop(y, sr=sr, r_range=bad_r)  # type: ignore[arg-type]
    for bad_bounds in ((-0.1, 1.0), (1.0, 0.5), (0.5, np.inf)):
        with pytest.raises(librosa.ParameterValueError, match="Must satisfy 0 <= r_min < r_max"):
            librosa.display.waveloop(y, sr=sr, r_range=bad_bounds)

    # Invalid method, direction, time_axis, and unsupported kwargs
    with pytest.raises(librosa.ParameterValueError, match="Invalid method"):
        librosa.display.waveloop(y, sr=sr, method="invalid")  # type: ignore[arg-type]
    for bad_dir in ("diagonal", True):
        with pytest.raises(librosa.ParameterValueError, match="Invalid direction"):
            librosa.display.waveloop(y, sr=sr, direction=bad_dir)  # type: ignore[arg-type]
    for bad_time_axis in ("chroma", "lag", "frames", True, False, 3.5):
        with pytest.raises(librosa.ParameterValueError, match="Unsupported time_axis"):
            librosa.display.waveloop(y, sr=sr, time_axis=bad_time_axis)  # type: ignore[arg-type]

    with pytest.raises(librosa.ParameterValueError, match="transpose=True is not supported"):
        librosa.display.waveloop(y, sr=sr, transpose=True)
    with pytest.raises(librosa.ParameterValueError, match="transform is managed internally"):
        librosa.display.waveloop(y, sr=sr, transform=mtransforms.IdentityTransform())
    with pytest.raises(librosa.ParameterValueError, match="Use 'time_axis' instead of 'axis'"):
        librosa.display.waveloop(y, sr=sr, axis="ms")


def test_loop_locator():
    from librosa.display.formatting import _LoopLocator

    assert _LoopLocator.PREF_DIVISIONS == (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    for n in _LoopLocator.PREF_DIVISIONS:
        loc = _LoopLocator(period=4.0, nbins=n)
        ticks = loc.tick_values(0.0, 4.0)
        assert len(ticks) == n
        assert np.allclose(ticks, np.linspace(0.0, 4.0, n, endpoint=False))

    # Nearest bin selection & explicit divisions (int, sequence, empty, seam)
    assert len(_LoopLocator(period=4.0, nbins=10).tick_values(0.0, 4.0)) in (8, 12)
    assert np.allclose(_LoopLocator(period=6.0, divisions=6).tick_values(0.0, 6.0), np.arange(6, dtype=float))
    custom_times = [0.25, 1.5, 3.75]
    assert np.allclose(_LoopLocator(period=4.0, divisions=custom_times).tick_values(0.0, 4.0), custom_times)
    assert len(_LoopLocator(period=4.0, divisions=[]).tick_values(0.0, 4.0)) == 0
    assert np.allclose(_LoopLocator(period=4.0, divisions=[0.0, 4.0]).tick_values(0.0, 4.0), [0.0])

    # Partial interval / sector preserves upper endpoint; inverted & non-finite bounds work
    loc8 = _LoopLocator(period=4.0, nbins=8)
    assert np.allclose(loc8.tick_values(1.0, 3.0), [1.0, 1.5, 2.0, 2.5, 3.0])
    assert np.allclose(_LoopLocator(period=4.0, divisions=4).tick_values(0.0, 2.0), [0.0, 1.0, 2.0])
    assert len(loc8.tick_values(np.nan, 4.0)) == 0
    assert len(loc8.tick_values(0.0, np.inf)) == 0

    assert np.allclose(loc8.tick_values(4.0, 0.0), np.linspace(0.0, 4.0, 8, endpoint=False))
    assert np.allclose(loc8.tick_values(-1.0, 1.0), [-1.0, -0.5, 0.0, 0.5, 1.0])

    # Calling without axis vs bound to an axis
    assert np.allclose(_LoopLocator(period=4.0, nbins=4)(), [0.0, 1.0, 2.0, 3.0])
    fig, ax = plt.subplots()
    loc8.set_axis(ax.xaxis)
    ax.set_xlim(0.0, 4.0)
    assert len(loc8()) == 8
    plt.close(fig)

    # Constructor validation
    invalid_cases: list[tuple[dict[str, Any], str]] = [
        ({"period": 0.0}, "period must be finite and strictly positive"),
        ({"period": -1.0}, "period must be finite and strictly positive"),
        ({"period": np.nan}, "period must be finite and strictly positive"),
        ({"period": np.inf}, "period must be finite and strictly positive"),
        ({"period": True}, "period must be finite and strictly positive"),
        ({"period": 4.0, "nbins": 0}, "nbins must be a positive integer"),
        ({"period": 4.0, "nbins": True}, "nbins must be a positive integer"),
        ({"period": 4.0, "divisions": True}, "divisions cannot be a boolean"),
        ({"period": 4.0, "divisions": False}, "divisions cannot be a boolean"),
        ({"period": 4.0, "divisions": 0}, "divisions must be positive"),
        ({"period": 4.0, "divisions": -4}, "divisions must be positive"),
        ({"period": 4.0, "divisions": "bad"}, "divisions must be an integer or sequence"),
        ({"period": 4.0, "divisions": ["a", "b"]}, "divisions must be an integer or sequence"),
        ({"period": 4.0, "divisions": [[0.0, 1.0]]}, "divisions must be a 1D sequence of finite numbers"),
        ({"period": 4.0, "divisions": [0.0, np.nan]}, "divisions must be a 1D sequence of finite numbers"),
    ]
    for kwargs_bad, match in invalid_cases:
        with pytest.raises(librosa.ParameterValueError, match=match):
            _LoopLocator(**kwargs_bad)


def test_transformed_tickers(monkeypatch: pytest.MonkeyPatch):
    import matplotlib.scale as mscale

    from librosa.display.formatting import (
        _AxisTransformProxy,
        _LoopLocator,
        _TransformedFormatter,
        _TransformedLocator,
    )

    pre = mtransforms.Affine2D().scale(2 * np.pi / 4.0, 1.0)
    base_loc = mplticker.FixedLocator([0.0, 1.0, 2.0, 3.0])
    trans_loc = _TransformedLocator(base_loc, pre)
    assert np.allclose(trans_loc.tick_values(0.0, 2 * np.pi), [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    assert len(trans_loc()) == 0
    assert len(_TransformedLocator(mplticker.FixedLocator([]), pre).tick_values(0.0, 2 * np.pi)) == 0

    _, ax = plt.subplots()
    base_fmt = librosa.display.TimeFormatter(unit="s")
    trans_fmt = _TransformedFormatter(base_fmt, pre)
    trans_fmt.set_axis(ax.xaxis)
    assert "2" in trans_fmt(np.pi)
    assert len(trans_fmt.format_ticks([0.0, np.pi, 2 * np.pi])) == 3
    assert trans_fmt.format_ticks([]) == []
    assert "2" in trans_fmt.format_data_short(np.pi)

    proxy = _AxisTransformProxy(ax.xaxis, pre)
    assert "_AxisTransformProxy wrapping" in repr(proxy)
    assert proxy.get_scale() == ax.xaxis.get_scale()
    ax.set_xlim(0.0, 2 * np.pi)
    assert np.allclose(proxy.get_view_interval(), [0.0, 4.0])
    monkeypatch.setattr(ax.xaxis, "get_data_interval", lambda: (float("-inf"), float("inf")))
    assert np.allclose(proxy.get_data_interval(), [0.0, 4.0])

    trans_fmt.set_axis(None)
    assert trans_fmt.axis is None
    trans_loc.set_axis(None)
    assert trans_loc.axis is None

    # Y-axis support
    pre_y = mtransforms.Affine2D().scale(1.0, 2.0)
    y_loc = _TransformedLocator(mplticker.FixedLocator([0.5, 1.5]), pre_y)
    y_loc.set_axis(ax.yaxis)
    ax.set_ylim(0.0, 4.0)
    assert np.allclose(y_loc(), [1.0, 3.0])
    y_fmt = _TransformedFormatter(librosa.display.TimeFormatter(unit="s"), pre_y)
    y_fmt.set_axis(ax.yaxis)
    assert "1" in y_fmt(2.0)
    assert "1" in y_fmt.format_ticks([2.0])[0]
    assert "1" in y_fmt.format_data_short(2.0)
    assert np.allclose(_AxisTransformProxy(ax.yaxis, pre_y).get_view_interval(), [0.0, 2.0])

    # 1D transform support (e.g., matplotlib.scale.FuncTransform)
    func_1d = mscale.FuncTransform(
        lambda x: 2.0 * np.asarray(x, dtype=float),
        lambda x: 0.5 * np.asarray(x, dtype=float),
    )
    loc_1d = _TransformedLocator(mplticker.FixedLocator([1.0, 2.0]), func_1d)
    loc_1d.set_axis(ax.xaxis)
    assert np.allclose(loc_1d.tick_values(0.0, 6.0), [2.0, 4.0])
    fmt_1d = _TransformedFormatter(librosa.display.TimeFormatter(unit="s"), func_1d)
    fmt_1d.set_axis(ax.xaxis)
    assert "1" in fmt_1d(2.0)
    assert "2" in fmt_1d.format_ticks([4.0])[0]

    # Validation for non-invertible, singular, non-separable, and >2D transforms
    class Tmp3DTransform(mtransforms.Transform):
        input_dims = 3
        output_dims = 3
        has_inverse = True

    loop_loc = _LoopLocator(period=4.0)
    for bad_trans, match in (
        (mtransforms.Transform(), "transform must be invertible"),
        (mtransforms.Affine2D().scale(0.0, 1.0), "transform must be invertible"),
        (mtransforms.Affine2D().rotate(np.pi / 4), "separable"),
        (Tmp3DTransform(), "1D or separable 2D"),
    ):
        with pytest.raises(librosa.ParameterValueError, match=match):
            _TransformedLocator(loop_loc, bad_trans)
        with pytest.raises(librosa.ParameterValueError, match=match):
            _TransformedFormatter(base_fmt, bad_trans)


def test_periodic_formatter():
    from librosa.display.formatting import _PeriodicFormatter

    base_fmt = librosa.display.TimeFormatter(unit="s")
    for bad_period in (0.0, np.nan, True):
        with pytest.raises(librosa.ParameterValueError, match="period must be finite and strictly positive"):
            _PeriodicFormatter(base_fmt, period=bad_period)

    _, ax = plt.subplots()
    p_fmt = _PeriodicFormatter(base_fmt, period=4.0)
    p_fmt.set_axis(ax.xaxis)
    assert "3" in p_fmt(-1.0)
    assert p_fmt.format_ticks([]) == []
    # Near-upper-boundary wrap snaps cleanly to 0.0s
    assert "0" in p_fmt.format_ticks([-1e-12])[0]
    assert "3" in p_fmt.format_data_short(-1.0)


def test_waveloop_time_axis(y, sr):
    dur = len(y) / sr

    # Default time_axis="time" -> 8 ticks on polar xaxis
    ad = librosa.display.waveloop(y, sr=sr)
    assert isinstance(ad, librosa.display.AdaptiveWaveplot)
    assert isinstance(ad.ax, PolarAxes)
    assert isinstance(ad.ax.xaxis.get_major_locator(), librosa.display.formatting._TransformedLocator)
    assert isinstance(ad.ax.xaxis.get_major_formatter(), librosa.display.formatting._TransformedFormatter)
    ticks = ad.ax.xaxis.get_majorticklocs()
    assert len(ticks) == 8 and np.all(ticks >= 0.0) and np.all(ticks < 2 * np.pi + 1e-4)

    # Integer division count: time_axis=16
    _, ax16 = plt.subplots()
    librosa.display.waveloop(y, sr=sr, time_axis=16, ax=ax16)
    assert len(ax16.xaxis.get_majorticklocs()) == 16

    # Explicit sequences (list and ndarray)
    for seq in ([0.0, 1.0, 2.0], np.array([0.0, 1.0, 2.0])):
        fig_s, ax_s = plt.subplots()
        librosa.display.waveloop(y, sr=sr, time_axis=seq, ax=ax_s)
        ticks_s = ax_s.xaxis.get_majorticklocs()
        assert len(ticks_s) == 3
        assert np.allclose(ticks_s, np.asarray(seq) * (2 * np.pi / dur), atol=1e-3)
        plt.close(fig_s)

    # Hidden ticks: None / "none" / "off"
    for no_time in (None, "none", "off"):
        fig_n, ax_n = plt.subplots()
        librosa.display.waveloop(y, sr=sr, time_axis=no_time, ax=ax_n)
        assert isinstance(ax_n.xaxis.get_major_formatter(), mplticker.NullFormatter)
        assert isinstance(ax_n.xaxis.get_major_locator(), mplticker.NullLocator)
        plt.close(fig_n)

    # Explicit unit: time_axis="ms"
    _, ax_ms = plt.subplots()
    librosa.display.waveloop(y, sr=sr, time_axis="ms", ax=ax_ms)
    labels = [ax_ms.xaxis.get_major_formatter()(t) for t in ax_ms.xaxis.get_majorticklocs()]
    assert any(float(lbl) >= 1000 for lbl in labels)

    # Partial polar sector preserves upper endpoint tick (180 deg)
    _, ax_sec = plt.subplots(subplot_kw={"projection": "polar"})
    assert isinstance(ax_sec, PolarAxes)
    librosa.display.waveloop(y[: int(1.0 * sr)], sr=sr, duration=4.0, time_axis=4, ax=ax_sec)
    ax_sec.set_thetamin(0)
    ax_sec.set_thetamax(180)
    assert np.allclose(np.rad2deg(ax_sec.xaxis.get_major_locator()()), [0.0, 90.0, 180.0])


def test_decorate_loop_time_axis_and_decorate_axis_validation(y, sr):
    _, ax = plt.subplots()
    # _decorate_axis rejects integer or sequence ax_type (restoring pre-PR validation)
    with pytest.raises(librosa.ParameterValueError, match="Unsupported axis type"):
        librosa.display.formatting._decorate_axis(ax.xaxis, ax_type=4)
    with pytest.raises(librosa.ParameterValueError, match="Unsupported axis type"):
        librosa.display.formatting._decorate_axis(ax.xaxis, ax_type=[0.0, 1.0])

    # _decorate_loop_time_axis works on Cartesian axes with and without transform
    librosa.display.formatting._decorate_loop_time_axis(ax.xaxis, "time", period=4.0)
    assert ax.xaxis.get_label_text() == "Time"
    assert isinstance(ax.xaxis.get_major_locator(), librosa.display.formatting._LoopLocator)
    assert isinstance(ax.xaxis.get_major_formatter(), librosa.display.formatting._PeriodicFormatter)

    pre = mtransforms.Affine2D().scale(2.0, 1.0)
    librosa.display.formatting._decorate_loop_time_axis(ax.xaxis, "s", period=2.0, transform=pre)
    assert ax.xaxis.get_label_text() == "Time (seconds)"
    assert isinstance(ax.xaxis.get_major_locator(), librosa.display.formatting._TransformedLocator)
    assert isinstance(ax.xaxis.get_major_formatter(), librosa.display.formatting._TransformedFormatter)

    librosa.display.formatting._decorate_loop_time_axis(ax.xaxis, 8, period=4.0, transform=pre)
    assert ax.xaxis.get_label_text() == "Time"

    librosa.display.formatting._decorate_loop_time_axis(ax.xaxis, None, period=4.0)
    assert ax.xaxis.get_label_text() == ""
    assert len(ax.xaxis.get_major_locator()()) == 0

    for bad_time_axis in ("chroma", "lag", "frames", True, 3.5):
        with pytest.raises(librosa.ParameterValueError, match="Unsupported time_axis"):
            librosa.display.formatting._decorate_loop_time_axis(ax.xaxis, bad_time_axis, period=4.0)  # type: ignore[arg-type]

    # Public callers reject invalid axis arguments instead of silently formatting as Time
    _, ax2 = plt.subplots()
    f0 = np.full(10, 440.0)
    with pytest.raises(librosa.ParameterValueError, match="Unsupported axis type"):
        librosa.display.wavef0(y, f0=f0, sr=sr, freq_axis=4, ax=ax2)  # type: ignore[arg-type]
    with pytest.raises(librosa.ParameterValueError, match="Unsupported axis type"):
        librosa.display.waveshow(y, sr=sr, axis=4, ax=ax2)  # type: ignore[arg-type]


def test_waveloop_offset(y, sr):
    dur = 4.0
    y_short = y[: int(2.0 * sr)]

    # Loop ticks always start at 0 (at theta_offset), while offset shifts the waveform in seconds
    adaptor = librosa.display.waveloop(y_short, sr=sr, offset=1.5, duration=dur, time_axis=8)
    assert isinstance(adaptor, librosa.display.AdaptiveWaveplot)
    ax = adaptor.ax
    assert isinstance(ax, PolarAxes)
    ticks = ax.xaxis.get_majorticklocs()
    assert len(ticks) == 8
    assert np.isclose(ticks[0], 0.0)
    assert "0" in ax.xaxis.get_major_formatter()(ticks[0])
    assert np.isclose(ax.dataLim.x0, 2 * np.pi * 1.5 / dur, atol=1e-2)
    assert np.isclose(ax.dataLim.x1, 2 * np.pi * 3.5 / dur, atol=1e-2)

    # Full-duration signal shifted by positive or negative offset wraps cleanly across 0
    for shift in (1.0, -0.5):
        fig_w, ax_w = plt.subplots(subplot_kw={"projection": "polar"})
        librosa.display.waveloop(y, sr=sr, offset=shift, ax=ax_w)
        assert isinstance(ax_w, PolarAxes)
        fig_w.canvas.draw()
        assert np.isclose(ax_w.get_thetamin(), 0.0)
        assert np.isclose(ax_w.get_thetamax(), 360.0)
        plt.close(fig_w)

    # Symmetric sector [-90, 90] deg wraps negative times cleanly via _PeriodicFormatter
    _, ax_sym = plt.subplots(subplot_kw={"projection": "polar"})
    librosa.display.waveloop(y_short, sr=sr, duration=dur, time_axis=4, ax=ax_sym)
    assert isinstance(ax_sym, PolarAxes)
    ax_sym.set_thetamin(-90)
    ax_sym.set_thetamax(90)
    labels = [ax_sym.xaxis.get_major_formatter()(t) for t in ax_sym.xaxis.get_majorticklocs()]
    assert any("3" in lbl for lbl in labels)


def test_extract_pre_transform_branches():
    fig, ax = plt.subplots()
    # 1. transform is None / transform == ax.transData
    pre, art = librosa.display.signal._extract_pre_transform(None, ax)
    assert isinstance(pre, mtransforms.IdentityTransform) and art is None
    pre, art = librosa.display.signal._extract_pre_transform(ax.transData, ax)
    assert isinstance(pre, mtransforms.IdentityTransform) and art == ax.transData

    # 2. Left-nested, right-nested, and pre + transData + post composites
    t1 = mtransforms.Affine2D().scale(2.0)
    t2 = mtransforms.Affine2D().translate(1.0, 0.0)
    expected = (t1 + t2).transform([[1.0, 0.0]])
    for chained in ((t1 + t2) + ax.transData, t1 + (t2 + ax.transData)):
        pre_c, art_c = librosa.display.signal._extract_pre_transform(chained, ax)
        assert pre_c is not None and art_c == chained
        assert np.allclose(pre_c.transform([[1.0, 0.0]]), expected)

    post = ax.transData + mtransforms.ScaledTranslation(5 / 72, 5 / 72, fig.dpi_scale_trans)
    pre_post, art_post = librosa.display.signal._extract_pre_transform(t1 + post, ax)
    assert pre_post is t1 and art_post is not None

    # 3. Non-data transforms (transAxes, blended) and Transformf0
    for non_data in (ax.transAxes, ax.get_xaxis_transform()):
        pre_nd, art_nd = librosa.display.signal._extract_pre_transform(non_data, ax)
        assert pre_nd is None and art_nd is non_data

    tf0 = librosa.display.Transformf0(f0=np.array([220.0, 440.0]), sr=1, hop_length=1)
    chained_f0 = tf0 + ax.transData
    pre_f0, art_f0 = librosa.display.signal._extract_pre_transform(chained_f0, ax)
    assert pre_f0 is tf0 and art_f0 is chained_f0

    # 4. PolarAxes composite transData stops at polar_ax.transData by identity
    _, ax_p = plt.subplots(subplot_kw={"projection": "polar"})
    pre_in = mtransforms.Affine2D().scale(2.0, 0.5)
    chained_p = pre_in + ax_p.transData
    pre_p, art_p = librosa.display.signal._extract_pre_transform(chained_p, ax_p)
    assert pre_p is pre_in and art_p is chained_p


def test_waveshow_custom_transform_transpose_and_mask(y, sr):
    _, ax = plt.subplots()
    pre = mtransforms.Affine2D().scale(2.0, 1.0)
    mask = np.ones(len(y), dtype=bool)
    mask[::3] = False
    adaptor = librosa.display.waveshow(y, sr=sr, transform=pre + ax.transData, transpose=True, mask=mask, ax=ax)
    assert isinstance(adaptor, librosa.display.AdaptiveWaveplot)
    assert adaptor.ax is ax


def test_adaptive_waveplot_out_of_bounds_viewport(y, sr):
    adaptor = librosa.display.waveshow(y, sr=sr)
    ax = adaptor.ax
    assert isinstance(ax, Axes)
    ax.set_xlim(1000.0, 2000.0)
    adaptor.update(ax)
    ax.set_xlim(2.0, 1.0)
    adaptor.update(ax)


def test_adaptive_waveplot_handler_expired_ref():
    class TmpParent:
        pass

    tmp = TmpParent()
    decoy = librosa.display.signal._WaveplotDecoy(cast("Any", tmp))
    del tmp
    handler = librosa.display.signal._AdaptiveWaveplotHandler()
    res = handler.create_artists(
        cast("Any", None),
        decoy,
        0.0,
        0.0,
        10.0,
        10.0,
        12.0,
        mtransforms.IdentityTransform(),
    )
    assert res == []


@pytest.mark.mpl_image_compare(
    baseline_images=["wavef0"],
    extensions=["png"],
    style=STYLE,
)
def test_wavef0(y, sr):
    fig, ax = plt.subplots(nrows=2, figsize=(8, 6))

    f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), sr=sr)
    # Waveform with f0
    librosa.display.wavef0(y, sr=sr, f0=f0, ax=ax[0], label="waveshow")
    ax[0].legend(loc="lower right")

    # Waveform with f0 and pitch
    librosa.display.wavef0(
        y, sr=sr, f0=f0, method="wavebars", freq_axis="cqt_hz", color="C1", ax=ax[1], label="wavebars"
    )
    ax[1].legend(loc="lower right")
    return fig


def test_legend_for_axes_no_axes():
    fig = plt.figure()
    try:
        with pytest.raises(librosa.ParameterValueError):
            librosa.display.legend_for_axes(fig=fig)
    finally:
        plt.close(fig)


def test_legend_for_axes_current():
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(np.arange(4), label="line")

    leg = librosa.display.legend_for_axes()
    assert leg is not None
    assert leg.figure is fig
    plt.close(fig)


def test_legend_for_axes_scalar():
    fig, ax = plt.subplots()
    ax.plot(np.arange(4), label="line")
    leg = librosa.display.legend_for_axes(axes=ax)
    assert leg is not None
    assert leg.figure is fig
    plt.close(fig)


def test_legend_for_axes_mismatched_figures():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    try:
        with pytest.raises(librosa.ParameterValueError):
            librosa.display.legend_for_axes([ax1, ax2])
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_legend_for_axes_explicit_bbox():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")

    leg = librosa.display.legend_for_axes([ax], loc="center", bbox_to_anchor=(0.5, 0.5, 0.2, 0.2))

    assert leg is not None
    plt.close(fig)


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_right"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_right():
    fig, axes = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(right=0.8)
    librosa.display.legend_for_axes(axes=axes, loc="outside center right")

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["wavef0_transpose"],
    extensions=["png"],
    style=STYLE,
)
def test_wavef0_transpose(y, sr):
    fig, ax = plt.subplots(ncols=2, figsize=(6, 8))

    f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), sr=sr)
    # Waveform with f0
    librosa.display.wavef0(y, sr=sr, f0=f0, ax=ax[0], label="waveshow", transpose=True)
    ax[0].legend(loc="lower right")

    # Waveform with f0 and pitch
    librosa.display.wavef0(
        y, sr=sr, f0=f0, method="wavebars", freq_axis="cqt_hz", color="C1", ax=ax[1], label="wavebars", transpose=True
    )
    ax[1].legend(loc="lower right")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_left"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_left():
    fig, axes = plt.subplots(nrows=2, figsize=(8, 4), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    fig.subplots_adjust(left=0.25)
    librosa.display.legend_for_axes(axes=axes, loc="outside center left")

    return fig


def test_wavef0_bad_method(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), sr=sr)
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.wavef0(y, f0=f0, sr=sr, method="bad_method")


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


@pytest.mark.parametrize("f0", [np.zeros(5), np.linspace(-10, -5, num=5), np.linspace(-10, 10, num=5)])
def test_transformf0_bad_f0(f0):
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.Transformf0(f0=f0, sr=1, hop_length=1)


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_above"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_above():
    fig, axes = plt.subplots(ncols=3, figsize=(8, 4), sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    librosa.display.legend_for_axes(axes=axes, loc="outside upper center", ncol=3)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_below"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_below():
    fig, axes = plt.subplots(ncols=3, figsize=(8, 4), sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    librosa.display.legend_for_axes(axes=axes, loc="outside lower center", ncol=3)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_1d"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_default_1d():
    fig, axes = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    librosa.display.legend_for_axes(axes=axes)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_row"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_default_row():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    axes = np.asarray(axes)[np.newaxis, :]
    librosa.display.legend_for_axes(axes=axes)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_col"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_default_col():
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

    axes = np.asarray(axes)[:, np.newaxis]
    librosa.display.legend_for_axes(axes=axes, loc="outside center right")

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["legend_for_axes_default_grid"],
    extensions=["png"],
    style=STYLE,
)
def test_legend_for_axes_default_grid():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharex=True, sharey=True)

    for i, ax in enumerate(np.ravel(axes)):
        ax.plot([0, 1], [i, i + 1], label=f"line-{i}", color=f"C{i}")

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


def test_resolve_multiplot_bad():
    with pytest.raises(librosa.ParameterValueError):
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
    axshape_out, nrows_out, ncols_out, multi_input_out = librosa.display._mp_get_layout(data, dims, orient)

    assert axshape_out == axshape
    assert nrows_out == nrows
    assert ncols_out == ncols
    assert multi_input_out is multi_input


def test_mp_get_layout_bad_orient():
    with pytest.raises(librosa.ParameterValueError):
        librosa.display._mp_get_layout((np.zeros((3, 10)),), dims=1, orient="q")  # type: ignore


def test_mp_get_layout_no_data():
    with pytest.raises(librosa.ParameterValueError):
        librosa.display._mp_get_layout(tuple(), dims=1, orient="v")


def test_mp_get_layout_bad_axshape():
    with pytest.raises(librosa.ParameterShapeError):
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
    plt.close(fig)  # type: ignore


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
    plt.close(fig)


@pytest.mark.parametrize(
    "orient,axes_in_shape,axes_out_shape,output_shape",
    [
        ("v", (3,), (3, 1), (3,)),
        ("h", (3,), (1, 3), (3,)),
        ("v", (2, 2), (2, 2), (2, 2)),
    ],
)
def test_mp_setup_axes_array_input(orient, axes_in_shape, axes_out_shape, output_shape):
    fig, axes_in = plt.subplots(*axes_in_shape) if len(axes_in_shape) == 2 else plt.subplots(axes_in_shape[0])

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
    plt.close(fig)  # type: ignore


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
    plt.close(fig)


def test_mp_setup_axes_bad_shape():
    fig, axes = plt.subplots(nrows=2)
    try:
        with pytest.raises(librosa.ParameterShapeError):
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
    finally:
        plt.close(fig)


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
    plt.close(fig)


def test_mp_setup_labels_none():
    labels = librosa.display._mp_setup_labels(None, (2, 2))

    assert labels.shape == (2, 2)
    assert labels.dtype == object
    # Comparison to None here is vectorized, so == is correct, not `is`.
    assert np.all(labels == None)  # noqa: E711


def test_mp_setup_labels_values():
    labels = librosa.display._mp_setup_labels(["a", "b", "c"], (3,))

    assert np.array_equal(labels, np.asarray(["a", "b", "c"], dtype=object))


def test_mp_setup_labels_bad_shape():
    with pytest.raises(ValueError):
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


def test_mp_setup_prop_group_bad_shape():
    with pytest.raises(librosa.ParameterShapeError):
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
        assert isinstance(prop, dict)
        assert "color" not in prop


@pytest.mark.mpl_image_compare(
    baseline_images=["multiplot_wave_constructed_axes_stacked"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
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
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
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
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
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
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_multiplot_img_existing_axes_variadic(y):
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
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
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


@pytest.mark.mpl_image_compare(
    baseline_images=["multiplot_waveloop"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_multiplot_waveloop_image(y, sr):
    y_stack = np.stack([y, y[::-1]])
    librosa.display.multiplot(
        "waveloop",
        y_stack,
        sr=sr,
        titles=["Track 1", "Track 2"],
    )
    return plt.gcf()


def test_multiplot_waveloop(y, sr):
    y_stack = np.stack([y, y[::-1]])

    # Constructed polar subplots
    out = librosa.display.multiplot(
        "waveloop",
        y_stack,
        sr=sr,
        titles=["Track 1", "Track 2"],
    )
    fig = plt.gcf()
    try:
        assert out.shape == (2,)
        assert len(fig.axes) == 2
        for ax in fig.axes:
            assert isinstance(ax, PolarAxes)
            assert ax.name == "polar"
            assert ax.get_theta_direction() == -1
            assert np.isclose(ax.get_theta_offset(), np.pi / 2)
            assert len(ax.get_yticks()) == 0
        assert fig.axes[0].get_title() == "Track 1"
        assert fig.axes[1].get_title() == "Track 2"
        fig.canvas.draw()
    finally:
        plt.close(fig)

    # With method="wavebars"
    out_bars = librosa.display.multiplot(
        "waveloop",
        y_stack,
        sr=sr,
        method="wavebars",
        n_bars=64,
    )
    fig_bars = plt.gcf()
    try:
        assert out_bars.shape == (2,)
        for coll in out_bars:
            assert isinstance(coll, matplotlib.collections.PatchCollection)
        fig_bars.canvas.draw()
    finally:
        plt.close(fig_bars)

    # Existing polar axes
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))
    try:
        out_exist = librosa.display.multiplot(
            "waveloop",
            y_stack,
            axes=axes,
            sr=sr,
        )
        assert out_exist.shape == axes.shape
        fig.canvas.draw()
    finally:
        plt.close(fig)

    # Existing rectilinear axes converted automatically
    fig, axes_rect = plt.subplots(1, 2)
    out_rect = librosa.display.multiplot(
        "waveloop",
        y_stack,
        axes=axes_rect,
        sr=sr,
    )
    assert out_rect.shape == axes_rect.shape
    assert all(a.name == "polar" for a in axes_rect)
    fig.canvas.draw()


def test_multiplot_waveloop_fig_kw(y, sr):
    y_stack = np.stack([y, y[::-1]])
    out = librosa.display.multiplot("waveloop", y_stack, sr=sr, fig_kw={"figsize": (6, 6)})
    fig = plt.gcf()
    assert out.shape == (2,)
    assert np.allclose(fig.get_size_inches(), (6, 6))

    # User subplot_kw merges with polar projection default without mutating input dicts
    subplot_kw_orig = {"facecolor": "0.9"}
    out_merged = librosa.display.multiplot(
        "waveloop",
        y_stack,
        sr=sr,
        fig_kw={"figsize": (6, 6), "subplot_kw": subplot_kw_orig},
    )
    fig_merged = plt.gcf()
    assert out_merged.shape == (2,)
    assert isinstance(fig_merged.axes[0], PolarAxes)
    assert "projection" not in subplot_kw_orig

    # Existing fig + subplot-level fig_kw works; figure-level fig_kw raises TypeError
    fig_custom = plt.figure(figsize=(7, 7))
    sub_kw_existing = {"facecolor": "0.9"}
    out2 = librosa.display.multiplot(
        "waveloop",
        y_stack,
        fig=fig_custom,
        fig_kw={"subplot_kw": sub_kw_existing},
        sr=sr,
    )
    assert out2.shape == (2,)
    assert isinstance(fig_custom.axes[0], PolarAxes)
    assert matplotlib.colors.same_color(fig_custom.axes[0].get_facecolor(), "0.9")
    assert "projection" not in sub_kw_existing
    assert np.allclose(fig_custom.get_size_inches(), (7, 7))

    with pytest.raises(TypeError):
        librosa.display.multiplot("waveloop", y_stack, fig=fig_custom, fig_kw={"figsize": (5, 5)}, sr=sr)

    fig_cart = plt.figure(figsize=(6, 6))
    out_cart = librosa.display.multiplot(
        "waveshow",
        y_stack,
        fig=fig_cart,
        fig_kw={"gridspec_kw": {"hspace": 0.5}},
        sr=sr,
    )
    assert out_cart.shape == (2,)


@pytest.mark.mpl_image_compare(
    baseline_images=["highlight_spectrum"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_highlight_spectrum(S_abs, y, sr):

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    f0, _, _ = librosa.pyin(y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), sr=sr)
    times = librosa.times_like(f0, sr=sr)
    # Column 1: dark color spectrogram
    # Column 2: light color spectrogram
    # Row 1: auto highlight
    # Row 2: auto highlight with specified colors
    # Row 3: explicit highlight

    for i in range(3):
        librosa.display.specshow(S_abs, x_axis="time", y_axis="log", cmap="magma", vscale="dBFS", ax=ax[i, 0])
        librosa.display.specshow(S_abs, x_axis="time", y_axis="log", cmap="magma_r", vscale="dBFS", ax=ax[i, 1])

    for j in range(2):
        hl1 = librosa.display.highlight(ax=ax[0, j], linewidth=5, alpha=1.0)
        ax[0, j].plot(times, f0, path_effects=hl1)

        hl2 = librosa.display.highlight(ax=ax[1, j], linewidth=5, alpha=1.0, bright_color="green", dark_color="pink")
        ax[1, j].plot(times, f0, path_effects=hl2)

        hl3 = librosa.display.highlight(ax=ax[2, j], linewidth=5, alpha=1.0, color="yellow")
        ax[2, j].plot(times, f0, path_effects=hl3)

    for axi in ax.flat:
        axi.label_outer()

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["highlight_axes"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_highlight_axescolor():

    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)

    x = np.linspace(0, 5, num=100)
    y = x * np.exp(-(x**2))

    # First axes has light face color, do nothing
    # Second axes has a dark face color
    ax[1].set_facecolor((0.1, 0.1, 0.1))

    # Third axes has a dark but transparent facecolor
    ax[2].set_facecolor((0, 0, 0, 0))

    hl0 = librosa.display.highlight(ax=ax[0], linewidth=5)
    ax[0].plot(x, y, path_effects=hl0)
    hl1 = librosa.display.highlight(ax=ax[1], linewidth=5)
    ax[1].plot(x, y, path_effects=hl1)
    hl2 = librosa.display.highlight(ax=ax[2], linewidth=5)
    ax[2].plot(x, y, path_effects=hl2)

    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["highlight_artist"],
    extensions=["png"],
    style=STYLE,
)
def test_highlight_artist():

    fig, ax = plt.subplots()
    x = np.linspace(0, 5, num=100)
    y = x * np.exp(-(x**2))

    hl = librosa.display.highlight(ax=ax, linewidth=5)
    ax.plot(x, y, path_effects=hl, label="path_effects=")

    lines2 = ax.plot(x, 1 + y, label="highlight(artist)")[0]
    librosa.display.highlight(artist=lines2, linewidth=5)

    hl2 = librosa.display.highlight(linewidth=5)
    ax.plot(x, 2 + y, path_effects=hl2, label="path_effects= + gca")
    ax.legend(loc="upper right")
    return fig


@pytest.mark.mpl_image_compare(
    baseline_images=["boolean_norm"],
    extensions=["png"],
    style=STYLE,
)
@pytest.mark.usefixtures("text_placeholders")
def test_specshow_boolean_norm(S_abs):

    fig, ax = plt.subplots(nrows=4, sharex=True, sharey=True)

    # First draw a sequential image with the same colormap as
    # the boolean default.
    # This case hits norm_cmap is None (vscale=None)
    # and cmap is not provided, so we have to infer
    i1 = librosa.display.specshow(
        librosa.amplitude_to_db(S_abs),
        y_axis="log_oct3",
        x_axis="time",
        ax=ax[0],
        cmap_seq="gray_r",
        cmap_bool="gray_r",
    )
    fig.colorbar(i1)

    # Now do the same thing, but explicitly using the same
    # colormap objects for seq and bool
    cmap = mcm["Reds"]
    i2 = librosa.display.specshow(
        librosa.amplitude_to_db(S_abs),
        y_axis="log_oct3",
        x_axis="time",
        ax=ax[1],
        cmap_seq=cmap,
        cmap_bool=cmap,
    )
    fig.colorbar(i2)

    # Now use binary data for input
    i3 = librosa.display.specshow(
        S_abs > 0.5,
        y_axis="log_oct3",
        x_axis="time",
        ax=ax[2],
    )
    fig.colorbar(i3)

    # And finally test the override case
    i4 = librosa.display.specshow(
        S_abs > 0.5,
        y_axis="log_oct3",
        x_axis="time",
        ax=ax[3],
        cmap=None,
        cmap_seq="gray_r",
        cmap_bool="gray_r",
    )
    fig.colorbar(i4)

    for axi in ax.flat:
        axi.label_outer()
    return fig


def test_wavef0_validation_method(y, sr):
    fig, ax = plt.subplots()
    f0 = np.array([440.0, 440.0])
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.wavef0(y, sr=sr, f0=f0, method="invalid", ax=ax)
    plt.close(fig)


def test_wavef0_empty_audio(sr):
    fig, ax = plt.subplots()
    f0 = np.array([440.0])
    with pytest.raises(librosa.ParameterValueError):
        librosa.display.wavef0(np.empty(0), sr=sr, f0=f0, ax=ax)
    plt.close(fig)


def test_wavef0_methods(y, sr):
    fig, ax = plt.subplots()
    f0 = np.array([440.0, 440.0])
    librosa.display.wavef0(y, sr=sr, f0=f0, method="waveshow", ax=ax)
    librosa.display.wavef0(y, sr=sr, f0=f0, method="wavebars", ax=ax)
    plt.close(fig)


def test_wavef0_orientation(y, sr):
    fig, ax = plt.subplots()
    f0 = np.array([440.0, 440.0])
    librosa.display.wavef0(y, sr=sr, f0=f0, transpose=False, ax=ax)
    librosa.display.wavef0(y, sr=sr, f0=f0, transpose=True, ax=ax)
    plt.close(fig)


def test_wavef0_spectrogram_overlay(y, sr):
    fig, ax = plt.subplots()
    f0 = np.array([440.0, 440.0])
    ax.plot([0, 1], [0, 1])
    librosa.display.wavef0(y, sr=sr, f0=f0, ax=ax)
    plt.close(fig)


def test_adaptive_formatter_base_not_implemented():
    from librosa.display.formatting import AdaptiveFormatterBase

    formatter = AdaptiveFormatterBase()
    with pytest.raises(librosa.LibrosaNotImplementedError):
        formatter._format_tick(0.0)


def test_chroma_svara_formatter_default_sa():
    csf = librosa.display.formatting.ChromaSvaraFormatter(Sa=None)
    assert csf.Sa == 0
