#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa core (__init__.py)
#

from __future__ import print_function

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import sys
import soundfile
import audioread.rawread
import librosa
import librosa.core
import librosa.core.spectrum
import glob
import numpy as np
import scipy.io
import scipy.signal
import pytest
import warnings
from unittest import mock
from typing import Any, Callable, Union, cast


# -- utilities --#
def files(pattern):
    test_files = glob.glob(pattern)
    test_files.sort()
    return test_files


def srand(seed=628318530):
    np.random.seed(seed)
    pass


def load(infile):
    return scipy.io.loadmat(infile, chars_as_strings=True)


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "core-load-*.mat"))
)
def test_load(infile):
    DATA = load(infile)
    y, sr = librosa.load(
        os.path.join("tests", DATA["wavfile"][0]), sr=None, mono=DATA["mono"]
    )

    # Verify that the sample rate is correct
    assert sr == DATA["sr"]

    assert np.allclose(y, DATA["y"])


def test_load_soundfile():

    fname = os.path.join("tests", "data", "test1_44100.wav")
    # Load from filename
    y, sr = librosa.load(fname, sr=None, mono=False)

    # Load from soundfile object

    sfo = soundfile.SoundFile(fname)
    y2, sr2 = librosa.load(sfo, sr=None, mono=False)

    assert np.allclose(y, y2)
    assert np.isclose(sr, sr2)


@pytest.mark.filterwarnings("ignore:librosa.core.audio.__audioread_load")
def test_load_audioread():
    fname = os.path.join("tests", "data", "test1_44100.wav")

    # Load using an existing audioread object
    reader = audioread.rawread.RawAudioFile(fname)
    y, sr = librosa.load(reader, sr=None)

    # Load using sndfile
    y2, sr2 = librosa.load(fname, sr=None)

    assert np.allclose(y, y2)
    assert np.isclose(sr, sr2)


@pytest.mark.parametrize("res_type", ["soxr_qq", "soxr_hq", "scipy"])
def test_load_resample(res_type):

    sr_target = 16000
    fn = os.path.join("tests", "data", "test1_44100.wav")

    y_native, sr = librosa.load(fn, sr=None, res_type=res_type)

    y2 = librosa.resample(y_native, orig_sr=sr, target_sr=sr_target, res_type=res_type)

    y, _ = librosa.load(fn, sr=sr_target, res_type=res_type)

    assert np.allclose(y2, y)


def test_segment_load():

    sample_len = 2003
    fs = 44100
    test_file = os.path.join("tests", "data", "test1_44100.wav")
    y, sr = librosa.load(
        test_file, sr=None, mono=False, offset=0.0, duration=sample_len / float(fs)
    )

    assert y.shape[-1] == sample_len

    y2, sr = librosa.load(test_file, sr=None, mono=False)
    assert np.allclose(y, y2[:, :sample_len])

    sample_offset = 2048
    y, sr = librosa.load(
        test_file, sr=None, mono=False, offset=sample_offset / float(fs), duration=1.0
    )

    assert y.shape[-1] == fs

    y2, sr = librosa.load(test_file, sr=None, mono=False)
    assert np.allclose(y, y2[:, sample_offset : sample_offset + fs])


@pytest.fixture(
    scope="module", params=["test1_44100.wav", "test1_22050.wav", "test2_8000.wav"]
)
def resample_audio(request):
    infile = request.param
    y, sr_in = librosa.load(
        os.path.join("tests", "data", infile), sr=None, duration=5, mono=False
    )
    return (y, sr_in)


@pytest.fixture(scope="module")
def resample_mono(resample_audio):
    y, sr = resample_audio
    y = librosa.to_mono(y)
    return (y, sr)


@pytest.mark.parametrize("sr_out", [8000, 22050])
@pytest.mark.parametrize(
    "res_type",
    [
        "kaiser_best",
        "kaiser_fast",
        "scipy",
        "fft",
        "polyphase",
        "linear",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
        "zero_order_hold",
        "soxr_qq",
        "soxr_lq",
        "soxr_mq",
        "soxr_hq",
        "soxr_vhq",
    ],
)
@pytest.mark.parametrize("fix", [False, True])
def test_resample_mono(resample_mono, sr_out, res_type, fix):

    y, sr_in = resample_mono
    y = librosa.to_mono(y)

    y2 = librosa.resample(
        y, orig_sr=sr_in, target_sr=sr_out, res_type=res_type, fix=fix
    )

    # First, check that the audio is valid
    librosa.util.valid_audio(y2)

    # If it's a no-op, make sure the signal is untouched
    if sr_out == sr_in:
        assert np.allclose(y, y2)

    # Check buffer contiguity
    assert y2.flags["C_CONTIGUOUS"] == y.flags["C_CONTIGUOUS"]
    assert y2.flags["F_CONTIGUOUS"] == y.flags["F_CONTIGUOUS"]

    # Check that we're within one sample of the target length
    target_length = y.shape[-1] * sr_out // sr_in
    assert np.abs(y2.shape[-1] - target_length) <= 1


@pytest.mark.parametrize("sr_out", [8000, 22050])
@pytest.mark.parametrize(
    "res_type",
    [
        "kaiser_best",
        "kaiser_fast",
        "scipy",
        "fft",
        "polyphase",
        "linear",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
        "zero_order_hold",
        "soxr_qq",
        "soxr_lq",
        "soxr_mq",
        "soxr_hq",
        "soxr_vhq",
    ],
)
@pytest.mark.parametrize("fix", [False, True])
def test_resample_stereo(resample_audio, sr_out, res_type, fix):

    y, sr_in = resample_audio

    y2 = librosa.resample(
        y, orig_sr=sr_in, target_sr=sr_out, res_type=res_type, fix=fix
    )

    # First, check that the audio is valid
    librosa.util.valid_audio(y2)

    assert y2.ndim == y.ndim

    # If it's a no-op, make sure the signal is untouched
    if sr_out == sr_in:
        assert np.allclose(y, y2)

    # Check that we're within one sample of target_length = y.shape[-1] * sr_out // sr_in
    target_length = y.shape[-1] * sr_out // sr_in
    assert np.abs(y2.shape[-1] - target_length) <= 1


@pytest.mark.parametrize(
    "res_type",
    [
        "fft",
        "kaiser_best",
        "kaiser_fast",
        "polyphase",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
        "soxr_lq",
        "soxr_mq",
        "soxr_hq",
        "soxr_vhq",
    ],
)
@pytest.mark.parametrize("sr_out", [11025, 22050, 44100])
def test_resample_scale(resample_mono, res_type, sr_out):

    y, sr_in = resample_mono

    y2 = librosa.resample(
        y, orig_sr=sr_in, target_sr=sr_out, res_type=res_type, scale=True
    )

    # First, check that the audio is valid
    librosa.util.valid_audio(y2)

    n_orig = np.sqrt(np.sum(np.abs(y) ** 2))
    n_res = np.sqrt(np.sum(np.abs(y2) ** 2))

    # If it's a no-op, make sure the signal is untouched
    assert np.allclose(n_orig, n_res, atol=1e-2), (n_orig, n_res)


@pytest.mark.parametrize("sr_in, sr_out", [(100, 100.1), (100.1, 100)])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_resample_poly_float(sr_in, sr_out):
    y = np.zeros(128)
    librosa.resample(y, orig_sr=sr_in, target_sr=sr_out, res_type="polyphase")


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "core-stft-*.mat"))
)
def test_stft(infile):

    DATA = load(infile)

    # Load the file
    (y, sr) = librosa.load(
        os.path.join("tests", DATA["wavfile"][0]), sr=None, mono=True
    )

    window: Union[Callable[..., Any], str]
    if DATA["hann_w"][0, 0] == 0:
        # Set window to ones, swap back to nfft
        window = np.ones
        win_length = None

    else:
        window = "hann"
        win_length = DATA["hann_w"][0, 0]

    # Compute the STFT
    D = librosa.stft(
        y,
        n_fft=DATA["nfft"][0, 0].astype(int),
        hop_length=DATA["hop_length"][0, 0].astype(int),
        win_length=win_length,
        window=window,
        center=False,
    )

    # conjugate matlab stft to fix the ' vs .' bug
    assert np.allclose(D, DATA["D"].conj())


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stft_toolong_left():
    y = np.zeros((128,))
    librosa.stft(y, n_fft=2048, center=False)


def test_stft_toolong_center():
    y = np.zeros((128,))
    with pytest.warns(UserWarning):
        librosa.stft(y, n_fft=2048, center=True)


def test_stft_winsizes():
    # Test for issue #1095
    x = np.zeros(1000000)

    for power in range(12, 17):
        N = 2**power
        H = N // 2
        librosa.stft(x, n_fft=N, hop_length=H, win_length=N)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize(
    "n_fft, hop_length",
    [(1023, 128), (1023, 129), (1023, 256), (2048, 512), (2048, 2048)],
)
@pytest.mark.parametrize("N", [1024, 2048, 8192])
def test_stft_preallocate(center, n_fft, hop_length, N):

    # Work in stereo by default
    y = np.random.randn(2, max(N, n_fft))

    D1 = librosa.stft(y, center=center, n_fft=n_fft, hop_length=hop_length)
    out = np.empty_like(D1)
    D2 = librosa.stft(y, center=center, n_fft=n_fft, hop_length=hop_length, out=out)
    assert D2 is out
    assert np.allclose(D1, D2)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize(
    "n_fft, hop_length",
    [(1023, 128), (1023, 129), (1023, 256), (2048, 512), (2048, 2048)],
)
@pytest.mark.parametrize("N", [2048])
def test_stft_preallocate_oversize(center, n_fft, hop_length, N):

    # Work in stereo by default
    y = np.random.randn(2, max(N, n_fft))

    D1 = librosa.stft(y, center=center, n_fft=n_fft, hop_length=hop_length)
    shape = list(D1.shape)
    shape[-1] *= 2
    out = np.empty_like(D1, shape=shape)
    D2 = librosa.stft(y, center=center, n_fft=n_fft, hop_length=hop_length, out=out)
    assert np.allclose(D1, D2)
    assert np.allclose(D1, out[..., : D2.shape[-1]])


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize(
    "n_fft, hop_length",
    [(1023, 128), (1023, 129), (1023, 256), (2048, 512), (2048, 2048)],
)
@pytest.mark.parametrize("N", [2048])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stft_preallocate_undersize(center, n_fft, hop_length, N):

    # Work in stereo by default
    y = np.random.randn(2, max(N, n_fft))

    D1 = librosa.stft(y, center=center, n_fft=n_fft, hop_length=hop_length)
    shape = list(D1.shape)
    shape[-1] //= 2
    out = np.empty_like(D1, shape=shape)
    D2 = librosa.stft(y, center=center, n_fft=n_fft, hop_length=hop_length, out=out)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize(
    "n_fft, hop_length",
    [(1023, 128), (1023, 129), (1023, 256), (2048, 512), (2048, 2048)],
)
@pytest.mark.parametrize("N", [1024, 2048, 8192])
def test_istft_preallocate(center, n_fft, hop_length, N):
    y = np.random.randn(2, max(N, n_fft))

    D = librosa.stft(y, center=center, n_fft=n_fft, hop_length=hop_length)

    y1 = librosa.istft(D, center=center, n_fft=n_fft, hop_length=hop_length)
    y2 = np.empty_like(y1)
    y3 = librosa.istft(D, center=center, n_fft=n_fft, hop_length=hop_length, out=y2)

    assert y3 is y2
    assert np.allclose(y1, y2)


# results for FFT bins containing multiple components will be unstable, as when
# using higher sampling rates or shorter windows with this test signal
@pytest.mark.parametrize("center", [False])
@pytest.mark.parametrize("sr", [256, 512, 2000, 2048])
@pytest.mark.parametrize("n_fft", [128, 255, 256, 512, 1280])
def test___reassign_frequencies(sr, n_fft, center):
    x = np.linspace(0, 5, 5 * sr, endpoint=False)
    y = np.sin(17 * x * 2 * np.pi) + np.sin(103 * x * 2 * np.pi)
    freqs, S = librosa.core.spectrum.__reassign_frequencies(
        y=y, sr=sr, n_fft=n_fft, hop_length=n_fft, center=center
    )

    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # frequencies should be reassigned to the closest component within 3 Hz
    # ignore reassigned estimates with low support
    lower = freqs[(freqs > 0) & (freqs < 60) & (S_db > -30)]
    assert np.allclose(lower, 17, atol=3)

    upper = freqs[(freqs >= 60) & (freqs < sr // 2) & (S_db > -30)]
    assert np.allclose(upper, 103, atol=3)


# regression tests originally for `ifgram`
@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "core-ifgram-*.mat"))
)
def test___reassign_frequencies_regress(infile):
    DATA = load(infile)

    y, sr = librosa.load(os.path.join("tests", DATA["wavfile"][0]), sr=None, mono=True)

    F, D = librosa.core.spectrum.__reassign_frequencies(
        y=y,
        sr=DATA["sr"][0, 0].astype(int),
        n_fft=DATA["nfft"][0, 0].astype(int),
        hop_length=DATA["hop_length"][0, 0].astype(int),
        win_length=DATA["hann_w"][0, 0].astype(int),
        center=False,
    )

    # D fails to match here because of fftshift()
    # assert np.allclose(D, DATA['D'])
    assert np.allclose(F, DATA["F"], rtol=1e-3, atol=1e-3)


# results for longer windows containing multiple impulses will be unstable
@pytest.mark.parametrize("sr", [1, 512, 2048, 22050])
@pytest.mark.parametrize("n_fft", [128, 256, 1024, 2099])
def test___reassign_times(sr, n_fft):
    y = np.zeros(4096)
    y[[263, 2633]] = 1

    # frames with no energy will have all NaN time reassignments
    expected_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=n_fft)
    expected = np.full((n_fft // 2 + 1, expected_frames.shape[1]), np.nan)

    # find the impulses again; needed if the signal is truncated by framing
    impulse_indices = np.nonzero(expected_frames.ravel("F"))[0]

    # find the frames that the impulses should be placed into
    expected_bins = librosa.samples_to_frames(impulse_indices, hop_length=n_fft)

    # in each frame that contains an impulse, the energy in every frequency bin
    # should be reassigned to the original sample time
    expected_times = librosa.samples_to_time(impulse_indices, sr=sr)
    expected[:, expected_bins] = np.tile(expected_times, (n_fft // 2 + 1, 1))

    times, S = librosa.core.spectrum.__reassign_times(
        y=y, sr=sr, n_fft=n_fft, hop_length=n_fft, center=False
    )

    # times should be reassigned within 0.5% of the window duration
    assert np.allclose(times, expected, atol=0.005 * n_fft / sr, equal_nan=True)


def test___reassign_times_center():
    y = np.zeros(4096)
    y[2049] = 1

    sr = 4000
    n_fft = 2048

    times, S = librosa.core.spectrum.__reassign_times(
        y=y, sr=sr, hop_length=n_fft, win_length=n_fft, center=True
    )

    expected = np.full_like(times, np.nan)
    expected[:, 1] = 2049 / float(sr)

    assert np.allclose(times, expected, atol=0.005 * n_fft / sr, equal_nan=True)


@pytest.mark.parametrize("clip", [False, True])
@mock.patch("librosa.core.spectrum.__reassign_times")
@mock.patch("librosa.core.spectrum.__reassign_frequencies")
def test_reassigned_spectrogram_clip(
    mock_reassign_frequencies, mock_reassign_times, clip
):
    mock_freqs = np.ones((5, 17))
    mock_freqs[0, 0] = -1
    mock_freqs[0, 1] = 33

    mock_times = np.ones((5, 17))
    mock_times[1, 0] = -1
    mock_times[1, 1] = 3

    mock_mags = np.ones((5, 17))

    mock_reassign_frequencies.return_value = mock_freqs, mock_mags
    mock_reassign_times.return_value = mock_times, mock_mags

    freqs, times, mags = librosa.reassigned_spectrogram(
        y=np.zeros(128), sr=64, n_fft=8, hop_length=8, clip=clip
    )

    # freqs and times outside the spectrogram bounds
    if clip:
        assert freqs[0, 0] == 0
        assert freqs[0, 1] == 32
        assert times[1, 0] == 0
        assert times[1, 1] == 2

    else:
        assert freqs[0, 0] == -1
        assert freqs[0, 1] == 33
        assert times[1, 0] == -1
        assert times[1, 1] == 3

    assert freqs[2, 0] == 1
    assert times[2, 1] == 1


@pytest.mark.parametrize("ref_power", [0.0, 1e-6, np.max])
@mock.patch("librosa.core.spectrum.__reassign_times")
@mock.patch("librosa.core.spectrum.__reassign_frequencies")
def test_reassigned_spectrogram_ref_power(
    mock_reassign_frequencies, mock_reassign_times, ref_power
):
    mock_freqs = np.ones((5, 17))
    mock_times = np.ones((5, 17))

    mock_mags = np.ones((5, 17))
    mock_mags[2, 0] = 0
    mock_mags[2, 1] = 0.1

    mock_reassign_frequencies.return_value = mock_freqs, mock_mags
    mock_reassign_times.return_value = mock_times, mock_mags

    freqs, times, mags = librosa.reassigned_spectrogram(
        y=np.zeros(128), sr=64, n_fft=8, hop_length=8, ref_power=ref_power
    )

    if ref_power is np.max:
        assert np.isnan(freqs[2, 0])
        assert np.isnan(freqs[2, 1])
        assert np.isnan(times[2, 0])
        assert np.isnan(times[2, 1])

    elif ref_power == 1e-6:
        assert np.isnan(freqs[2, 0])
        assert freqs[2, 1] == 1
        assert np.isnan(times[2, 0])
        assert times[2, 1] == 1

    elif ref_power == 0:
        assert freqs[2, 0] == 1
        assert freqs[2, 1] == 1
        assert times[2, 0] == 1
        assert times[2, 1] == 1

    assert freqs[2, 2] == 1
    assert times[2, 2] == 1


@pytest.mark.parametrize("fill_nan", [False, True])
@mock.patch("librosa.core.spectrum.__reassign_times")
@mock.patch("librosa.core.spectrum.__reassign_frequencies")
def test_reassigned_spectrogram_fill_nan(
    mock_reassign_frequencies, mock_reassign_times, fill_nan
):
    mock_freqs = np.ones((5, 17))
    mock_times = np.ones((5, 17))

    # mock divide by zero
    mock_freqs[3, 0] = np.nan
    mock_times[3, 0] = np.nan

    # mock below default ref_power threshold (<1e-6)
    mock_mags = np.ones((5, 17))
    mock_mags[3, 1] = 0

    mock_reassign_frequencies.return_value = mock_freqs, mock_mags
    mock_reassign_times.return_value = mock_times, mock_mags

    freqs, times, mags = librosa.reassigned_spectrogram(
        y=np.zeros(128), sr=64, n_fft=8, hop_length=8, center=False, fill_nan=fill_nan
    )

    if fill_nan:
        # originally NaN due to divide-by-zero
        assert freqs[3, 0] == 24
        assert times[3, 0] == 0.0625

        # originally NaN due to low power
        assert freqs[3, 1] == 24
        assert times[3, 1] == 0.1875

    else:
        assert np.isnan(freqs[3, 0])
        assert np.isnan(times[3, 0])

        assert np.isnan(freqs[3, 1])
        assert np.isnan(times[3, 1])

    assert mags[3, 1] == 0

    assert freqs[3, 2] == 1
    assert times[3, 2] == 1


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("reassign_times", [False, True])
@pytest.mark.parametrize("reassign_frequencies", [False, True])
def test_reassigned_spectrogram_flags(reassign_frequencies, reassign_times, center):

    if not reassign_frequencies and not reassign_times:
        return

    freqs, times, mags = librosa.reassigned_spectrogram(
        y=np.zeros(2048),
        center=center,
        reassign_frequencies=reassign_frequencies,
        reassign_times=reassign_times,
    )

    if reassign_frequencies:
        assert np.all(np.isnan(freqs))

    else:
        bin_freqs = librosa.fft_frequencies()
        assert np.array_equiv(freqs, bin_freqs[:, np.newaxis])

    if reassign_times:
        assert np.all(np.isnan(times))

    else:
        frame_times = librosa.frames_to_time(np.arange(freqs.shape[1]))

        if not center:
            frame_times = frame_times + (2048.0 / 22050.0 / 2.0)

        assert np.array_equiv(times, frame_times[np.newaxis, :])


def test_reassigned_spectrogram_parameters():
    with pytest.raises(librosa.ParameterError):
        freqs, times, mags = librosa.reassigned_spectrogram(
            y=np.zeros(2048), ref_power=-1
        )

    with pytest.raises(librosa.ParameterError):
        freqs, times, mags = librosa.reassigned_spectrogram(
            y=np.zeros(2048), reassign_frequencies=False, reassign_times=False
        )


def test_salience_basecase():
    (y, sr) = librosa.load(os.path.join("tests", "data", "test1_22050.wav"))
    S = np.abs(librosa.stft(y))
    freqs = librosa.core.fft_frequencies(sr=sr)
    harms = [1]
    weights = [1.0]
    S_sal = librosa.core.salience(
        S,
        freqs=freqs,
        harmonics=harms,
        weights=weights,
        filter_peaks=False,
        kind="quadratic",
    )
    assert np.allclose(S_sal, S)


def test_salience_basecase2():
    (y, sr) = librosa.load(os.path.join("tests", "data", "test1_22050.wav"))
    S = np.abs(librosa.stft(y))
    freqs = librosa.core.fft_frequencies(sr=sr)
    harms = [1, 0.5, 2.0]
    weights = [1.0, 0.0, 0.0]
    S_sal = librosa.core.salience(
        S,
        freqs=freqs,
        harmonics=harms,
        weights=weights,
        filter_peaks=False,
        kind="quadratic",
    )
    assert np.allclose(S_sal, S)


def test_salience_defaults():
    S = np.array([[0.1, 0.5, 0.0], [0.2, 1.2, 1.2], [0.0, 0.7, 0.3], [1.3, 3.2, 0.8]])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    actual = librosa.core.salience(
        S, freqs=freqs, harmonics=harms, kind="quadratic", fill_value=0.0
    )

    expected = (
        np.array([[0.0, 0.0, 0.0], [0.3, 2.4, 1.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        / 3.0
    )
    assert np.allclose(expected, actual)


def test_salience_weights():
    S = np.array([[0.1, 0.5, 0.0], [0.2, 1.2, 1.2], [0.0, 0.7, 0.3], [1.3, 3.2, 0.8]])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    weights = [1.0, 1.0, 1.0]
    actual = librosa.core.salience(
        S,
        freqs=freqs,
        harmonics=harms,
        weights=weights,
        kind="quadratic",
        fill_value=0.0,
    )

    expected = (
        np.array([[0.0, 0.0, 0.0], [0.3, 2.4, 1.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        / 3.0
    )
    assert np.allclose(expected, actual)


def test_salience_no_peak_filter():
    S = np.array([[0.1, 0.5, 0.0], [0.2, 1.2, 1.2], [0.0, 0.7, 0.3], [1.3, 3.2, 0.8]])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    weights = [1.0, 1.0, 1.0]
    actual = librosa.core.salience(
        S,
        freqs=freqs,
        harmonics=harms,
        weights=weights,
        filter_peaks=False,
        kind="quadratic",
    )

    expected = (
        np.array([[0.3, 1.7, 1.2], [0.3, 2.4, 1.5], [1.5, 5.1, 2.3], [1.3, 3.9, 1.1]])
        / 3.0
    )
    assert np.allclose(expected, actual)


def test_salience_aggregate():
    S = np.array([[0.1, 0.5, 0.0], [0.2, 1.2, 1.2], [0.0, 0.7, 0.3], [1.3, 3.2, 0.8]])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    weights = [1.0, 1.0, 1.0]
    actual = librosa.core.salience(
        S,
        freqs=freqs,
        harmonics=harms,
        weights=weights,
        aggregate=np.max,
        kind="quadratic",
        fill_value=0.0,
    )

    expected = np.array(
        [[0.0, 0.0, 0.0], [0.2, 1.2, 1.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert np.allclose(expected, actual)


@pytest.fixture(scope="module")
def y_22050():
    y, sr = librosa.load(os.path.join("tests", "data", "test1_22050.wav"))
    return y


@pytest.fixture(scope="module")
def y_44100():
    y, sr = librosa.load(os.path.join("tests", "data", "test1_44100.wav"), sr=None)
    return y


def test_magphase(y_22050):

    D = librosa.stft(y_22050)

    S, P = librosa.magphase(D)

    assert S.dtype is y_22050.dtype  # float
    assert P.dtype is D.dtype  # complex
    assert np.allclose(np.abs(P), 1.0)
    assert np.allclose(S * P, D)


def test_magphase_zero():

    D = np.zeros((128, 128), dtype=np.complex64)
    S, P = librosa.magphase(D)

    assert S.dtype is np.dtype("float32")
    assert P.dtype is np.dtype("complex64")
    assert np.allclose(S, 0)
    assert np.allclose(P, 1 + 0j)


def test_magphase_denormalized():

    D = 1.0e-42j * np.ones((128, 128), dtype=np.complex64)
    S, P = librosa.magphase(D)

    assert S.dtype is np.dtype("float32")
    assert P.dtype is np.dtype("complex64")
    assert np.allclose(S, 1.0e-42)
    assert np.allclose(P, 0 + 1j)


def test_magphase_real():

    D = np.array([[-1.0, -0.0], [0.0, 1.0]], dtype=np.float64)
    S, P = librosa.magphase(D)

    assert S.dtype is np.dtype("float64")
    assert P.dtype is np.dtype("complex128")
    assert np.allclose(S, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(
        [
            [P[0, 0], P[0, 1] ** 2],  # negative zero can have phase +1 or -1
            [P[1, 0], P[1, 1]],
        ],
        np.array([[-1 + 0j, 1 + 0j], [1 + 0j, 1 + 0j]]),
    )


@pytest.fixture(scope="module", params=[22050, 44100])
def y_chirp_istft(request):
    sr = request.param
    return (librosa.chirp(fmin=32, fmax=8192, sr=sr, duration=2.0), sr)


@pytest.mark.parametrize("n_fft", [1024, 1025, 2048, 4096])
@pytest.mark.parametrize("window", ["hann", "blackmanharris"])
@pytest.mark.parametrize("hop_length", [128, 256, 512])
def test_istft_reconstruction(y_chirp_istft, n_fft, hop_length, window):

    x, sr = y_chirp_istft
    S = librosa.core.stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
    x_reconstructed = librosa.core.istft(
        S, hop_length=hop_length, window=window, n_fft=n_fft, length=len(x)
    )

    # NaN/Inf/-Inf should not happen
    assert np.all(np.isfinite(x_reconstructed))

    # should be almost approximately reconstructed
    assert np.allclose(x, x_reconstructed, atol=1e-6)


@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("duration", [None, 0, 0.5, 1, 2])
@pytest.mark.parametrize("mono", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "filename", files(os.path.join("tests", "data", "test1_22050.*"))
)
def test_load_options(filename, offset, duration, mono, dtype):
    y, sr = librosa.load(
        filename, mono=mono, offset=offset, duration=duration, dtype=dtype
    )

    if duration is not None:
        assert np.allclose(y.shape[-1], int(sr * duration))

    if mono:
        assert y.ndim == 1
    else:
        # This test file is stereo, so y.ndim should be 2
        assert y.ndim == 2

    # Check the dtype
    assert np.issubdtype(y.dtype, dtype)
    assert np.issubdtype(dtype, y.dtype)


@pytest.mark.parametrize("sr", [8000, 11025])
@pytest.mark.parametrize("dur", [0.25, 1.0])
def test_get_duration_buffer(sr, dur):
    y = np.zeros(int(sr * dur))
    dur_est = librosa.get_duration(y=y, sr=sr)
    assert np.isclose(dur_est, dur, atol=1e-4)


@pytest.mark.parametrize("sr", [8000, 11025])
@pytest.mark.parametrize("dur", [0.25, 1.0])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("n_fft", [512, 2048])
@pytest.mark.parametrize("hop_length", [256, 512])
def test_get_duration_specgram(sr, dur, n_fft, hop_length, center):
    z = np.zeros(int(sr * dur))
    S = librosa.util.frame(z, frame_length=(n_fft // 2 + 1), hop_length=hop_length)

    dur_est = librosa.get_duration(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, center=center
    )

    if center:
        # If we're assuming centering, estimated samples lose 2 * int(n_fft//2)
        assert np.abs(dur_est + 2 * int(n_fft // 2) / sr - dur) <= n_fft / sr
    else:
        # If we're not assuming centering, then duration should be off by at most n_fft
        assert np.abs(dur_est - dur) <= n_fft / sr


def test_get_duration_filename():

    filename = os.path.join("tests", "data", "test2_8000.wav")
    true_duration = 30.197625

    duration_fn = librosa.get_duration(path=filename)
    y, sr = librosa.load(filename, sr=None)
    duration_y = librosa.get_duration(y=y, sr=sr)

    assert np.allclose(duration_fn, true_duration)
    assert np.allclose(duration_fn, duration_y)


def test_get_duration_mp3():
    filename = os.path.join("tests", "data", "test1_22050.mp3")
    true_duration = 4.587528344671202

    duration_fn = librosa.get_duration(path=filename)
    y, sr = librosa.load(filename, sr=None)
    duration_y = librosa.get_duration(y=y, sr=sr)
    # mp3 duration at low sampling rate isn't too reliable
    assert np.allclose(duration_fn, duration_y, atol=1e-1)
    assert np.allclose(duration_fn, true_duration, atol=1e-1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_get_duration_fail():
    librosa.get_duration(y=None, S=None, path=None)


@pytest.mark.parametrize(
    "y",
    [np.random.randn(256, 384), np.exp(1.0j * np.random.randn(256, 384))],
    ids=["real", "complex"],
)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("max_size", [None, 128, 256, 512])
def test_autocorrelate(y, axis, max_size):

    truth = [
        np.asarray(
            [
                scipy.signal.fftconvolve(yi, yi[::-1].conj(), mode="full")[
                    len(yi) - 1 :
                ]
                for yi in y.T
            ]
        ).T,
        np.asarray(
            [
                scipy.signal.fftconvolve(yi, yi[::-1].conj(), mode="full")[
                    len(yi) - 1 :
                ]
                for yi in y
            ]
        ),
    ]

    ac = librosa.autocorrelate(y, max_size=max_size, axis=axis)

    my_slice = [slice(None)] * truth[axis].ndim
    if max_size is not None and max_size <= y.shape[axis]:
        my_slice[axis] = slice(min(max_size, y.shape[axis]))

    if not np.iscomplexobj(y):
        assert not np.iscomplexobj(ac)

    assert np.allclose(ac, truth[axis][tuple(my_slice)])


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "core-lpcburg-*.mat"))
)
def test_lpc_regress(infile):

    test_data = scipy.io.loadmat(infile, squeeze_me=True)
    for i in range(len(test_data["signal"])):
        signal = test_data["signal"][i]
        order = int(test_data["order"][i])
        true_coeffs = test_data["true_coeffs"][i]
        est_coeffs = test_data["est_coeffs"][i]

        test_coeffs = librosa.lpc(signal, order=order)
        assert np.allclose(test_coeffs, est_coeffs)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_lpc_simple(dtype):
    srand()

    n = 5000
    est_a = np.zeros((n, 6), dtype=dtype)
    truth_a = np.array([1, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=dtype)
    for i in range(n):
        noise = np.random.randn(1000).astype(dtype)
        filtered = scipy.signal.lfilter(dtype([1]), truth_a, noise)
        est_a[i, :] = librosa.lpc(filtered, order=5)
    assert dtype == est_a.dtype
    assert np.allclose(truth_a, np.mean(est_a, axis=0), rtol=0, atol=1e-3)


@pytest.mark.parametrize(
    "y", [np.arange(5.0), np.zeros((2, 5), dtype=float)], ids=["mono", "stereo"]
)
def test_to_mono(y):

    y_mono = librosa.to_mono(y)

    assert y_mono.ndim == 1
    assert len(y_mono) == y.shape[-1]

    if y.ndim == 1:
        assert np.allclose(y, y_mono)


@pytest.mark.parametrize(
    "y", [np.ones((2, 10)), np.ones((2, 3, 10)), np.ones((2, 3, 4, 10))]
)
def test_to_mono_multi(y):
    y_mono = librosa.to_mono(y)

    assert y_mono.ndim == 1
    assert len(y_mono) == y.shape[-1]


@pytest.mark.parametrize("data", [np.random.randn(32)])
@pytest.mark.parametrize("threshold", [0, 1e-10])
@pytest.mark.parametrize("ref_magnitude", [None, 0.1, np.max])
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("zp", [False, True])
def test_zero_crossings(data, threshold, ref_magnitude, pad, zp):

    zc = librosa.zero_crossings(
        y=data, threshold=threshold, ref_magnitude=ref_magnitude, pad=pad, zero_pos=zp
    )

    idx = np.flatnonzero(zc)

    if pad:
        idx = idx[1:]

    for i in idx:
        assert np.sign(data[i]) != np.sign(data[i - 1])


@pytest.mark.parametrize("resolution", [1e-2, 1e-3])
@pytest.mark.parametrize("tuning", [-0.5, -0.375, -0.25, 0.0, 0.25, 0.375])
@pytest.mark.parametrize("bins_per_octave", [12])
def test_pitch_tuning(resolution, bins_per_octave, tuning):

    hz = librosa.midi_to_hz(tuning + np.arange(128))
    est_tuning = librosa.pitch_tuning(
        hz, resolution=resolution, bins_per_octave=bins_per_octave
    )

    assert np.abs(tuning - est_tuning) <= resolution


@pytest.fixture(params=[2048, 4096], scope="module")
def pip_nfft(request):
    return request.param


@pytest.fixture(params=[None, 4, 2], scope="module")
def pip_hop(request, pip_nfft):
    if request.param is None:
        return None
    else:
        return pip_nfft // request.param


@pytest.fixture(scope="module")
def pip_spec(y_22050, pip_nfft, pip_hop):
    return np.abs(librosa.stft(y_22050, n_fft=pip_nfft, hop_length=pip_hop))


@pytest.mark.parametrize("fmin", [0, 100])
@pytest.mark.parametrize("fmax", [4000, 8000, 11025])
@pytest.mark.parametrize("threshold", [0.1, 0.2, 0.5])
@pytest.mark.parametrize("ref", [None, 1.0, np.max])
def test_piptrack_properties(pip_spec, pip_nfft, pip_hop, fmin, fmax, threshold, ref):

    n_fft = pip_nfft
    hop_length = pip_hop
    S = pip_spec

    pitches, mags = librosa.core.piptrack(
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        threshold=threshold,
        ref=ref,
    )

    # Shape tests
    assert S.shape == pitches.shape
    assert S.shape == mags.shape

    # Make sure all magnitudes are positive
    assert np.all(mags >= 0)

    # Check the frequency estimates for bins with non-zero magnitude
    idx = mags > 0
    assert np.all(pitches[idx] >= fmin)
    assert np.all(pitches[idx] <= fmax)

    # And everywhere else, pitch should be 0
    assert np.all(pitches[~idx] == 0)


def test_piptrack_errors():
    np.seterr(divide="raise")

    pitches, mags = librosa.piptrack(
        y=None,
        sr=22050,
        S=np.asarray([[1.0, 0.0, 0.0]]).T,
        n_fft=4096,
        hop_length=None,
        fmin=150,
        fmax=4000,
        threshold=0.1,
    )


@pytest.mark.parametrize("freq", [110, 220, 440, 880])
def test_yin_tone(freq):
    y = librosa.tone(freq, duration=1.0)
    f0 = librosa.yin(y, fmin=110, fmax=880, center=False)
    assert np.allclose(np.log2(f0), np.log2(freq), rtol=0, atol=1e-2)


def test_yin_chirp():
    # test yin on a chirp, using output from the vamp plugin as ground truth
    y = librosa.chirp(fmin=220, fmax=640, duration=1.0)
    f0 = librosa.yin(
        y, fmin=110, fmax=880, center=False, frame_length=1024, hop_length=512
    )

    # adjust frames to the removal of win_length from yin
    f0 = f0[:-2]

    target_f0 = np.load(os.path.join("tests", "data", "pitch-yin.npy"))
    assert np.allclose(np.log2(f0), np.log2(target_f0), rtol=0, atol=1e-2)


def test_yin_chirp_instant():
    # test yin on a chirp, using frame-wise instantaneous frequency as ground truth
    sr = 22050
    chirp_min, chirp_max = 220, 640

    t = np.arange(sr) / sr
    f = chirp_min * (chirp_max / chirp_min) ** t

    fl = 2048
    hl = 512

    y = librosa.chirp(fmin=chirp_min, fmax=chirp_max, sr=sr, duration=1.0, linear=False)
    target_f0 = librosa.util.frame(f, frame_length=fl, hop_length=hl).mean(axis=0)

    f0 = librosa.yin(
        y, fmin=110, fmax=880, sr=sr, frame_length=fl, hop_length=hl, center=False
    )
    assert np.allclose(np.log2(f0), np.log2(target_f0), rtol=0, atol=1e-2)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "fmin,fmax,frame_length",
    [
        (None, None, 2048),  # neither
        (110, None, 2048),  # no fmax
        (None, 880, 2048),  # no fmin
        (-1, 440, 2048),  # Negative fmin
        (440, 220, 2048),  # fmin > fmax
        (440, 16000, 2048),  # fmax > nyquist
        (10, 21, 2048),  # sr / fmin >= frame_length - 1
    ],
)
def test_yin_fail(fmin, fmax, frame_length):
    y = librosa.tone(110, duration=1.0)
    librosa.yin(y, fmin=fmin, fmax=fmax, frame_length=frame_length)


def test_yin_warn():
    y = librosa.tone(110, duration=1.0)

    # win_length is deprecated
    with pytest.warns(FutureWarning, match="deprecated"):
        librosa.yin(y, fmin=110, fmax=1000, win_length=1024)

    # sr / fmin >= frame_length // 2
    with pytest.warns(UserWarning, match="two periods"):
        librosa.yin(y, fmin=20, fmax=1000)


@pytest.mark.parametrize("freq", [110, 220, 440, 880])
def test_pyin_tone(freq):
    y = librosa.tone(freq, duration=1.0)
    f0, _, _ = librosa.pyin(y, fmin=110, fmax=1000, center=False)
    assert np.allclose(np.log2(f0), np.log2(freq), rtol=0, atol=1e-2)


def test_pyin_multi():
    y = np.stack([librosa.tone(440, duration=1.0), librosa.tone(560, duration=1.0)])

    # Taper the signal
    h = librosa.filters.get_window("triangle", y.shape[-1])

    # Filter it
    y = y * h[np.newaxis, :]

    # Disable nans so we can use allclose checks
    fall, vall, vpall = librosa.pyin(y, fmin=100, fmax=1000, center=False, fill_na=-1)
    f0, v0, vp0 = librosa.pyin(y[0], fmin=100, fmax=1000, center=False, fill_na=-1)
    f1, v1, vp1 = librosa.pyin(y[1], fmin=100, fmax=1000, center=False, fill_na=-1)

    assert np.allclose(fall[0], f0)
    assert np.allclose(fall[1], f1)
    assert np.allclose(vall[0], v0)
    assert np.allclose(vall[1], v1)
    assert np.allclose(vpall[0], vp0)
    assert np.allclose(vpall[1], vp1)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Skip on OSX due to openblas issue"
)
def test_pyin_multi_center():
    # Note: this test has issues on OSX with libopenblas 0.3.26,
    # so we disable it for now.  We may re-enable it some time in the future.
    y = np.stack([librosa.tone(440, duration=1.0), librosa.tone(560, duration=1.0)])

    # Taper the signal
    h = librosa.filters.get_window("triangle", y.shape[-1])

    # Filter it
    y = y * h[np.newaxis, :]

    fleft, vleft, vpleft = librosa.pyin(y, fmin=100, fmax=1000, center=False)
    fc, vc, vpc = librosa.pyin(y, fmin=100, fmax=1000, center=True)

    # Centering will pad by half a frame on either side
    # hop length is one quarter frame
    # ==> match on 2:-2

    # Loosening tolerances here to account for platform differences
    assert np.allclose(vpleft, vpc[..., 2:-2])
    assert np.allclose(vleft, vc[..., 2:-2])
    assert np.allclose(fleft, fc[..., 2:-2], equal_nan=True), np.max(
        np.abs(fleft - fc[..., 2:-2])
    )


def test_pyin_chirp():
    # test yin on a chirp, using output from the vamp plugin as ground truth
    y = librosa.chirp(fmin=220, fmax=640, duration=1.0)
    y = np.pad(y, (22050,))

    # default values as set in https://code.soundsoftware.ac.uk/projects/pyin/repository/
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=60,
        fmax=900,
        center=False,
        frame_length=1024,
        hop_length=512,
        resolution=0.2,
    )

    # adjust frames to the removal of win_length from yin
    f0 = f0[:-2]
    voiced_flag = voiced_flag[:-2]

    target_f0 = np.load(os.path.join("tests", "data", "pitch-pyin.npy"))
    # test if correct frames are voiced
    assert np.array_equal(voiced_flag, target_f0 > 0)
    # test voiced frames are within one cent of the target
    assert np.allclose(
        np.log2(f0[voiced_flag]), np.log2(target_f0[target_f0 > 0]), rtol=0, atol=1e-2
    )


def test_pyin_chirp_instant():
    # test pyin on a chirp, using frame-wise instantaneous frequency as ground truth
    sr = 22050
    chirp_min, chirp_max = 220, 640

    t = np.arange(sr) / sr
    f = chirp_min * (chirp_max / chirp_min) ** t
    f = np.pad(f, (sr,))

    fl = 2048
    hl = 512

    # Note: this raises warnings on the empty frames
    target_f0 = librosa.util.frame(f, frame_length=fl, hop_length=hl)
    with pytest.warns(RuntimeWarning):
        target_f0 = target_f0.mean(axis=0, where=target_f0 > 0)

    y = librosa.chirp(fmin=chirp_min, fmax=chirp_max, sr=sr, duration=1.0, linear=False)
    y = np.pad(y, (sr,))

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=110, fmax=880, frame_length=fl, hop_length=hl, center=False
    )

    # test if correct frames are voiced
    assert np.array_equal(voiced_flag, target_f0 > 0)

    # test voiced frames are within one cent of the target
    cents = np.log2(f0[voiced_flag])
    target_cents = np.log2(target_f0[target_f0 > 0])

    assert np.allclose(
        np.log2(cents[1:-1]), np.log2(target_cents[1:-1]), rtol=0, atol=1e-2
    )

    # higher tolerance for the first and last frames, accounting for abrupt start / end
    assert np.abs(cents[0] - target_cents[0]) <= 1e-1
    assert np.abs(cents[-1] - target_cents[-1]) <= 1e-1


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "fmin,fmax,frame_length",
    [
        (None, None, 2048),
        (110, None, 2048),
        (None, 880, 2048),
    ],
)
def test_pyin_fail(fmin, fmax, frame_length):
    y = librosa.tone(110, duration=1.0)
    librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=frame_length)


def test_pyin_warn():
    y = librosa.tone(110, duration=1.0)

    # win_length is deprecated
    with pytest.warns(FutureWarning, match="deprecated"):
        librosa.pyin(y, fmin=110, fmax=1000, win_length=1024)


@pytest.mark.parametrize("freq", [110, 220, 440, 880])
@pytest.mark.parametrize("n_fft", [1024, 2048, 4096])
def test_piptrack(freq, n_fft):

    y = librosa.tone(freq, sr=22050, duration=0.2)
    S = np.abs(librosa.stft(y, n_fft=n_fft, center=False))

    pitches, mags = librosa.piptrack(S=S, fmin=100)

    idx = mags > 0

    assert len(idx) > 0

    recovered_pitches = pitches[idx]

    # We should be within one cent of the target
    assert np.all(np.abs(np.log2(recovered_pitches) - np.log2(freq)) <= 1e-2)


@pytest.mark.parametrize("center_note", [69, 84, 108])
@pytest.mark.parametrize("tuning", np.linspace(-0.5, 0.5, 5, endpoint=False))
@pytest.mark.parametrize("bins_per_octave", [12])
@pytest.mark.parametrize("resolution", [1e-2])
@pytest.mark.parametrize("sr", [11025, 22050])
def test_estimate_tuning(
    sr, center_note: int, tuning: float, bins_per_octave, resolution
):

    target_hz = librosa.midi_to_hz(center_note + tuning)

    y = librosa.tone(target_hz, duration=0.5, sr=sr)

    tuning_est = librosa.estimate_tuning(
        resolution=resolution,
        bins_per_octave=bins_per_octave,
        y=y,
        sr=sr,
        n_fft=2048,
        fmin=librosa.note_to_hz("C4"),
        fmax=librosa.note_to_hz("G#9"),
    )

    # Round to the proper number of decimals
    deviation = np.around(tuning - tuning_est, int(-np.log10(resolution)))

    # Take the minimum floating point for positive and negative deviations
    max_dev = np.min([np.mod(deviation, 1.0), np.mod(-deviation, 1.0)])

    # We'll accept an answer within three bins of the resolution
    assert max_dev <= 3 * resolution


@pytest.mark.parametrize("y", [np.zeros(4000)])
@pytest.mark.parametrize("sr", [11025, 22050])
@pytest.mark.parametrize("resolution", [1e-2])
@pytest.mark.parametrize("bins_per_octave", [12])
def test_estimate_tuning_null(y, sr, resolution, bins_per_octave):
    with pytest.warns(UserWarning, match="Trying to estimate tuning"):
        tuning_est = librosa.estimate_tuning(
            resolution=resolution, bins_per_octave=bins_per_octave, y=y, sr=sr
        )
        assert np.allclose(tuning_est, 0)


@pytest.mark.parametrize("n_fft", [1024, 755, 2048, 2049])
@pytest.mark.parametrize("hop_length", [None, 512])
@pytest.mark.parametrize("power", [1, 2])
def test__spectrogram(y_22050, n_fft, hop_length, power):

    y = y_22050
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** power

    S_, n_fft_ = librosa.core.spectrum._spectrogram(
        y=y, S=S, n_fft=n_fft, hop_length=hop_length, power=power
    )

    # First check with all parameters
    assert np.allclose(S, S_)
    assert np.allclose(n_fft, n_fft_)

    # Then check with only the audio
    S_, n_fft_ = librosa.core.spectrum._spectrogram(
        y=y, n_fft=n_fft, hop_length=hop_length, power=power
    )
    assert np.allclose(S, S_)
    assert np.allclose(n_fft, n_fft_)

    # And only the spectrogram
    S_, n_fft_ = librosa.core.spectrum._spectrogram(
        S=S, n_fft=n_fft, hop_length=hop_length, power=power
    )
    assert np.allclose(S, S_)
    assert np.allclose(n_fft, n_fft_)

    # And only the spectrogram with no shape parameters
    S_, n_fft_ = librosa.core.spectrum._spectrogram(S=S, power=power)
    assert np.allclose(S, S_)
    if n_fft % 2 == 0:
        # Inference will be wrong if the frame length was odd
        assert np.allclose(n_fft, n_fft_)

    # And only the spectrogram but with incorrect n_fft
    S_, n_fft_ = librosa.core.spectrum._spectrogram(S=S, n_fft=2 * n_fft, power=power)
    assert np.allclose(S, S_)

    assert np.allclose(2 * (S.shape[-2] - 1), n_fft_)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test__spectrogram_no_nfft():
    librosa.core.spectrum._spectrogram(S=None, n_fft=None)


@pytest.mark.parametrize(
    "x", [np.linspace(0, 2e5, num=10), np.linspace(0, 2e5, num=10) * np.exp(1.0j)]
)
@pytest.mark.parametrize("ref", [1.0, np.max])
@pytest.mark.parametrize("amin", [1e-10, 1e3])
@pytest.mark.parametrize("top_db", [None, 0, 40, 80])
def test_power_to_db(x, ref, amin, top_db):

    if np.iscomplexobj(x):
        with pytest.warns(UserWarning, match="power_to_db was called on complex input"):
            y = librosa.power_to_db(x, ref=ref, amin=amin, top_db=top_db)
    else:
        y = librosa.power_to_db(x, ref=ref, amin=amin, top_db=top_db)

    assert np.isrealobj(y)
    assert y.shape == x.shape

    if top_db is not None:
        assert y.min() >= y.max() - top_db


@pytest.mark.parametrize("x", [np.ones(10)])
@pytest.mark.parametrize("top_db,amin", [(20, -1), (20, 0), (-5, 1e-10)])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_power_to_db_fail(x, top_db, amin):
    librosa.power_to_db(x, amin=amin, top_db=top_db)


@pytest.mark.parametrize("erp", range(-5, 6))
@pytest.mark.parametrize("k", range(-5, 6))
def test_power_to_db_inv(erp, k):

    y_true = (k - erp) * 10
    x = 10.0**k
    rp = 10.0**erp
    y = librosa.power_to_db(x, ref=rp, top_db=None)

    assert np.isclose(y, y_true)


def test_amplitude_to_db():
    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    db1 = librosa.amplitude_to_db(x, top_db=None)
    db2 = librosa.power_to_db(x**2, top_db=None)

    assert np.allclose(db1, db2)


def test_amplitude_to_db_complex():
    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    with pytest.warns(UserWarning, match="amplitude_to_db was called on complex input"):
        db1 = librosa.amplitude_to_db(x.astype(complex), top_db=None)

    db2 = librosa.power_to_db(x**2, top_db=None)

    assert np.allclose(db1, db2)


def test_amplitude_to_db_scalar():
    assert np.isclose(librosa.amplitude_to_db(1), 0)
    assert np.isclose(librosa.amplitude_to_db(2), 6.0206)


def test_power_to_db_scalar():
    assert np.isclose(librosa.power_to_db(1), 0)
    assert np.isclose(librosa.power_to_db(2), 3.0103)


def test_db_to_amplitude_scalar():
    assert np.isclose(librosa.db_to_amplitude(0), 1)
    assert np.isclose(librosa.db_to_amplitude(6.0206), 2)


def test_db_to_power_scalar():
    assert np.isclose(librosa.db_to_power(0), 1)
    assert np.isclose(librosa.db_to_power(3.0103), 2)


@pytest.mark.parametrize("ref_p", range(-3, 4))
@pytest.mark.parametrize("xp", [(np.abs(np.random.randn(1000)) + 1e-5) ** 2])
def test_db_to_power_inv(ref_p, xp):

    ref = 10.0**ref_p
    db = librosa.power_to_db(xp, ref=ref, top_db=None)
    xp2 = librosa.db_to_power(db, ref=ref)

    assert np.allclose(xp, xp2)


@pytest.mark.parametrize("erp", range(-5, 6))
@pytest.mark.parametrize("db", range(-100, 101, 10))
def test_db_to_power(erp, db):

    y = db
    rp = 10.0**erp
    x_true = 10.0**erp * (10.0 ** (0.1 * db))

    x = librosa.db_to_power(y, ref=rp)

    assert np.isclose(x, x_true), (x, x_true, y, rp)


@pytest.mark.parametrize("ref_p", range(-3, 4))
@pytest.mark.parametrize("xp", [np.abs(np.random.randn(1000)) + 1e-5])
def test_db_to_amplitude_inv(xp, ref_p):

    ref = 10.0**ref_p
    db = librosa.amplitude_to_db(xp, ref=ref, top_db=None)
    xp2 = librosa.db_to_amplitude(db, ref=ref)

    assert np.allclose(xp, xp2)


def test_db_to_amplitude():

    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    db = librosa.amplitude_to_db(x, top_db=None)
    x2 = librosa.db_to_amplitude(db)

    assert np.allclose(x, x2)


@pytest.mark.parametrize("times", [np.linspace(0, 10.0, num=5)])
@pytest.mark.parametrize("sr", [11025, 22050])
@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize("click", [None, np.ones(1000)])
@pytest.mark.parametrize("use_frames", [False, True])
@pytest.mark.parametrize("click_freq", [1000])
@pytest.mark.parametrize("click_duration", [0.1])
@pytest.mark.parametrize("length", [None, 5 * 22050])
def test_clicks(
    times, sr, hop_length, click_freq, click_duration, click, length, use_frames
):
    if use_frames:
        frames = librosa.time_to_frames(times, sr=sr, hop_length=hop_length)
        times = None
    else:
        frames = None

    y = librosa.clicks(
        times=times,
        frames=frames,
        sr=sr,
        hop_length=hop_length,
        click_freq=click_freq,
        click_duration=click_duration,
        click=click,
        length=length,
    )

    if times is not None:
        nmax = librosa.time_to_samples(times, sr=sr).max()
    else:
        assert frames is not None
        nmax = librosa.frames_to_samples(frames, hop_length=hop_length).max()

    if length is not None:
        assert len(y) == length
    elif click is not None:
        assert len(y) == nmax + len(click)


@pytest.mark.parametrize(
    "times,click_freq,click_duration,click,length",
    [
        (None, 1000, 0.1, None, None),
        ([0, 2, 4, 8], 1000, 0.1, None, 0),
        ([0, 2, 4, 8], 0, 0.1, None, None),
        ([0, 2, 4, 8], 1000, 0, None, None),
    ],
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_clicks_fail(times, click_freq, click_duration, click, length):
    librosa.clicks(
        times=times,
        frames=None,
        sr=22050,
        hop_length=512,
        click_freq=click_freq,
        click_duration=click_duration,
        click=click,
        length=length,
    )


@pytest.mark.parametrize("frequency", [440])
@pytest.mark.parametrize("sr", [11025, 22050, 44100])
@pytest.mark.parametrize(
    "length,duration", [(None, 0.5), (1740, None), (22050, None), (1740, 0.5)]
)
@pytest.mark.parametrize("phi", [None, np.pi])
def test_tone(frequency, sr, length, duration, phi):

    y = librosa.tone(
        frequency=frequency, sr=sr, length=length, duration=duration, phi=phi
    )

    if length is not None:
        assert len(y) == length
    else:
        assert len(y) == int(duration * sr)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "frequency,length,duration", [(None, 22050, 1), (440, None, None)]
)
def test_tone_fail(frequency, length, duration):
    librosa.tone(frequency=frequency, sr=22050, length=length, duration=duration)


@pytest.mark.parametrize("fmin,fmax", [(440, 880), (880, 440)])
@pytest.mark.parametrize("sr", [11025, 22050])
@pytest.mark.parametrize("length,duration", [(None, 0.5), (11025, None), (11025, 0.5)])
@pytest.mark.parametrize("phi", [None, np.pi / 2])
@pytest.mark.parametrize("linear", [False, True])
def test_chirp(fmin, fmax, sr, length, duration, linear, phi):

    y = librosa.chirp(
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        length=length,
        duration=duration,
        linear=linear,
        phi=phi,
    )

    if length is not None:
        assert len(y) == length
    else:
        assert len(y) == int(duration * sr)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "fmin,fmax,length,duration",
    [
        (None, None, 22050, 1),
        (440, None, 22050, 1),
        (None, 880, 22050, 1),
        (440, 880, None, None),
    ],
)
def test_chirp_fail(fmin, fmax, length, duration):
    librosa.chirp(fmin=fmin, fmax=fmax, sr=22050, length=length, duration=duration)


@pytest.fixture(scope="module", params=[1, 2, 3])
def OVER_SAMPLE(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 2, 3.0 / 2, 5.0 / 4, 9.0 / 8])
def SCALE(request):
    return request.param


@pytest.fixture(scope="module")
def y_res(SCALE):
    y = np.sin(2 * np.pi * np.linspace(0, 1, num=int(SCALE * 256), endpoint=False))
    y /= np.sqrt(SCALE)
    return y


@pytest.fixture(scope="module")
def y_orig():
    return np.sin(2 * np.pi * np.linspace(0, 1, num=256, endpoint=False))


@pytest.mark.parametrize(
    "kind,atol", [("slinear", 1e-4), ("quadratic", 1e-5), ("cubic", 1e-6)]
)
@pytest.mark.parametrize("n_fmt", [None, 64, 128, 256, 512])
def test_fmt_scale(y_orig, y_res, n_fmt, kind, atol, SCALE, OVER_SAMPLE):

    # Make sure our signals preserve energy
    assert np.allclose(np.sum(y_orig**2), np.sum(y_res**2))

    # Scale-transform the original
    f_orig = librosa.fmt(
        y_orig, t_min=0.5, n_fmt=n_fmt, over_sample=OVER_SAMPLE, kind=kind
    )

    # Force to the same length
    n_fmt_res = 2 * len(f_orig) - 2

    # Scale-transform the new signal to match
    f_res = librosa.fmt(
        y_res, t_min=SCALE * 0.5, n_fmt=n_fmt_res, over_sample=OVER_SAMPLE, kind=kind
    )

    # Due to sampling alignment, we'll get some phase deviation here
    # The shape of the spectrum should be approximately preserved though.
    assert np.allclose(np.abs(f_orig), np.abs(f_res), atol=atol, rtol=1e-7)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("y", [np.random.randn(128)])
@pytest.mark.parametrize(
    "t_min,n_fmt,over_sample",
    [
        (-1, None, 2),
        (0, None, 2),
        (1, -1, 2),
        (1, 0, 2),
        (1, 1, 2),
        (1, 2, 2),
        (1, None, -1),
        (1, None, 0),
        (1, None, 0.5),
    ],
)
def test_fmt_fail(y, t_min, n_fmt, over_sample):
    librosa.fmt(y, t_min=t_min, n_fmt=n_fmt, over_sample=over_sample)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("y", [np.inf + np.ones(128), np.ones(2)], ids=["inf", "short"])
def test_fmt_fail_badinput(y):
    librosa.fmt(y, t_min=1, n_fmt=128, over_sample=1)


def test_fmt_axis():

    srand()
    y = np.random.randn(32, 32)

    f1 = librosa.fmt(y, axis=-1)
    f2 = librosa.fmt(y.T, axis=0).T

    assert np.allclose(f1, f2)


def test_harmonics_1d():

    x = np.arange(16)
    y = np.linspace(-8, 8, num=len(x), endpoint=False) ** 2

    h = [0.25, 0.5, 1, 2, 4]

    yh = librosa.interp_harmonics(y, freqs=x, harmonics=h, axis=0)

    assert yh.shape[1:] == y.shape
    assert yh.shape[0] == len(h)
    for i in range(len(h)):
        if h[i] <= 1:
            # Check that subharmonics match
            step = int(1.0 / h[i])
            vals = yh[i, ::step]
            assert np.allclose(vals, y[: len(vals)])
        else:
            # Else check that harmonics match
            step = cast(int, h[i])
            vals = y[::step]
            assert np.allclose(vals, yh[i, : len(vals)])


def test_harmonics_2d():

    x = np.arange(16)
    y = np.linspace(-8, 8, num=len(x), endpoint=False) ** 2
    y = np.tile(y, (5, 1)).T
    h = [0.25, 0.5, 1, 2, 4]

    yh = librosa.interp_harmonics(y, freqs=x, harmonics=h, axis=0)

    assert yh.shape[1:] == y.shape
    assert yh.shape[0] == len(h)
    for i in range(len(h)):
        if h[i] <= 1:
            # Check that subharmonics match
            step = int(1.0 / h[i])
            vals = yh[i, ::step]
            assert np.allclose(vals, y[: len(vals)])
        else:
            # Else check that harmonics match
            step = cast(int, h[i])
            vals = y[::step]
            assert np.allclose(vals, yh[i, : len(vals)])


def test_harmonics_1d_nonunique():
    x = np.arange(-8, 8) ** 2
    y = np.linspace(-8, 8, num=len(x), endpoint=False) ** 2

    h = [0.25, 0.5, 1, 2, 4]

    with pytest.warns(UserWarning):
        yh = librosa.interp_harmonics(y, freqs=x, harmonics=h, axis=0)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_harmonics_badshape_1d():
    freqs = np.zeros(100)
    obs = np.zeros((5, 10))
    librosa.interp_harmonics(obs, freqs=freqs, harmonics=[1])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_harmonics_badshape_2d():
    freqs = np.zeros((5, 5))
    obs = np.zeros((5, 10))
    librosa.interp_harmonics(obs, freqs=freqs, harmonics=[1])


def test_harmonics_2d_varying():

    x = np.arange(16)
    y = np.linspace(-8, 8, num=len(x), endpoint=False) ** 2
    x = np.tile(x, (5, 1)).T
    y = np.tile(y, (5, 1)).T
    h = [0.25, 0.5, 1, 2, 4]

    yh = librosa.interp_harmonics(y, freqs=x, harmonics=h, axis=-2)

    assert yh.shape[1:] == y.shape
    assert yh.shape[0] == len(h)
    for i in range(len(h)):
        if h[i] <= 1:
            # Check that subharmonics match
            step = int(1.0 / h[i])
            vals = yh[i, ::step]
            assert np.allclose(vals, y[: len(vals)])
        else:
            # Else check that harmonics match
            step = cast(int, h[i])
            vals = y[::step]
            assert np.allclose(vals, yh[i, : len(vals)])


def test_harmonics_2d_varying_nonunique():

    x = np.arange(-8, 8) ** 2
    y = np.linspace(-8, 8, num=len(x), endpoint=False) ** 2
    x = np.tile(x, (5, 1)).T
    y = np.tile(y, (5, 1)).T
    h = [0.25, 0.5, 1, 2, 4]

    with pytest.warns(UserWarning):
        yh = librosa.interp_harmonics(y, freqs=x, harmonics=h, axis=-2)


def test_show_versions():
    # Nothing to test here, except that everything passes.
    librosa.show_versions()


def test_iirt():
    gt = scipy.io.loadmat(
        os.path.join("tests", "data", "features-CT-cqt"), squeeze_me=True
    )["f_cqt"]

    # There shouldn't be a load here, but test1_44100 was resampled for this fixture :\
    y, sr = librosa.load(os.path.join("tests", "data", "test1_44100.wav"))

    mut1 = librosa.iirt(y, sr=sr, hop_length=2205, win_length=4410, flayout="ba")

    assert np.allclose(mut1, gt[23:108, : mut1.shape[1]], atol=1.8)

    mut2 = librosa.iirt(y, sr=sr, hop_length=2205, win_length=4410, flayout="sos")

    assert np.allclose(mut2, gt[23:108, : mut2.shape[1]], atol=1.8)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_iirt_flayout1(y_44100):
    y = y_44100
    sr = 44100
    librosa.iirt(y, hop_length=2205, win_length=4410, flayout="foo")


def test_iirt_peaks():
    # Test for PR #1157

    Fs = 4000
    length = 180
    click_times = [10, 50, 90, 130, 170]

    x = np.zeros(length * Fs)
    for click in click_times:
        x[click * Fs] = 1

    win_length = 200
    hop_length = 50
    center_freqs = librosa.midi_to_hz(np.arange(40, 95))
    sample_rates = np.asarray(
        len(np.arange(40, 46))
        * [
            1000.0,
        ]
        + len(np.arange(46, 80))
        * [
            1750.0,
        ]
        + len(np.arange(80, 95))
        * [
            4000.0,
        ]
    )

    X = librosa.iirt(
        x,
        center_freqs=center_freqs,
        sample_rates=sample_rates,
        win_length=win_length,
        hop_length=hop_length,
    )

    for cur_band in X:
        cur_peaks = scipy.signal.find_peaks(
            cur_band, height=np.mean(cur_band), distance=1000
        )[0]
        assert len(cur_peaks) == 5

        cur_peak_times = cur_peaks * hop_length / Fs
        assert all(abs(cur_peak_times - click_times) < (2 * win_length / Fs))


def test_iirt_padding():
    Fs = 22050
    x = np.zeros(Fs)
    H = 512
    num_frames = np.empty(5)

    for i in range(1, 6):
        N = i * H
        X = librosa.iirt(x, sr=Fs, hop_length=H, win_length=N, center=True)
        num_frames[i - 1] = X.shape[1]

    assert np.all(num_frames == num_frames[0])


@pytest.fixture(scope="module")
def S_pcen():
    return np.abs(np.random.randn(9, 30))


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "gain,bias,power,b,time_constant,eps,ms",
    [
        (-1, 1, 1, 0.5, 0.5, 1e-6, 1),  # gain < 0
        (1, -1, 1, 0.5, 0.5, 1e-6, 1),  # bias < 0
        (1, 1, -0.1, 0.5, 0.5, 1e-6, 1),  # power < 0
        (1, 1, 1, -2, 0.5, 1e-6, 1),  # b < 0
        (1, 1, 1, 2, 0.5, 1e-6, 1),  # b > 1
        (1, 1, 1, 0.5, -2, 1e-6, 1),  # time_constant <= 0
        (1, 1, 1, 0.5, 0.5, 0, 1),  # eps <= 0
        (1, 1, 1, 0.5, 0.5, 1e-6, 1.5),
        (1, 1, 1, 0.5, 0.5, 1e-6, 0),  # max_size not int, < 1
    ],
)
def test_pcen_failures(gain, bias, power, b, time_constant, eps, ms, S_pcen):
    librosa.pcen(
        S_pcen,
        gain=gain,
        bias=bias,
        power=power,
        time_constant=time_constant,
        eps=eps,
        b=b,
        max_size=ms,
    )


@pytest.mark.parametrize("p", [0.5, 1, 2])
def test_pcen_power(S_pcen, p):
    # when b=1, gain=0, bias=0, all filtering is disabled;
    # this test just checks that power calculations work as expected
    P = librosa.pcen(
        S_pcen, gain=0, bias=0, power=p, b=1, time_constant=0.5, eps=1e-6, max_size=1
    )
    assert np.allclose(P, S_pcen**p)


def test_pcen_ones(S_pcen):
    # when gain=1, bias=0, power=1, b=1, eps=1e-20, we should get all ones
    P = librosa.pcen(
        S_pcen, gain=1, bias=0, power=1, b=1, time_constant=0.5, eps=1e-20, max_size=1
    )
    assert np.allclose(P, np.ones_like(S_pcen))


@pytest.mark.parametrize("power", [0, 1e-3])
@pytest.mark.parametrize("bias", [0, 1])
def test_pcen_drc(S_pcen, bias, power):
    P = librosa.pcen(S_pcen, gain=0.0, bias=bias, power=power, eps=1e-20)
    if power == 0:
        ref = np.expm1(P)
    else:
        if bias == 0:
            ref = np.exp(1.0 / power * np.log(P))
        else:
            ref = np.expm1(1.0 / power * np.log1p(P))

    assert np.allclose(S_pcen, ref)


def test_pcen_complex():
    S = np.ones((9, 30), dtype=complex)
    Pexp = np.ones((9, 30))

    with warnings.catch_warnings(record=True) as out:

        P = librosa.pcen(
            S, gain=1, bias=0, power=1, time_constant=0.5, eps=1e-20, b=1, max_size=1
        )

        if np.issubdtype(S.dtype, np.complexfloating):
            assert len(out) > 0
            assert "complex" in str(out[0].message).lower()

    assert P.shape == S.shape
    assert np.all(P >= 0)
    assert np.all(np.isfinite(P))

    if Pexp is not None:
        assert np.allclose(P, Pexp)


@pytest.mark.parametrize("max_size", [1, 3])
@pytest.mark.parametrize("Z", [np.zeros((9, 30))])
def test_pcen_zeros(max_size, Z):
    P = librosa.pcen(
        Z,
        gain=0.98,
        bias=2.0,
        power=0.5,
        b=None,
        time_constant=0.395,
        eps=1e-6,
        max_size=max_size,
    )

    # PCEN should map zeros to zeros
    assert np.allclose(P, Z)


def test_pcen_axes():

    srand()
    # Make a power spectrogram
    X = np.random.randn(3, 100, 50) ** 2

    # First, test that axis setting works
    P1 = librosa.pcen(X[0])
    P1a = librosa.pcen(X[0], axis=-1)
    P2 = librosa.pcen(X[0].T, axis=0).T

    assert np.allclose(P1, P2)
    assert np.allclose(P1, P1a)

    # Test that it works with max-filtering
    P1 = librosa.pcen(X[0], max_size=3)
    P1a = librosa.pcen(X[0], axis=-1, max_size=3)
    P2 = librosa.pcen(X[0].T, axis=0, max_size=3).T

    assert np.allclose(P1, P2)
    assert np.allclose(P1, P1a)

    # Test that it works with multi-dimensional input, no filtering
    P0 = librosa.pcen(X[0])
    P1 = librosa.pcen(X[1])
    P2 = librosa.pcen(X[2])
    Pa = librosa.pcen(X)

    assert np.allclose(P0, Pa[0])
    assert np.allclose(P1, Pa[1])
    assert np.allclose(P2, Pa[2])

    # Test that it works with multi-dimensional input, max-filtering
    P0 = librosa.pcen(X[0], max_size=3)
    P1 = librosa.pcen(X[1], max_size=3)
    P2 = librosa.pcen(X[2], max_size=3)
    Pa = librosa.pcen(X, max_size=3, max_axis=1)

    assert np.allclose(P0, Pa[0])
    assert np.allclose(P1, Pa[1])
    assert np.allclose(P2, Pa[2])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_pcen_axes_nomax():
    srand()
    # Make a power spectrogram
    X = np.random.randn(3, 100, 50) ** 2

    librosa.pcen(X, max_size=3)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_pcen_max1():

    librosa.pcen(np.arange(100), max_size=3)


def test_pcen_ref():

    srand()
    # Make a power spectrogram
    X = np.random.randn(100, 50) ** 2

    # Edge cases:
    #   gain=1, bias=0, power=1, b=1 => ones
    ones = np.ones_like(X)

    Y = librosa.pcen(X, gain=1, bias=0, power=1, b=1, eps=1e-20)
    assert np.allclose(Y, ones)

    # with ref=ones, we should get X / (eps + ones) == X
    Y2 = librosa.pcen(X, gain=1, bias=0, power=1, b=1, ref=ones, eps=1e-20)
    assert np.allclose(Y2, X)


@pytest.mark.parametrize("x", [np.arange(100), np.arange(100).reshape((10, 10))])
def test_pcen_stream(x):

    if x.ndim == 1:
        x1 = x[:20]
        x2 = x[20:]
    else:
        x1 = x[:, :20]
        x2 = x[:, 20:]

    p1, zf1 = librosa.pcen(x1, return_zf=True)
    p2, zf2 = librosa.pcen(x2, zi=zf1, return_zf=True)

    pfull = librosa.pcen(x)

    assert np.allclose(pfull, np.hstack([p1, p2]))


@pytest.mark.parametrize("axis", [0, 1, 2, -2, -1])
def test_pcen_stream_multi(axis):
    srand()

    # Generate a random power spectrum
    x = np.random.randn(20, 50, 60) ** 2

    # Make slices along the target axis
    slice1 = [slice(None)] * x.ndim
    slice1[axis] = slice(0, 10)
    slice2 = [slice(None)] * x.ndim
    slice2[axis] = slice(10, None)

    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    # Compute pcen piecewise
    p1, zf1 = librosa.pcen(x[slice1], return_zf=True, axis=axis)
    p2, zf2 = librosa.pcen(x[slice2], zi=zf1, return_zf=True, axis=axis)

    # And the full pcen
    pfull = librosa.pcen(x, axis=axis)

    # Compare full to concatenated results
    assert np.allclose(pfull, np.concatenate([p1, p2], axis=axis))


def test_get_fftlib():
    import scipy.fft as fft

    assert librosa.get_fftlib() is fft


def test_set_fftlib():
    with pytest.warns(FutureWarning):
        librosa.set_fftlib("foo")  # type: ignore
    assert librosa.get_fftlib() == "foo"  # type: ignore
    with pytest.warns(FutureWarning):
        librosa.set_fftlib()


def test_reset_fftlib():
    import scipy.fft as fft

    with pytest.warns(FutureWarning):
        librosa.set_fftlib()
    assert librosa.get_fftlib() is fft


@pytest.fixture
def y_chirp():
    sr = 22050
    y = librosa.chirp(fmin=55, fmax=55 * 2**7, length=sr // 8, sr=sr)
    return y


@pytest.mark.parametrize("hop_length", [None, 1024])
@pytest.mark.parametrize("win_length", [None, 1024])
@pytest.mark.parametrize("n_fft", [2048, 2049])
@pytest.mark.parametrize("window", ["hann", "rect"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_length", [False, True])
@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
@pytest.mark.parametrize("momentum", [0, 0.99])
@pytest.mark.parametrize("random_state", [None, 0, np.random.RandomState()])
@pytest.mark.parametrize("init", [None, "random"])
def test_griffinlim(
    y_chirp,
    hop_length,
    win_length,
    n_fft,
    window,
    center,
    dtype,
    use_length,
    pad_mode,
    momentum,
    init,
    random_state,
):

    if use_length:
        length = len(y_chirp)
    else:
        length = None

    D = librosa.stft(
        y_chirp,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    S = np.abs(D)

    y_rec = librosa.griffinlim(
        S,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        pad_mode=pad_mode,
        n_iter=3,
        momentum=momentum,
        init=init,
        random_state=random_state,
    )

    # First, check length
    if use_length:
        assert len(y_rec) == length

    # Next, check dtype
    assert y_rec.dtype == dtype


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_badinit():
    x = np.zeros((33, 3))
    librosa.griffinlim(x, init="garbage")


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_badrng():
    x = np.zeros((33, 3))
    librosa.griffinlim(x, random_state="garbage")  # type: ignore


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_momentum():
    x = np.zeros((33, 3))
    librosa.griffinlim(x, momentum=-1)


def test_griffinlim_momentum_warn():
    x = np.zeros((33, 3))
    with pytest.warns(UserWarning):
        librosa.griffinlim(x, momentum=2)


@pytest.mark.parametrize("ext", ["wav", "mp3"])
def test_get_samplerate(ext):

    path = os.path.join("tests", "data", os.path.extsep.join(["test1_22050", ext]))

    sr = librosa.get_samplerate(path)
    assert sr == 22050


def test_get_samplerate_soundfile():

    path = os.path.join("tests", "data", os.path.extsep.join(["test1_22050", "wav"]))

    sfo = soundfile.SoundFile(path)

    sr2 = librosa.get_samplerate(sfo)

    assert sr2 == 22050


@pytest.fixture(params=["as_file", "as_string", "as_sfo"])
def path(request):

    # test data is stereo, int 16
    path = os.path.join("tests", "data", "test1_22050.wav")

    if request.param == "as_string":
        yield path
    elif request.param == "as_file":
        with open(path, "rb") as f:
            yield f
    elif request.param == "as_sfo":
        with soundfile.SoundFile(path) as f:
            yield f


@pytest.mark.parametrize(
    "block_length,frame_length,hop_length",
    [(0, 1024, 512), (10, 0, 512), (10, 1024, 0)],
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stream_badparam(path, block_length, frame_length, hop_length):
    next(
        librosa.stream(
            path,
            block_length=block_length,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    )


@pytest.mark.parametrize("block_length", [10, np.int64(30)])
@pytest.mark.parametrize("frame_length", [1024, np.int64(2048)])
@pytest.mark.parametrize("hop_length", [512, np.int64(1024)])
@pytest.mark.parametrize("mono", [False, True])
@pytest.mark.parametrize("offset", [0.0, 2.0])
@pytest.mark.parametrize("duration", [None, 1.0])
@pytest.mark.parametrize("fill_value", [None, 999.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stream(
    path,
    block_length,
    frame_length,
    hop_length,
    mono,
    offset,
    duration,
    fill_value,
    dtype,
):

    stream = librosa.stream(
        path,
        block_length=block_length,
        frame_length=frame_length,
        hop_length=hop_length,
        dtype=dtype,
        mono=mono,
        offset=offset,
        duration=duration,
        fill_value=fill_value,
    )

    y_frame_stream = []
    target_length = frame_length + (block_length - 1) * hop_length

    for y_block in stream:
        # Check the dtype
        assert y_block.dtype == dtype

        # Check for mono
        if mono:
            assert y_block.ndim == 1
        else:
            assert y_block.ndim == 2
            assert y_block.shape[0] == 2

        # Check the length
        if fill_value is None:
            assert y_block.shape[-1] <= target_length
        else:
            assert y_block.shape[-1] == target_length

        # frame this for easy checking
        y_b_mono = librosa.to_mono(y_block)
        if len(y_b_mono) >= frame_length:
            y_b_frame = librosa.util.frame(
                y_b_mono, frame_length=frame_length, hop_length=hop_length
            )
            y_frame_stream.append(y_b_frame)

    # Concatenate the framed blocks together
    y_frame_stream = np.concatenate(y_frame_stream, axis=1)

    # Load the reference data.
    # We'll cast to mono here to simplify checking

    # File objects have to be reset before loading
    if hasattr(path, "seek"):
        path.seek(0)

    y_full, sr = librosa.load(
        path, sr=None, dtype=dtype, mono=True, offset=offset, duration=duration
    )
    # First, check the rate
    y_frame = librosa.util.frame(
        y_full, frame_length=frame_length, hop_length=hop_length
    )

    # Raw audio will not be padded
    n = y_frame.shape[1]
    assert np.allclose(y_frame[:, :n], y_frame_stream[:, :n])


@pytest.mark.parametrize("mu", [15, 31, 255])
@pytest.mark.parametrize("quantize", [False, True])
def test_mu_compress(mu, quantize):

    x = np.linspace(-1, 1, num=5, endpoint=True)
    y = librosa.mu_compress(x, mu=mu, quantize=quantize)

    # Do we preserve sign?
    assert np.all(np.sign(y) == np.sign(x))

    if quantize:
        # Check that y is between -(mu+1) and mu
        assert np.all(y >= -(mu + 1)) and np.all(y <= mu)
        assert np.issubdtype(y.dtype, np.integer)
    else:
        assert np.all(y >= -1) and np.all(y <= 1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mu_compress_badmu():
    x = np.linspace(-1, 1, num=5)
    librosa.mu_compress(x, mu=-1)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "x",
    [
        np.linspace(-2, 1, num=5, endpoint=True),
        np.linspace(-1, 2, num=5, endpoint=True),
    ],
)
def test_mu_compress_badx(x):
    librosa.mu_compress(x)


@pytest.mark.parametrize("mu", [15, 31, 255])
@pytest.mark.parametrize("quantize", [False, True])
def test_mu_expand(mu, quantize):
    # Really this is an integration test for companding. YOLO

    x = np.linspace(-1, 1, num=5, endpoint=True)
    y = librosa.mu_compress(x, mu=mu, quantize=quantize)

    z = librosa.mu_expand(y, mu=mu, quantize=quantize)

    assert np.all(z <= 1) and np.all(z >= -1)
    assert np.all(np.sign(z) == np.sign(x))

    if not quantize:
        # Without quantization, companding should be exact
        assert np.allclose(x, z)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mu_expand_badmu():
    x = np.linspace(-1, 1, num=5)
    librosa.mu_expand(x, mu=-1)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "x",
    [
        np.linspace(-2, 1, num=5, endpoint=True),
        np.linspace(-1, 2, num=5, endpoint=True),
    ],
)
def test_mu_expand_badx(x):
    librosa.mu_expand(x, quantize=False)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stft_bad_prealloc_shape():
    y = np.zeros(22050)

    # Output shape here is incorrect, and should trigger a failure
    S1 = librosa.stft(
        y, n_fft=512, hop_length=128, out=np.zeros((100, 10), dtype=complex)
    )


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stft_bad_prealloc_dtype():
    y = np.zeros(22050)
    D = librosa.stft(y)

    Dbad = np.zeros(D.shape, dtype=np.float32)
    librosa.stft(y, out=Dbad)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_istft_bad_prealloc_shape():
    D = np.zeros((1025, 5), dtype=np.complex64)
    librosa.istft(D, out=np.zeros(100))


# Tests to force audioread decoding
def test_load_force_audioread():
    path = os.path.join("tests", "data", "test2_8000.mkv")
    with warnings.catch_warnings(record=True) as out:
        y, sr = librosa.load(path)

        assert len(out) > 0
        assert "audioread" in str(out[0].message).lower()


@pytest.mark.filterwarnings("ignore:PySoundFile failed")
def test_get_duration_audioread():
    path = os.path.join("tests", "data", "test2_8000.mkv")
    duration = librosa.get_duration(path=path)

    # Duration is 30.2 seconds if using ffmpeg
    # Duration is 30.23 seconds otherwise (eg gstreamer)
    # To avoid floating point issues, we'll just check that it's close
    assert np.isclose(duration, 30.2, atol=0.1)


@pytest.mark.filterwarnings("ignore:PySoundFile failed")
def test_get_samplerate_audioread():
    path = os.path.join("tests", "data", "test2_8000.mkv")
    sr = librosa.get_samplerate(path=path)

    assert sr == 8000


def test_f0_harmonics_static():

    freqs = np.arange(8)

    data = np.multiply.outer(freqs, np.arange(2, 5))
    # array([[ 0,  0,  0],
    #        [ 2,  3,  4],
    #        [ 4,  6,  8],
    #        [ 6,  9, 12],
    #        [ 8, 12, 16],
    #        [10, 15, 20],
    #        [12, 18, 24],
    #        [14, 21, 28]])

    f0 = np.array([1, 2, 0])
    harmonics = [0.5, 1, 3]

    yh = librosa.f0_harmonics(data, f0=f0, freqs=freqs, harmonics=harmonics)

    assert yh.shape[0] == len(harmonics)
    assert yh.shape[1:] == data.shape[1:]

    # The 1 here comes from linear interpolation of the 0.5 subharmonic
    # All else are data[1,2,3]
    assert np.allclose(yh[:, 0], [1, 2, 6])
    # Values here come from f0 = 2
    assert np.allclose(yh[:, 1], [3, 6, 18])
    # Last frame has f0 = 0, so all harmonics will evaluate to 0
    assert np.allclose(yh[:, 2], [0, 0, 0])


def test_f0_harmonics_dynamic():

    # Cook up a dynamic frequency grid
    freqs = np.add.outer(np.arange(8), np.arange(3))
    # array([[0, 1, 2],
    #        [1, 2, 3],
    #        [2, 3, 4],
    #        [3, 4, 5],
    #        [4, 5, 6],
    #        [5, 6, 7],
    #        [6, 7, 8],
    #        [7, 8, 9]])

    # Broadcast multiply to make some measurements
    data = freqs * np.arange(2, 5)
    # array([[ 0,  3,  8],
    #        [ 2,  6, 12],
    #        [ 4,  9, 16],
    #        [ 6, 12, 20],
    #        [ 8, 15, 24],
    #        [10, 18, 28],
    #        [12, 21, 32],
    #        [14, 24, 36]])

    # f0 at each frame
    f0 = np.array([2, 4, 5])

    # Harmonics
    harmonics = [0.5, 1, 2]

    # Interpolate
    yh = librosa.f0_harmonics(data, f0=f0, freqs=freqs, harmonics=harmonics)

    assert yh.shape[0] == len(harmonics)
    assert yh.shape[1:] == data.shape[1:]

    # f0 in frame 0 = 2, f0 * h => [1, 2, 4]
    # data[f0 * h] = [2, 4, 8]
    assert np.allclose(yh[:, 0], [2, 4, 8])
    # f0 in frame 1 = 4, f0 * h => [2, 4, 8]
    # data[f0 * h] = [6, 12, 24]
    assert np.allclose(yh[:, 1], [6, 12, 24])
    # f0 in frame 2 = 5, f0 * h = [2.5, 5, 10]
    # interpolation happens here
    # last frequency falls off the edge of the frequency grid, filled as 0
    assert np.allclose(yh[:, 2], [10, 20, 0])


def test_f0_harmonics_1d_nonunique():
    freqs = np.arange(-8, 8) ** 2
    data = np.multiply.outer(freqs, np.arange(5))

    h = [1, 2, 3]
    f0 = np.ones(data.shape[-1])
    with pytest.warns(UserWarning):
        librosa.f0_harmonics(data, freqs=freqs, harmonics=h, f0=f0)


def test_f0_harmonics_2d_nonunique():
    freqs = np.add.outer(np.arange(-8, 8) ** 2, np.arange(5))

    data = freqs * np.arange(5)

    h = [1, 2, 3]
    f0 = np.ones(data.shape[-1])
    with pytest.warns(UserWarning):
        librosa.f0_harmonics(data, freqs=freqs, harmonics=h, f0=f0)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_f0_harmonics_incompat():

    # Freq axis does not match data shape
    freqs = np.arange(5)
    data = np.zeros((6, 7))
    f0 = np.arange(7)
    harmonics = np.arange(1, 3)

    librosa.f0_harmonics(data, freqs=freqs, harmonics=harmonics, f0=f0)
