#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import print_function
import warnings
import numpy as np

from nose.tools import raises, eq_

import librosa

from test_core import load, srand

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

__EXAMPLE_FILE = 'data/test1_22050.wav'
warnings.resetwarnings()
warnings.simplefilter('always')


# utils submodule
def test_delta():
    # Note: this test currently only checks first-order differences
    #       and width=3 filters

    def __test(width, order, axis, x):
        # Compare trimmed and untrimmed versions
        delta = librosa.feature.delta(x,
                                      width=width,
                                      order=order,
                                      axis=axis,
                                      trim=False)
        delta_t = librosa.feature.delta(x,
                                        width=width,
                                        order=order,
                                        axis=axis,
                                        trim=True)

        # Check that trimming matches the expected shape
        eq_(x.shape, delta_t.shape)

        # Check that trimming gives the right values in the right places
        _s = [slice(None)] * delta.ndim
        _s[axis] = slice(- width//2 - x.shape[axis], -(width//2)-1)
        delta_retrim = delta[_s]
        assert np.allclose(delta_t, delta_retrim)

        # Once we're sufficiently far into the signal (ie beyond half_len)
        # (x + delta_t)[t] should approximate x[t+1] if x is actually linear
        slice_orig = [slice(None)] * x.ndim
        slice_out = [slice(None)] * delta.ndim
        slice_orig[axis] = slice(width//2 + 1, -width//2 + 1)
        slice_out[axis] = slice(width//2, -width//2)
        assert np.allclose((x + delta_t)[slice_out], x[slice_orig])

    x = np.vstack([np.arange(100.0)] * 3)

    for width in range(-1, 8):
        for slope in np.linspace(-2, 2, num=6):
            for bias in [-10, 0, 10]:
                for order in [0, 1]:
                    for axis in range(x.ndim):
                        tf = __test
                        if width < 3 or np.mod(width, 2) != 1:
                            tf = raises(librosa.ParameterError)(__test)
                        if order != 1:
                            tf = raises(librosa.ParameterError)(__test)
                        yield tf, width, order, axis, slope * x + bias


def test_stack_memory():

    def __test(data, n_steps, delay):
        data_stack = librosa.feature.stack_memory(data,
                                                  n_steps=n_steps,
                                                  delay=delay)

        # If we're one-dimensional, reshape for testing
        if data.ndim == 1:
            data = data.reshape((1, -1))

        d, t = data.shape

        eq_(data_stack.shape[0], n_steps * d)
        eq_(data_stack.shape[1], t)

        for i in range(d):
            for step in range(1, n_steps):
                assert np.allclose(data[i, :- step * delay],
                                   data_stack[step * d + i, step * delay:])

    srand()

    for ndim in [1, 2]:
        data = np.random.randn(* ([5] * ndim))

        for n_steps in [-1, 0, 1, 2, 3, 4]:
            for delay in [-1, 0, 1, 2, 4]:
                tf = __test
                if n_steps < 1:
                    tf = raises(librosa.ParameterError)(__test)
                if delay < 1:
                    tf = raises(librosa.ParameterError)(__test)
                yield tf, data, n_steps, delay


# spectral submodule
def test_spectral_centroid_synthetic():

    k = 5

    def __test(S, freq, sr, n_fft):
        cent = librosa.feature.spectral_centroid(S=S, freq=freq)

        if freq is None:
            freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        assert np.allclose(cent, freq[k])

    srand()
    # construct a fake spectrogram
    sr = 22050
    n_fft = 1024
    S = np.zeros((1 + n_fft // 2, 10))

    S[k, :] = 1.0

    yield __test, S, None, sr, n_fft

    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    yield __test, S, freq, sr, n_fft

    # And if we modify the frequencies
    freq *= 3
    yield __test, S, freq, sr, n_fft

    # Or if we make up random frequencies for each frame
    freq = np.random.randn(*S.shape)
    yield __test, S, freq, sr, n_fft


def test_spectral_centroid_errors():

    @raises(librosa.ParameterError)
    def __test(S):
        librosa.feature.spectral_centroid(S=S)

    S = - np.ones((513, 10))
    yield __test, S

    S = - np.ones((513, 10)) * 1.j
    yield __test, S


def test_spectral_centroid_empty():

    def __test(y, sr, S):
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, S=S)
        assert not np.any(cent)

    sr = 22050
    y = np.zeros(3 * sr)
    yield __test, y, sr, None

    S = np.zeros((1025, 10))
    yield __test, None, sr, S


def test_spectral_bandwidth_synthetic():
    # This test ensures that a signal confined to a single frequency bin
    # always achieves 0 bandwidth
    k = 5

    def __test(S, freq, sr, n_fft, norm, p):
        bw = librosa.feature.spectral_bandwidth(S=S, freq=freq, norm=norm, p=p)

        assert not np.any(bw)

    srand()
    # construct a fake spectrogram
    sr = 22050
    n_fft = 1024
    S = np.zeros((1 + n_fft // 2, 10))
    S[k, :] = 1.0

    for norm in [False, True]:
        for p in [1, 2]:
            # With vanilla frequencies
            yield __test, S, None, sr, n_fft, norm, p

            # With explicit frequencies
            freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            yield __test, S, freq, sr, n_fft, norm, p

            # And if we modify the frequencies
            freq = 3 * librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            yield __test, S, freq, sr, n_fft, norm, p

            # Or if we make up random frequencies for each frame
            freq = np.random.randn(*S.shape)
            yield __test, S, freq, sr, n_fft, norm, p


def test_spectral_bandwidth_errors():

    @raises(librosa.ParameterError)
    def __test(S):
        librosa.feature.spectral_bandwidth(S=S)

    S = - np.ones((513, 10))
    yield __test, S

    S = - np.ones((513, 10)) * 1.j
    yield __test, S


def test_spectral_rolloff_synthetic():

    srand()

    sr = 22050
    n_fft = 2048

    def __test(S, freq, pct):

        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, freq=freq,
                                                   roll_percent=pct)

        if freq is None:
            freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        idx = np.floor(pct * freq.shape[0]).astype(int)
        assert np.allclose(rolloff, freq[idx])

    S = np.ones((1 + n_fft // 2, 10))

    for pct in [0.25, 0.5, 0.95]:
        # Implicit frequencies
        yield __test, S, None, pct

        # Explicit frequencies
        freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        yield __test, S, freq, pct

        # And time-varying frequencies
        freq = np.cumsum(np.abs(np.random.randn(*S.shape)), axis=0)
        yield __test, S, freq, pct


def test_spectral_rolloff_errors():

    @raises(librosa.ParameterError)
    def __test(S, p):
        librosa.feature.spectral_rolloff(S=S, roll_percent=p)

    S = - np.ones((513, 10))
    yield __test, S, 0.95

    S = - np.ones((513, 10)) * 1.j
    yield __test, S, 0.95

    S = np.ones((513, 10))
    yield __test, S, -1

    S = np.ones((513, 10))
    yield __test, S, 2


def test_spectral_contrast_log():
    # We already have a regression test for linear energy difference
    # This test just does a sanity-check on the log-scaled version

    y, sr = librosa.load(__EXAMPLE_FILE)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, linear=False)

    assert not np.any(contrast < 0)


def test_spectral_contrast_errors():

    @raises(librosa.ParameterError)
    def __test(S, freq, fmin, n_bands, quantile):
        librosa.feature.spectral_contrast(S=S,
                                          freq=freq,
                                          fmin=fmin,
                                          n_bands=n_bands,
                                          quantile=quantile)

    S = np.ones((1025, 10))

    # ill-shaped frequency set: scalar
    yield __test, S, 0, 200, 6, 0.02

    # ill-shaped frequency set: wrong-length vector
    yield __test, S, np.zeros((S.shape[0]+1,)), 200, 6, 0.02

    # ill-shaped frequency set: matrix
    yield __test, S, np.zeros(S.shape), 200, 6, 0.02

    # negative fmin
    yield __test, S, None, -1, 6, 0.02

    # zero fmin
    yield __test, S, None, 0, 6, 0.02

    # negative n_bands
    yield __test, S, None, 200, -1, 0.02

    # bad quantile
    yield __test, S, None, 200, 6, -1

    # bad quantile
    yield __test, S, None, 200, 6, 2


def test_rmse():

    def __test(n):
        S = np.ones((n, 5))

        # RMSE of an all-ones band is 1
        rmse = librosa.feature.rmse(S=S)

        assert np.allclose(rmse, np.ones_like(rmse))

    def __test_consistency(frame_length, hop_length):
        y, sr = librosa.load(__EXAMPLE_FILE, sr=None)

        # Ensure audio is divisible into frame size.
        y = librosa.util.fix_length(y, y.size - y.size % frame_length)
        assert y.size % frame_length == 0

        # STFT magnitudes with a constant windowing function and no centering.
        S = librosa.magphase(librosa.stft(y,
                                          n_fft=frame_length,
                                          hop_length=hop_length,
                                          window=np.ones,
                                          center=False))[0]

        # Try both RMS methods.
        rms1 = librosa.feature.rmse(S=S, frame_length=frame_length,
                                    hop_length=hop_length)
        rms2 = librosa.feature.rmse(y=y, frame_length=frame_length,
                                    hop_length=hop_length)

        # Normalize envelopes.
        rms1 /= rms1.max()
        rms2 /= rms2.max()

        # Ensure results are similar.
        np.testing.assert_allclose(rms1, rms2, rtol=1e-2)

    for frame_length in [2048, 4096]:
        for hop_length in [128, 512, 1024]:
            yield __test_consistency, frame_length, hop_length

    for n in range(10, 100, 10):
        yield __test, n


def test_rmse_nfft():

    warnings.resetwarnings()
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as out:
        librosa.feature.rmse(y=np.zeros(8192), n_fft=1024)
        assert len(out) > 0
        assert out[0].category is DeprecationWarning
        assert 'renamed' in str(out[0].message).lower()


def test_zcr_synthetic():

    def __test_zcr(rate, y, frame_length, hop_length, center):
        zcr = librosa.feature.zero_crossing_rate(y,
                                                 frame_length=frame_length,
                                                 hop_length=hop_length,
                                                 center=center)

        # We don't care too much about the edges if there's padding
        if center:
            zcr = zcr[:, frame_length//2:-frame_length//2]

        # We'll allow 1% relative error
        assert np.allclose(zcr, rate, rtol=1e-2)

    sr = 16384
    for period in [32, 16, 8, 4, 2]:
        y = np.ones(sr)
        y[::period] = -1
        # Every sign flip induces two crossings
        rate = 2./period
        # 1+2**k so that we get both sides of the last crossing
        for frame_length in [513, 2049]:
            for hop_length in [128, 256]:
                for center in [False, True]:
                    yield __test_zcr, rate, y, frame_length, hop_length, center


def test_poly_features_synthetic():

    srand()
    sr = 22050
    n_fft = 2048

    def __test(S, coeffs, freq):

        order = coeffs.shape[0] - 1
        p = librosa.feature.poly_features(S=S, sr=sr, n_fft=n_fft,
                                          order=order, freq=freq)

        for i in range(S.shape[-1]):
            assert np.allclose(coeffs, p[::-1, i].squeeze())

    def __make_data(coeffs, freq):
        S = np.zeros_like(freq)
        for i, c in enumerate(coeffs):
            S = S + c * freq**i

        S = S.reshape((freq.shape[0], -1))
        return S

    for order in range(1, 3):
        freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        coeffs = np.atleast_1d(np.arange(1, 1+order))

        # First test: vanilla
        S = __make_data(coeffs, freq)
        yield __test, S, coeffs, None

        # And with explicit frequencies
        yield __test, S, coeffs, freq

        # And with alternate frequencies
        freq = freq**2.0
        S = __make_data(coeffs, freq)
        yield __test, S, coeffs, freq

        # And multi-dimensional
        freq = np.cumsum(np.abs(np.random.randn(1 + n_fft//2, 2)), axis=0)
        S = __make_data(coeffs, freq)
        yield __test, S, coeffs, freq


def test_tonnetz():
    y, sr = librosa.load(librosa.util.example_audio_file())
    tonnetz_chroma = np.load("data/feature-tonnetz-chroma.npy")
    tonnetz_msaf = np.load("data/feature-tonnetz-msaf.npy")

    # Use cqt chroma
    def __audio():
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        assert tonnetz.shape[0] == 6

    # Use pre-computed chroma
    def __stft():
        tonnetz = librosa.feature.tonnetz(chroma=tonnetz_chroma)
        assert tonnetz.shape[1] == tonnetz_chroma.shape[1]
        assert tonnetz.shape[0] == 6
        assert np.allclose(tonnetz_msaf, tonnetz)

    def __cqt():
        # Use high resolution cqt chroma
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=24)
        tonnetz = librosa.feature.tonnetz(chroma=chroma_cqt)
        assert tonnetz.shape[1] == chroma_cqt.shape[1]
        assert tonnetz.shape[0] == 6
        # Using stft chroma won't generally match cqt chroma
        # skip the equivalence check

    # Call the function with not enough parameters
    yield (raises(librosa.ParameterError)(librosa.feature.tonnetz))
    yield __audio
    yield __stft
    yield __cqt


def test_tempogram_fail():

    @raises(librosa.ParameterError)
    def __test(y, sr, onset_envelope, hop_length, win_length, center, window, norm):

        librosa.feature.tempogram(y=y,
                                  sr=sr,
                                  onset_envelope=onset_envelope,
                                  hop_length=hop_length,
                                  win_length=win_length,
                                  center=center,
                                  window=window,
                                  norm=norm)

    sr = 22050
    hop_length = 512
    duration = 10

    y = np.zeros(duration * sr)

    # Fail when no input is provided
    yield __test, None, sr, None, hop_length, 384, True, 'hann', np.inf

    # Fail when win_length is too small
    for win_length in [-384, -1, 0]:
        yield __test, y, sr, None, hop_length, win_length, True, 'hann', np.inf

    # Fail when len(window) != win_length
    yield __test, y, sr, None, hop_length, 384, True, np.ones(win_length + 1), np.inf


def test_tempogram_audio():

    def __test(y, sr, oenv, hop_length):

        # Get the tempogram from audio
        t1 = librosa.feature.tempogram(y=y, sr=sr,
                                       onset_envelope=None,
                                       hop_length=hop_length)

        # Get the tempogram from oenv
        t2 = librosa.feature.tempogram(y=None, sr=sr,
                                       onset_envelope=oenv,
                                       hop_length=hop_length)

        # Make sure it works when both are provided
        t3 = librosa.feature.tempogram(y=y, sr=sr,
                                       onset_envelope=oenv,
                                       hop_length=hop_length)

        # And that oenv overrides y
        t4 = librosa.feature.tempogram(y=0 * y, sr=sr,
                                       onset_envelope=oenv,
                                       hop_length=hop_length)

        assert np.allclose(t1, t2)
        assert np.allclose(t1, t3)
        assert np.allclose(t1, t4)

    y, sr = librosa.load(__EXAMPLE_FILE)

    for hop_length in [512, 1024]:
        oenv = librosa.onset.onset_strength(y=y,
                                            sr=sr,
                                            hop_length=hop_length)

        yield __test, y, sr, oenv, hop_length


def test_tempogram_odf():

    sr = 22050
    hop_length = 512
    duration = 8

    def __test_equiv(tempo, center):
        odf = np.zeros(duration * sr // hop_length)
        spacing = sr * 60. // (hop_length * tempo)
        odf[::int(spacing)] = 1

        odf_ac = librosa.autocorrelate(odf)

        tempogram = librosa.feature.tempogram(onset_envelope=odf,
                                              sr=sr,
                                              hop_length=hop_length,
                                              win_length=len(odf),
                                              window=np.ones,
                                              center=center,
                                              norm=None)

        idx = 0
        if center:
            idx = len(odf)//2

        assert np.allclose(odf_ac, tempogram[:, idx])

    # Generate a synthetic onset envelope
    def __test_peaks(tempo, win_length, window, norm):
        # Generate an evenly-spaced pulse train
        odf = np.zeros(duration * sr // hop_length)
        spacing = sr * 60. // (hop_length * tempo)
        odf[::int(spacing)] = 1

        tempogram = librosa.feature.tempogram(onset_envelope=odf,
                                              sr=sr,
                                              hop_length=hop_length,
                                              win_length=win_length,
                                              window=window,
                                              norm=norm)

        # Check the shape of the output
        eq_(tempogram.shape[0], win_length)

        eq_(tempogram.shape[1], len(odf))

        # Mean over time to wash over the boundary padding effects
        idx = np.where(librosa.util.localmax(tempogram.max(axis=1)))[0]

        # Indices should all be non-zero integer multiples of spacing
        assert np.allclose(idx, spacing * np.arange(1, 1 + len(idx)))

    for tempo in [60, 90, 120, 160, 200]:
        for center in [False, True]:
            yield __test_equiv, tempo, center

        for win_length in [192, 384]:
            for window in ['hann', np.ones, np.ones(win_length)]:
                for norm in [None, 1, 2, np.inf]:
                    yield __test_peaks, tempo, win_length, window, norm


def test_cens():
    # load CQT data from Chroma Toolbox
    ct_cqt = load(os.path.join('data', 'features-CT-cqt.mat'))

    fn_ct_chroma_cens = ['features-CT-CENS_9-2.mat',
                         'features-CT-CENS_21-5.mat',
                         'features-CT-CENS_41-1.mat']

    cens_params = [(9, 2), (21, 5), (41, 1)]

    for cur_test_case, cur_fn_ct_chroma_cens in enumerate(fn_ct_chroma_cens):
        win_len_smooth = cens_params[cur_test_case][0]
        downsample_smooth = cens_params[cur_test_case][1]

        # plug into librosa cens computation
        lr_chroma_cens = librosa.feature.chroma_cens(C=ct_cqt['f_cqt'],
                                                     win_len_smooth=win_len_smooth,
                                                     fmin=librosa.core.midi_to_hz(1),
                                                     bins_per_octave=12,
                                                     n_octaves=10)

        # leaving out frames to match chroma toolbox behaviour
        # lr_chroma_cens = librosa.resample(lr_chroma_cens, orig_sr=1, target_sr=1/downsample_smooth)
        lr_chroma_cens = lr_chroma_cens[:, ::downsample_smooth]

        # load CENS-41-1 features
        ct_chroma_cens = load(os.path.join('data', cur_fn_ct_chroma_cens))

        assert np.allclose(ct_chroma_cens['f_CENS'], lr_chroma_cens, rtol=1e-15, atol=1e-15)
