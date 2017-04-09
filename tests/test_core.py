#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa core (__init__.py)
#
# Run me as follows:
#   cd tests/
#   nosetests -v --with-coverage --cover-package=librosa
#

from __future__ import print_function
# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import glob
import numpy as np
import scipy.io
import six
from nose.tools import eq_, raises, make_decorator

import warnings
warnings.resetwarnings()
warnings.simplefilter('always')


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


def test_load():
    # Note: this does not test resampling.
    # That is a separate unit test.

    def __test(infile):
        DATA = load(infile)
        y, sr = librosa.load(DATA['wavfile'][0],
                             sr=None,
                             mono=DATA['mono'])

        # Verify that the sample rate is correct
        eq_(sr, DATA['sr'])

        assert np.allclose(y, DATA['y'])

    for infile in files('data/core-load-*.mat'):
        yield (__test, infile)
    pass


def test_load_resample():

    sr_target = 16000
    offset = 10
    duration = 5

    def __test(res_type):
        y_native, sr = librosa.load(librosa.util.example_audio_file(),
                                    sr=None,
                                    offset=offset,
                                    duration=duration,
                                    res_type=res_type)

        y2 = librosa.resample(y_native, sr, sr_target, res_type=res_type)

        y, _ = librosa.load(librosa.util.example_audio_file(),
                            sr=sr_target,
                            offset=offset,
                            duration=duration,
                            res_type=res_type)

        assert np.allclose(y2, y)

    for res_type in ['kaiser_fast', 'kaiser_best', 'scipy']:
        yield __test, res_type


def test_segment_load():

    sample_len = 2003
    fs = 44100
    test_file = 'data/test1_44100.wav'
    y, sr = librosa.load(test_file, sr=None, mono=False,
                         offset=0., duration=sample_len/float(fs))

    eq_(y.shape[-1], sample_len)

    y2, sr = librosa.load(test_file, sr=None, mono=False)
    assert np.allclose(y, y2[:, :sample_len])

    sample_offset = 2048
    y, sr = librosa.load(test_file, sr=None, mono=False,
                         offset=sample_offset/float(fs), duration=1.0)

    eq_(y.shape[-1], fs)

    y2, sr = librosa.load(test_file, sr=None, mono=False)
    assert np.allclose(y, y2[:, sample_offset:sample_offset+fs])


def test_resample_mono():

    def __test(y, sr_in, sr_out, res_type, fix):

        y2 = librosa.resample(y, sr_in, sr_out,
                              res_type=res_type,
                              fix=fix)

        # First, check that the audio is valid
        librosa.util.valid_audio(y2, mono=True)

        # If it's a no-op, make sure the signal is untouched
        if sr_out == sr_in:
            assert np.allclose(y, y2)

        # Check buffer contiguity
        assert y2.flags['C_CONTIGUOUS']

        # Check that we're within one sample of the target length
        target_length = y.shape[-1] * sr_out // sr_in
        assert np.abs(y2.shape[-1] - target_length) <= 1

    for infile in ['data/test1_44100.wav',
                   'data/test1_22050.wav',
                   'data/test2_8000.wav']:
        y, sr_in = librosa.load(infile, sr=None, duration=5)

        for sr_out in [8000, 22050]:
            for res_type in ['kaiser_best', 'kaiser_fast', 'scipy']:
                for fix in [False, True]:
                    yield (__test, y, sr_in, sr_out, res_type, fix)


def test_resample_stereo():

    def __test(y, sr_in, sr_out, res_type, fix):

        y2 = librosa.resample(y, sr_in, sr_out,
                              res_type=res_type,
                              fix=fix)

        # First, check that the audio is valid
        librosa.util.valid_audio(y2, mono=False)

        eq_(y2.ndim, y.ndim)

        # If it's a no-op, make sure the signal is untouched
        if sr_out == sr_in:
            assert np.allclose(y, y2)

        # Check buffer contiguity
        assert y2.flags['C_CONTIGUOUS']

        # Check that we're within one sample of the target length
        target_length = y.shape[-1] * sr_out // sr_in
        assert np.abs(y2.shape[-1] - target_length) <= 1

    y, sr_in = librosa.load('data/test1_44100.wav', mono=False, sr=None, duration=5)

    for sr_out in [8000, 22050]:
        for res_type in ['kaiser_fast', 'scipy']:
            for fix in [False, True]:
                yield __test, y, sr_in, sr_out, res_type, fix


def test_resample_scale():

    def __test(sr_in, sr_out, res_type, y):

        y2 = librosa.resample(y, sr_in, sr_out,
                              res_type=res_type,
                              scale=True)

        # First, check that the audio is valid
        librosa.util.valid_audio(y2, mono=True)

        n_orig = np.sqrt(np.sum(np.abs(y)**2))
        n_res = np.sqrt(np.sum(np.abs(y2)**2))

        # If it's a no-op, make sure the signal is untouched
        assert np.allclose(n_orig, n_res, atol=1e-2), (n_orig, n_res)

    y, sr_in = librosa.load('data/test1_22050.wav', mono=True, sr=None, duration=3)

    for res_type in ['scipy', 'kaiser_best', 'kaiser_fast']:
        for sr_out in [11025, 22050, 44100]:
            yield __test, sr_in, sr_out, res_type, y
            yield __test, sr_out, sr_in, res_type, y


def test_stft():

    def __test(infile):
        DATA = load(infile)

        # Load the file
        (y, sr) = librosa.load(DATA['wavfile'][0], sr=None, mono=True)

        if DATA['hann_w'][0, 0] == 0:
            # Set window to ones, swap back to nfft
            window = np.ones
            win_length = None

        else:
            window = 'hann'
            win_length = DATA['hann_w'][0, 0]

        # Compute the STFT
        D = librosa.stft(y,
                         n_fft=DATA['nfft'][0, 0].astype(int),
                         hop_length=DATA['hop_length'][0, 0].astype(int),
                         win_length=win_length,
                         window=window,
                         center=False)

        assert np.allclose(D, DATA['D'])

    for infile in files('data/core-stft-*.mat'):
        yield (__test, infile)


def test_ifgram():

    def __test(infile):
        DATA = load(infile)

        y, sr = librosa.load(DATA['wavfile'][0], sr=None, mono=True)

        # Compute the IFgram
        F, D = librosa.ifgram(y,
                              n_fft=DATA['nfft'][0, 0].astype(int),
                              hop_length=DATA['hop_length'][0, 0].astype(int),
                              win_length=DATA['hann_w'][0, 0].astype(int),
                              sr=DATA['sr'][0, 0].astype(int),
                              ref_power=0.0,
                              clip=False,
                              center=False)

        # D fails to match here because of fftshift()
        # assert np.allclose(D, DATA['D'])
        assert np.allclose(F, DATA['F'], rtol=1e-1, atol=1e-1)

    for infile in files('data/core-ifgram-*.mat'):
        yield (__test, infile)


def test_ifgram_matches_stft():

    y, sr = librosa.load('data/test1_22050.wav')

    def __test(n_fft, hop_length, win_length, center, norm, dtype):
        D_stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, center=center,
                              dtype=dtype)

        _, D_ifgram = librosa.ifgram(y, sr, n_fft=n_fft,
                                     hop_length=hop_length,
                                     win_length=win_length, center=center,
                                     norm=norm, dtype=dtype)

        if norm:
            # STFT doesn't do window normalization;
            # let's just ignore the relative scale to make this easy
            D_stft = librosa.util.normalize(D_stft, axis=0)
            D_ifgram = librosa.util.normalize(D_ifgram, axis=0)

        assert np.allclose(D_stft, D_ifgram)

    for n_fft in [1024, 2048]:
        for hop_length in [None, n_fft // 2, n_fft // 4]:
            for win_length in [None, n_fft // 2, n_fft // 4]:
                for center in [False, True]:
                    for norm in [False, True]:
                        for dtype in [np.complex64, np.complex128]:
                            yield (__test, n_fft, hop_length, win_length,
                                   center, norm, dtype)


def test_ifgram_if():

    y, sr = librosa.load('data/test1_22050.wav')

    def __test(ref, clip):

        F, D = librosa.ifgram(y, sr=sr, ref_power=ref, clip=clip)

        if clip:
            assert np.all(0 <= F) and np.all(F <= 0.5 * sr)

        assert np.all(np.isfinite(F))

    for ref in [-10, 0.0, 1e-6, np.max]:
        for clip in [False, True]:
            if six.callable(ref) or ref >= 0.0:
                tf = __test
            else:
                tf = raises(librosa.ParameterError)(__test)

            yield tf, ref, clip


def test_salience_basecase():
    (y, sr) = librosa.load('data/test1_22050.wav')
    S = np.abs(librosa.stft(y))
    freqs = librosa.core.fft_frequencies(sr)
    harms = [1]
    weights = [1.0]
    S_sal = librosa.core.salience(
        S, freqs, harms, weights, filter_peaks=False, kind='quadratic'
    )
    assert np.allclose(S_sal, S)


def test_salience_basecase2():
    (y, sr) = librosa.load('data/test1_22050.wav')
    S = np.abs(librosa.stft(y))
    freqs = librosa.core.fft_frequencies(sr)
    harms = [1, 0.5, 2.0]
    weights = [1.0, 0.0, 0.0]
    S_sal = librosa.core.salience(
        S, freqs, harms, weights, filter_peaks=False, kind='quadratic'
    )
    assert np.allclose(S_sal, S)


def test_salience_defaults():
    S = np.array([
        [0.1, 0.5, 0.0],
        [0.2, 1.2, 1.2],
        [0.0, 0.7, 0.3],
        [1.3, 3.2, 0.8]
    ])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    actual = librosa.core.salience(
        S, freqs, harms, kind='quadratic', fill_value=0.0
    )

    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 2.4, 1.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]) / 3.0
    assert np.allclose(expected, actual)


def test_salience_weights():
    S = np.array([
        [0.1, 0.5, 0.0],
        [0.2, 1.2, 1.2],
        [0.0, 0.7, 0.3],
        [1.3, 3.2, 0.8]
    ])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    weights = [1.0, 1.0, 1.0]
    actual = librosa.core.salience(
        S, freqs, harms, weights, kind='quadratic', fill_value=0.0
    )

    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 2.4, 1.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]) / 3.0
    assert np.allclose(expected, actual)


def test_salience_no_peak_filter():
    S = np.array([
        [0.1, 0.5, 0.0],
        [0.2, 1.2, 1.2],
        [0.0, 0.7, 0.3],
        [1.3, 3.2, 0.8]
    ])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    weights = [1.0, 1.0, 1.0]
    actual = librosa.core.salience(
        S, freqs, harms, weights, filter_peaks=False, kind='quadratic'
    )

    expected = np.array([
        [0.3, 1.7, 1.2],
        [0.3, 2.4, 1.5],
        [1.5, 5.1, 2.3],
        [1.3, 3.9, 1.1]
    ]) / 3.0
    assert np.allclose(expected, actual)


def test_salience_aggregate():
    S = np.array([
        [0.1, 0.5, 0.0],
        [0.2, 1.2, 1.2],
        [0.0, 0.7, 0.3],
        [1.3, 3.2, 0.8]
    ])
    freqs = np.array([50.0, 100.0, 200.0, 400.0])
    harms = [0.5, 1, 2]
    weights = [1.0, 1.0, 1.0]
    actual = librosa.core.salience(
        S, freqs, harms, weights, aggregate=np.ma.max, kind='quadratic',
        fill_value=0.0
    )

    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 1.2, 1.2],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    assert np.allclose(expected, actual)


def test_magphase():

    (y, sr) = librosa.load('data/test1_22050.wav')

    D = librosa.stft(y)

    S, P = librosa.magphase(D)

    assert np.allclose(S * P, D)


def test_istft_reconstruction():
    from scipy.signal import bartlett, hann, hamming, blackman, blackmanharris

    def __test(x, n_fft, hop_length, window, atol):
        S = librosa.core.stft(
            x, n_fft=n_fft, hop_length=hop_length, window=window)
        x_reconstructed = librosa.core.istft(
            S, hop_length=hop_length, window=window)

        L = min(len(x), len(x_reconstructed))
        x = np.resize(x, L)
        x_reconstructed = np.resize(x_reconstructed, L)

        # NaN/Inf/-Inf should not happen
        assert np.all(np.isfinite(x_reconstructed))

        # should be almost approximately reconstucted
        assert np.allclose(x, x_reconstructed, atol=atol)

    srand()
    # White noise
    x1 = np.random.randn(2 ** 15)

    # Sin wave
    x2 = np.sin(np.linspace(-np.pi, np.pi, 2 ** 15))

    # Real music signal
    x3, sr = librosa.load('data/test1_44100.wav', sr=None, mono=True)
    assert sr == 44100

    for x, atol in [(x1, 1.0e-6), (x2, 1.0e-7), (x3, 1.0e-7)]:
        for window_func in [bartlett, hann, hamming, blackman, blackmanharris]:
            for n_fft in [512, 1024, 2048, 4096]:
                win = window_func(n_fft, sym=False)
                symwin = window_func(n_fft, sym=True)
                # tests with pre-computed window fucntions
                for hop_length_denom in six.moves.range(2, 9):
                    hop_length = n_fft // hop_length_denom
                    yield (__test, x, n_fft, hop_length, win, atol)
                    yield (__test, x, n_fft, hop_length, symwin, atol)
                # also tests with passing widnow function itself
                yield (__test, x, n_fft, n_fft // 9, window_func, atol)

        # test with default paramters
        x_reconstructed = librosa.core.istft(librosa.core.stft(x))
        L = min(len(x), len(x_reconstructed))
        x = np.resize(x, L)
        x_reconstructed = np.resize(x_reconstructed, L)

        assert np.allclose(x, x_reconstructed, atol=atol)


def test_load_options():

    filename = 'data/test1_22050.wav'

    def __test(offset, duration, mono, dtype):

        y, sr = librosa.load(filename, mono=mono, offset=offset,
                             duration=duration, dtype=dtype)

        if duration is not None:
            assert np.allclose(y.shape[-1], int(sr * duration))

        if mono:
            eq_(y.ndim, 1)
        else:
            # This test file is stereo, so y.ndim should be 2
            eq_(y.ndim, 2)

        # Check the dtype
        assert np.issubdtype(y.dtype, dtype)
        assert np.issubdtype(dtype, y.dtype)

    for offset in [0, 1, 2]:
        for duration in [None, 0, 0.5, 1, 2]:
            for mono in [False, True]:
                for dtype in [np.float32, np.float64]:
                    yield __test, offset, duration, mono, dtype
    pass


def test_get_duration_wav():

    def __test_audio(filename, mono, sr, duration):
        y, sr = librosa.load(filename, sr=sr, mono=mono, duration=duration)

        duration_est = librosa.get_duration(y=y, sr=sr)

        assert np.allclose(duration_est, duration, rtol=1e-3, atol=1e-5)

    def __test_spec(filename, sr, duration, n_fft, hop_length, center):
        y, sr = librosa.load(filename, sr=sr, duration=duration)

        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=center)

        duration_est = librosa.get_duration(S=S, sr=sr, n_fft=n_fft,
                                            hop_length=hop_length,
                                            center=center)

        # We lose a little accuracy in framing without centering, so it's
        # not as precise as time-domain duration
        assert np.allclose(duration_est, duration, rtol=1e-1, atol=1e-2)

    test_file = 'data/test1_22050.wav'

    for sr in [8000, 11025, 22050]:
        for duration in [1.0, 2.5]:
            for mono in [False, True]:
                yield __test_audio, test_file, mono, sr, duration

            for n_fft in [256, 512, 1024]:
                for hop_length in [n_fft // 8, n_fft // 4, n_fft // 2]:
                    for center in [False, True]:
                        yield (__test_spec, test_file, sr,
                               duration, n_fft, hop_length, center)


def test_get_duration_filename():

    filename = 'data/test2_8000.wav'
    true_duration = 30.197625

    duration_fn = librosa.get_duration(filename=filename)
    y, sr = librosa.load(filename, sr=None)
    duration_y = librosa.get_duration(y=y, sr=sr)

    assert np.allclose(duration_fn, true_duration)
    assert np.allclose(duration_fn, duration_y)


def test_autocorrelate():

    def __test(y, truth, max_size, axis):

        ac = librosa.autocorrelate(y, max_size=max_size, axis=axis)

        my_slice = [slice(None)] * truth.ndim
        if max_size is not None and max_size <= y.shape[axis]:
            my_slice[axis] = slice(min(max_size, y.shape[axis]))

        if not np.iscomplexobj(y):
            assert not np.iscomplexobj(ac)

        assert np.allclose(ac, truth[my_slice])

    srand()
    # test with both real and complex signals
    for y in [np.random.randn(256, 256), np.exp(1.j * np.random.randn(256, 256))]:

        # Make ground-truth autocorrelations along each axis
        truth = [np.asarray([scipy.signal.fftconvolve(yi, yi[::-1].conj(),
                                                      mode='full')[len(yi)-1:] for yi in y.T]).T,
                 np.asarray([scipy.signal.fftconvolve(yi, yi[::-1].conj(),
                                                      mode='full')[len(yi)-1:] for yi in y])]

        for axis in [0, 1, -1]:
            for max_size in [None, y.shape[axis]//2, y.shape[axis], 2 * y.shape[axis]]:
                yield __test, y, truth[axis], max_size, axis


def test_to_mono():

    def __test(filename, mono):
        y, sr = librosa.load(filename, mono=mono)

        y_mono = librosa.to_mono(y)

        eq_(y_mono.ndim, 1)
        eq_(len(y_mono), y.shape[-1])

        if mono:
            assert np.allclose(y, y_mono)

    filename = 'data/test1_22050.wav'

    for mono in [False, True]:
        yield __test, filename, mono


def test_zero_crossings():

    def __test(data, threshold, ref_magnitude, pad, zp):

        zc = librosa.zero_crossings(y=data,
                                    threshold=threshold,
                                    ref_magnitude=ref_magnitude,
                                    pad=pad,
                                    zero_pos=zp)

        idx = np.flatnonzero(zc)

        if pad:
            idx = idx[1:]

        for i in idx:
            assert np.sign(data[i]) != np.sign(data[i-1])

    srand()
    data = np.random.randn(32)

    for threshold in [None, 0, 1e-10]:
        for ref_magnitude in [None, 0.1, np.max]:
            for pad in [False, True]:
                for zero_pos in [False, True]:

                    yield __test, data, threshold, ref_magnitude, pad, zero_pos


def test_pitch_tuning():

    def __test(hz, resolution, bins_per_octave, tuning):

        est_tuning = librosa.pitch_tuning(hz,
                                          resolution=resolution,
                                          bins_per_octave=bins_per_octave)

        assert np.abs(tuning - est_tuning) <= resolution

    for resolution in [1e-2, 1e-3]:
        for bins_per_octave in [12]:
            # Make up some frequencies
            for tuning in [-0.5, -0.375, -0.25, 0.0, 0.25, 0.375]:

                note_hz = librosa.midi_to_hz(tuning + np.arange(128))

                yield __test, note_hz, resolution, bins_per_octave, tuning


def test_piptrack_properties():

    def __test(S, n_fft, hop_length, fmin, fmax, threshold):

        pitches, mags = librosa.core.piptrack(S=S,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              fmin=fmin,
                                              fmax=fmax,
                                              threshold=threshold)

        # Shape tests
        eq_(S.shape, pitches.shape)
        eq_(S.shape, mags.shape)

        # Make sure all magnitudes are positive
        assert np.all(mags >= 0)

        # Check the frequency estimates for bins with non-zero magnitude
        idx = (mags > 0)
        assert np.all(pitches[idx] >= fmin)
        assert np.all(pitches[idx] <= fmax)

        # And everywhere else, pitch should be 0
        assert np.all(pitches[~idx] == 0)

    y, sr = librosa.load('data/test1_22050.wav')

    for n_fft in [2048, 4096]:
        for hop_length in [None, n_fft // 4, n_fft // 2]:
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            for fmin in [0, 100]:
                for fmax in [4000, 8000, sr // 2]:
                    for threshold in [0.1, 0.2, 0.5]:
                        yield __test, S, n_fft, hop_length, fmin, fmax, threshold


def test_piptrack_errors():

    def __test(y, sr, S, n_fft, hop_length, fmin, fmax, threshold):
        pitches, mags = librosa.piptrack(
            y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, fmin=fmin,
            fmax=fmax, threshold=threshold)

    S = np.asarray([[1, 0, 0]]).T
    np.seterr(divide='raise')
    yield __test, None, 22050, S, 4096, None, 150.0, 4000.0, 0.1


def test_piptrack():

    def __test(S, freq):
        pitches, mags = librosa.piptrack(S=S, fmin=100)

        idx = (mags > 0)

        assert len(idx) > 0

        recovered_pitches = pitches[idx]

        # We should be within one cent of the target
        assert np.all(np.abs(np.log2(recovered_pitches) - np.log2(freq)) <= 1e-2)

    sr = 22050
    duration = 3.0

    for freq in [110, 220, 440, 880]:
        # Generate a sine tone
        y = np.sin(2 * np.pi * freq * np.linspace(0, duration, num=duration*sr))
        for n_fft in [1024, 2048, 4096]:
            # Using left-aligned frames eliminates reflection artifacts at the boundaries
            S = np.abs(librosa.stft(y, n_fft=n_fft, center=False))

            yield __test, S, freq


def test_estimate_tuning():

    def __test(target_hz, resolution, bins_per_octave, tuning):

        y = np.sin(2 * np.pi * target_hz * t)
        tuning_est = librosa.estimate_tuning(resolution=resolution,
                                             bins_per_octave=bins_per_octave,
                                             y=y,
                                             sr=sr,
                                             n_fft=2048,
                                             fmin=librosa.note_to_hz('C4'),
                                             fmax=librosa.note_to_hz('G#9'))

        # Round to the proper number of decimals
        deviation = np.around(np.abs(tuning - tuning_est),
                              int(-np.log10(resolution)))

        # We'll accept an answer within three bins of the resolution
        assert deviation <= 3 * resolution

    for sr in [11025, 22050]:
        duration = 5.0

        t = np.linspace(0, duration, duration * sr)

        for resolution in [1e-2]:
            for bins_per_octave in [12]:
                # test a null-signal tuning estimate
                yield (__test, 0.0, resolution, bins_per_octave, 0.0)

                for center_note in [69, 84, 108]:
                    for tuning in np.linspace(-0.5, 0.5, 8, endpoint=False):
                        target_hz = librosa.midi_to_hz(center_note + tuning)

                        yield (__test, np.asscalar(target_hz), resolution,
                               bins_per_octave, tuning)


def test__spectrogram():

    y, sr = librosa.load('data/test1_22050.wav')

    def __test(n_fft, hop_length, power):

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**power

        S_, n_fft_ = librosa.core.spectrum._spectrogram(y=y, S=S, n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        power=power)

        # First check with all parameters
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # Then check with only the audio
        S_, n_fft_ = librosa.core.spectrum._spectrogram(y=y, n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # And only the spectrogram
        S_, n_fft_ = librosa.core.spectrum._spectrogram(S=S, n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # And only the spectrogram with no shape parameters
        S_, n_fft_ = librosa.core.spectrum._spectrogram(S=S, power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # And only the spectrogram but with incorrect n_fft
        S_, n_fft_ = librosa.core.spectrum._spectrogram(S=S, n_fft=2*n_fft,
                                                        power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

    for n_fft in [1024, 2048]:
        for hop_length in [None, 512]:
            for power in [1, 2]:
                yield __test, n_fft, hop_length, power
    assert librosa.core.spectrum._spectrogram(y)


def test_logamplitude():

    # Fake up some data
    def __test(x, ref, amin, top_db):

        y = librosa.logamplitude(x,
                                 ref=ref,
                                 amin=amin,
                                 top_db=top_db)

        assert np.isrealobj(y)
        eq_(y.shape, x.shape)

        if top_db is not None:
            assert y.min() >= y.max()-top_db

    for n in [1, 2, 10]:
        x = np.linspace(0, 2e5, num=n)
        phase = np.exp(1.j * x)

        for ref in [1.0, np.max]:
            for amin in [-1, 0, 1e-10, 1e3]:
                for top_db in [None, -10, 0, 40, 80]:
                    tf = __test
                    if amin <= 0 or (top_db is not None and top_db < 0):
                        tf = raises(librosa.ParameterError)(__test)
                    yield tf, x, ref, amin, top_db
                    yield tf, x * phase, ref, amin, top_db


def test_power_to_db_logamp():

    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    db1 = librosa.power_to_db(x**2, top_db=None)
    db2 = librosa.logamplitude(x**2, top_db=None)

    assert np.allclose(db1, db2)


def test_power_to_db():

    def __test(y_true, x, rp):
        y = librosa.power_to_db(x, ref=rp, top_db=None)

        assert np.isclose(y, y_true)

    for erp in range(-5, 6):
        for k in range(-5, 6):
            yield __test, (k-erp)*10, 10.0**k, 10.0**erp


def test_amplitude_to_db():

    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    db1 = librosa.amplitude_to_db(x, top_db=None)
    db2 = librosa.logamplitude(x**2, top_db=None)

    assert np.allclose(db1, db2)


def test_db_to_power_inv():

    srand()

    NOISE_FLOOR = 1e-5

    # Make some noise
    xp = (np.abs(np.random.randn(1000)) + NOISE_FLOOR)**2

    def __test(ref):

        db = librosa.power_to_db(xp, ref=ref, top_db=None)
        xp2 = librosa.db_to_power(db, ref=ref)

        assert np.allclose(xp, xp2)

    for ref_p in range(-3, 4):
        yield __test, 10.0**ref_p


def test_db_to_power():

    def __test(y, rp, x_true):

        x = librosa.db_to_power(y, ref=rp)

        assert np.isclose(x, x_true), (x, x_true, y, rp)

    for erp in range(-5, 6):
        for db in range(-100, 101, 10):
            yield __test, db, 10.0**erp, 10.0**erp * 10.0**(0.1 * db)


def test_db_to_amplitude_inv():

    srand()

    NOISE_FLOOR = 1e-5

    # Make some noise
    xp = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    def __test(ref):

        db = librosa.amplitude_to_db(xp, ref=ref, top_db=None)
        xp2 = librosa.db_to_amplitude(db, ref=ref)

        assert np.allclose(xp, xp2)

    for ref_p in range(-3, 4):
        yield __test, 10.0**ref_p


def test_db_to_amplitude():

    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    db = librosa.amplitude_to_db(x, top_db=None)
    x2 = librosa.db_to_amplitude(db)

    assert np.allclose(x, x2)


def test_clicks():

    def __test(times, frames, sr, hop_length, click_freq, click_duration, click, length):

        y = librosa.clicks(times=times,
                           frames=frames,
                           sr=sr,
                           hop_length=hop_length,
                           click_freq=click_freq,
                           click_duration=click_duration,
                           click=click,
                           length=length)

        if times is not None:
            nmax = librosa.time_to_samples(times, sr=sr).max()
        else:
            nmax = librosa.frames_to_samples(frames, hop_length=hop_length).max()

        if length is not None:
            assert len(y) == length
        elif click is not None:
            assert len(y) == nmax + len(click)

    test_times = np.linspace(0, 10.0, num=5)

    # Bad cases
    yield raises(librosa.ParameterError)(__test), None, None, 22050, 512, 1000, 0.1, None, None
    yield raises(librosa.ParameterError)(__test), test_times, None, 22050, 512, 1000, 0.1, np.ones((2, 10)), None
    yield raises(librosa.ParameterError)(__test), test_times, None, 22050, 512, 1000, 0.1, None, 0
    yield raises(librosa.ParameterError)(__test), test_times, None, 22050, 512, 0, 0.1, None, None
    yield raises(librosa.ParameterError)(__test), test_times, None, 22050, 512, 1000, 0, None, None

    for sr in [11025, 22050]:
        for hop_length in [512, 1024]:
            test_frames = librosa.time_to_frames(test_times, sr=sr, hop_length=hop_length)

            for click in [None, np.ones(sr // 10)]:

                for length in [None, 5 * sr, 15 * sr]:
                    yield __test, test_times, None, sr, hop_length, 1000, 0.1, click, length
                    yield __test, None, test_frames, sr, hop_length, 1000, 0.1, click, length


def test_fmt_scale():
    # This test constructs a single-cycle cosine wave, applies various axis scalings,
    # and tests that the FMT is preserved

    def __test(scale, n_fmt, over_sample, kind, y_orig, y_res, atol):

        # Make sure our signals preserve energy
        assert np.allclose(np.sum(y_orig**2), np.sum(y_res**2))

        # Scale-transform the original
        f_orig = librosa.fmt(y_orig,
                             t_min=0.5,
                             n_fmt=n_fmt,
                             over_sample=over_sample,
                             kind=kind)

        # Force to the same length
        n_fmt_res = 2 * len(f_orig) - 2

        # Scale-transform the new signal to match
        f_res = librosa.fmt(y_res,
                            t_min=scale * 0.5,
                            n_fmt=n_fmt_res,
                            over_sample=over_sample,
                            kind=kind)

        # Due to sampling alignment, we'll get some phase deviation here
        # The shape of the spectrum should be approximately preserved though.
        assert np.allclose(np.abs(f_orig), np.abs(f_res), atol=atol, rtol=1e-7)

    # Our test signal is a single-cycle sine wave
    def f(x):
        freq = 1
        return np.sin(2 * np.pi * freq * x)

    bounds = [0, 1.0]
    num = 2**8

    x = np.linspace(bounds[0], bounds[1], num=num, endpoint=False)

    y_orig = f(x)

    atol = {'slinear': 1e-4, 'quadratic': 1e-5, 'cubic': 1e-6}

    for scale in [2, 3./2, 5./4, 9./8]:

        # Scale the time axis
        x_res = np.linspace(bounds[0], bounds[1], num=scale * num, endpoint=False)
        y_res = f(x_res)

        # Re-normalize the energy to match that of y_orig
        y_res /= np.sqrt(scale)

        for kind in ['slinear', 'quadratic', 'cubic']:
            for n_fmt in [None, 64, 128, 256, 512]:
                for cur_os in [1, 2, 3]:
                    yield __test, scale, n_fmt, cur_os, kind, y_orig, y_res, atol[kind]

                # Over-sampling with down-scaling gets dicey at the end-points
                yield __test, 1./scale, n_fmt, 1, kind, y_res, y_orig, atol[kind]


def test_fmt_fail():

    @raises(librosa.ParameterError)
    def __test(t_min, n_fmt, over_sample, y):
        librosa.fmt(y, t_min=t_min, n_fmt=n_fmt, over_sample=over_sample)

    srand()
    y = np.random.randn(256)

    # Test for bad t_min
    for t_min in [-1, 0]:
        yield __test, t_min, None, 2, y

    # Test for bad n_fmt
    for n_fmt in [-1, 0, 1, 2]:
        yield __test, 1, n_fmt, 2, y

    # Test for bad over_sample
    for over_sample in [-1, 0, 0.5]:
        yield __test, 1, None, over_sample, y

    # Test for bad input
    y[len(y)//2:] = np.inf
    yield __test, 1, None, 2, y

    # Test for insufficient samples
    yield __test, 1, None, 1, np.ones(2)


def test_fmt_axis():

    srand()
    y = np.random.randn(32, 32)

    f1 = librosa.fmt(y, axis=-1)
    f2 = librosa.fmt(y.T, axis=0).T

    assert np.allclose(f1, f2)


def test_harmonics_1d():

    x = np.arange(16)
    y = np.linspace(-8, 8, num=len(x), endpoint=False)**2

    h = [0.25, 0.5, 1, 2, 4]

    yh = librosa.interp_harmonics(y, x, h)

    eq_(yh.shape[1:], y.shape)
    eq_(yh.shape[0], len(h))
    for i in range(len(h)):
        if h[i] <= 1:
            # Check that subharmonics match
            step = int(1./h[i])
            vals = yh[i, ::step]
            assert np.allclose(vals, y[:len(vals)])
        else:
            # Else check that harmonics match
            step = h[i]
            vals = y[::step]
            assert np.allclose(vals, yh[i, :len(vals)])


def test_harmonics_2d():

    x = np.arange(16)
    y = np.linspace(-8, 8, num=len(x), endpoint=False)**2
    y = np.tile(y, (5, 1)).T
    h = [0.25, 0.5, 1, 2, 4]

    yh = librosa.interp_harmonics(y, x, h, axis=0)

    eq_(yh.shape[1:], y.shape)
    eq_(yh.shape[0], len(h))
    for i in range(len(h)):
        if h[i] <= 1:
            # Check that subharmonics match
            step = int(1./h[i])
            vals = yh[i, ::step]
            assert np.allclose(vals, y[:len(vals)])
        else:
            # Else check that harmonics match
            step = h[i]
            vals = y[::step]
            assert np.allclose(vals, yh[i, :len(vals)])


@raises(librosa.ParameterError)
def test_harmonics_badshape_1d():
    freqs = np.zeros(100)
    obs = np.zeros((5, 10))
    librosa.interp_harmonics(obs, freqs, [1])


@raises(librosa.ParameterError)
def test_harmonics_badshape_2d():
    freqs = np.zeros((5, 5))
    obs = np.zeros((5, 10))
    librosa.interp_harmonics(obs, freqs, [1])


def test_harmonics_2d_varying():

    x = np.arange(16)
    y = np.linspace(-8, 8, num=len(x), endpoint=False)**2
    x = np.tile(x, (5, 1)).T
    y = np.tile(y, (5, 1)).T
    h = [0.25, 0.5, 1, 2, 4]

    yh = librosa.interp_harmonics(y, x, h, axis=0)

    eq_(yh.shape[1:], y.shape)
    eq_(yh.shape[0], len(h))
    for i in range(len(h)):
        if h[i] <= 1:
            # Check that subharmonics match
            step = int(1./h[i])
            vals = yh[i, ::step]
            assert np.allclose(vals, y[:len(vals)])
        else:
            # Else check that harmonics match
            step = h[i]
            vals = y[::step]
            assert np.allclose(vals, yh[i, :len(vals)])
