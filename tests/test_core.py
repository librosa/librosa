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

import matplotlib
matplotlib.use('Agg')
from nose.tools import nottest, eq_, raises


# -- utilities --#
def files(pattern):
    test_files = glob.glob(pattern)
    test_files.sort()
    return test_files


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


def test_resample():

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
        target_length = len(y) * sr_out // sr_in
        assert np.abs(len(y2) - target_length) <= 1

    for infile in ['data/test1_44100.wav',
                   'data/test1_22050.wav',
                   'data/test2_8000.wav']:
        y, sr_in = librosa.load(infile, sr=None, duration=5)

        for sr_out in [8000, 22050]:
            for res_type in ['sinc_fastest', 'sinc_best', 'scipy']:
                for fix in [False, True]:
                    yield (__test, y, sr_in, sr_out, res_type, fix)


@nottest
def __deprecated_test_resample():

    def __test(infile, scipy_resample):
        DATA = load(infile)

        # load the wav file
        (y_in, sr_in) = librosa.load(DATA['wavfile'][0], sr=None, mono=True)

        # Resample it to the target rate
        y_out = librosa.resample(y_in, DATA['sr_in'], DATA['sr_out'],
                                 scipy_resample=scipy_resample)

        # Are we the same length?
        if len(y_out) == len(DATA['y_out']):
            # Is the data close?
            assert np.allclose(y_out, DATA['y_out'])
        elif len(y_out) == len(DATA['y_out']) - 1:
            assert (np.allclose(y_out, DATA['y_out'][:-1, 0]) or
                    np.allclose(y_out, DATA['y_out'][1:, 0]))
        elif len(y_out) == len(DATA['y_out']) + 1:
            assert (np.allclose(y_out[1:], DATA['y_out']) or
                    np.allclose(y_out[:-2], DATA['y_out']))
        else:
            assert False
        pass

    for infile in files('data/core-resample-*.mat'):
        for scipy_resample in [False, True]:
            yield (__test, infile, scipy_resample)
    pass


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
            window = None
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

    def __test(ref_power, clip):

        F, D = librosa.ifgram(y, sr=sr, ref_power=ref_power, clip=clip)

        if clip:
            assert np.all(0 <= F) and np.all(F <= 0.5 * sr)

        assert np.all(np.isfinite(F))

    for ref_power in [-10, 0.0, 1e-6, np.max]:
        for clip in [False, True]:
            if six.callable(ref_power) or ref_power >= 0.0:
                tf = __test
            else:
                tf = raises(librosa.ParameterError)(__test)

            yield tf, ref_power, clip


def test_magphase():

    (y, sr) = librosa.load('data/test1_22050.wav')

    D = librosa.stft(y)

    S, P = librosa.magphase(D)

    assert np.allclose(S * P, D)


def test_istft():
    def __test(infile):
        DATA = load(infile)

        if DATA['hann_w'][0, 0] == 0:
            window = np.ones
            win_length = 2 * (DATA['D'].shape[0] - 1)
        else:
            window = None
            win_length = DATA['hann_w'][0, 0]

        Dinv = librosa.istft(DATA['D'],
                             hop_length=DATA['hop_length'][0, 0].astype(int),
                             win_length=win_length,
                             window=window,
                             center=False)

        assert np.allclose(Dinv, DATA['Dinv'])

    for infile in files('data/core-istft-*.mat'):
        yield (__test, infile)


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
                for hop_length in [n_fft / 8, n_fft / 4, n_fft / 2]:
                    for center in [False, True]:
                        yield (__test_spec, test_file, sr,
                               duration, n_fft, hop_length, center)


def test_autocorrelate():

    def __test(y, max_size):

        ac = librosa.autocorrelate(y, max_size=max_size)

        if max_size is None or max_size > len(y):
            eq_(len(ac), len(y))

        else:
            eq_(len(ac), max_size)

    y = np.random.randn(256)

    for max_size in [None, len(y), 2 * len(y)]:
        yield __test, y, max_size


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

        print('target_hz={:.3f}'.format(target_hz))
        print('tuning={:.3f}, estimated={:.3f}'.format(tuning, tuning_est))
        print('resolution={:.2e}'.format(resolution))

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
    def __test(x, ref_power, amin, top_db):
    
        y = librosa.logamplitude(x,
                                 ref_power=ref_power,
                                 amin=amin,
                                 top_db=top_db)

        assert np.isrealobj(y)
        eq_(y.shape, x.shape)

        if top_db is not None:
            assert y.min() >= y.max()-top_db

    for n in [1, 2, 10]:
        x = np.linspace(0, 2e5, num=n)
        phase = np.exp(1.j * x)

        for ref_power in [1.0, np.max]:
            for amin in [-1, 0, 1e-10, 1e3]:
                for top_db in [None, -10, 0, 40, 80]:
                    tf = __test
                    if amin <= 0 or (top_db is not None and top_db < 0):
                        tf = raises(librosa.ParameterError)(__test)
                    yield tf, x, ref_power, amin, top_db
                    yield tf, x * phase, ref_power, amin, top_db
