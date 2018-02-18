#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for output functions'''

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import numpy as np
import tempfile
from nose.tools import raises, eq_

from test_core import srand
import warnings
warnings.resetwarnings()
warnings.simplefilter('always')


def test_write_wav():

    def __test(mono, norm):

        fpath = os.path.join('data', 'test1_22050.wav')
        y, sr = librosa.load(fpath, sr=None, mono=mono)

        _, tfname = tempfile.mkstemp()
        os.close(_)

        librosa.output.write_wav(tfname, y, sr, norm=norm)

        y_2, sr2 = librosa.load(tfname, sr=None, mono=mono)

        os.unlink(tfname)

        librosa.util.valid_audio(y_2, mono=mono)

        assert np.allclose(sr2, sr)

        if norm:
            assert np.allclose(librosa.util.normalize(y, axis=-1),
                               librosa.util.normalize(y_2, axis=-1),
                               rtol=1e-3, atol=1e-4)
        else:
            assert np.allclose(y, y_2, rtol=1e-3, atol=1e-4)

    for mono in [False, True]:
        for norm in [False, True]:
            yield __test, mono, norm


def test_times_csv():

    def __test(times, annotations, sep):

        _, tfname = tempfile.mkstemp()
        os.close(_)

        # Dump to disk
        librosa.output.times_csv(tfname, times, annotations=annotations,
                                 delimiter=sep)

        # Load it back
        recons = np.loadtxt(tfname, delimiter=sep, dtype=object)

        # Remove the file
        os.unlink(tfname)

        if recons.ndim > 1:
            times_r = recons[:, 0].astype(np.float)
        else:
            times_r = recons.astype(np.float)

        assert np.allclose(times_r, times, atol=1e-3, rtol=1e-3)

        if len(times) and annotations is not None:
            eq_(list(recons[:, 1]), annotations)

    __test_fail = raises(librosa.ParameterError)(__test)

    srand()
    for times in [[], np.linspace(0, 10, 20)]:
        for annotations in [None, ['abcde'[q] for q in np.random.randint(0, 5,
                                   size=len(times))], list('abcde')]:
            for sep in [',', '\t', ' ']:
                if annotations is not None and len(annotations) != len(times):
                    yield __test_fail, times, annotations, sep
                else:
                    yield __test, times, annotations, sep


def test_annotation():

    def __test(times, annotations, sep):

        _, tfname = tempfile.mkstemp()
        os.close(_)

        # Dump to disk
        librosa.output.annotation(tfname, times, annotations=annotations,
                                  delimiter=sep)

        # Load it back
        recons = np.loadtxt(tfname, delimiter=sep, dtype=object)

        # Remove the file
        os.unlink(tfname)

        if recons.shape[1] > 2:
            times_r = recons[:, :2].astype(np.float)
        else:
            times_r = recons.astype(np.float)

        assert np.allclose(times_r, times, atol=1e-3, rtol=1e-3)

        if len(times) and annotations is not None:
            eq_(list(recons[:, 2]), annotations)

    __test_fail = raises(librosa.ParameterError)(__test)

    srand()
    times = np.random.randn(20, 2)

    for annotations in [None, ['abcde'[q] for q in np.random.randint(0, 5,
                               size=len(times))], list('abcde')]:
        for sep in [',', '\t', ' ']:
            if annotations is not None and len(annotations) != len(times):
                yield __test_fail, times, annotations, sep
            else:
                yield __test, times, annotations, sep
