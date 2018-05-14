#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for output functions'''

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import six
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

        kwargs = dict()
        if six.PY3:
            kwargs['newline'] = '\n'

        # Load it back
        with open(tfname, 'r', **kwargs) as fdesc:
            for i, line in enumerate(fdesc):
                row = line.strip().split(sep)
                assert np.allclose(float(row[0]), times[i], atol=1e-3, rtol=1e-3), (row, times)

                if annotations is not None:
                    assert row[1] == annotations[i]

        # Remove the file
        os.unlink(tfname)

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
        kwargs = dict()
        if six.PY3:
            kwargs['newline'] = '\n'

        with open(tfname, 'r', **kwargs) as fdesc:
            for i, line in enumerate(fdesc):
                row = line.strip().split(sep)
                assert np.allclose([float(row[0]), float(row[1])], times[i], atol=1e-3, rtol=1e-3), (row, times)

                if annotations is not None:
                    assert row[2] == annotations[i]

        # Remove the file
        os.unlink(tfname)

    __test_fail = raises(librosa.ParameterError)(__test)

    srand()
    # Make times and durations strictly non-negative
    times = np.random.randn(20, 2)**2
    times = np.cumsum(times, axis=1)

    for annotations in [None, ['abcde'[q]
                               for q in np.random.randint(0, 5, size=len(times))], list('abcde')]:
        for sep in [',', '\t', ' ']:
            if annotations is not None and len(annotations) != len(times):
                yield __test_fail, times, annotations, sep
            else:
                yield __test, times, annotations, sep
