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
from nose.tools import raises


def test_write_wav():

    def __test(mono):

        y, sr = librosa.load('data/test1_22050.wav', sr=None, mono=mono)

        _, tfname = tempfile.mkstemp()

        librosa.output.write_wav(tfname, y, sr, norm=False)

        y_2, sr2 = librosa.load(tfname, sr=None, mono=mono)

        os.unlink(tfname)

        librosa.util.valid_audio(y_2, mono=mono)

        assert np.allclose(sr2, sr)

        assert np.allclose(y, y_2, rtol=1e-3, atol=1e-4)

    for mono in [False, True]:
        yield __test, mono


def test_times_csv():

    def __test(times, annotations, sep):

        _, tfname = tempfile.mkstemp()

        # Dump to disk
        librosa.output.times_csv(tfname, times, annotations=annotations,
                                 delimiter=sep)

        # Load it back
        with open(tfname, 'r') as fdesc:
            for i, line in enumerate(fdesc):
                if annotations is None:
                    t_in = line.strip()
                else:
                    t_in, ann_in = line.strip().split(sep, 2)
                t_in = float(t_in)

                assert np.allclose(times[i], t_in, atol=1e-3, rtol=1e-3)
                if annotations is not None:
                    assert str(annotations[i]) == ann_in

    __test_fail = raises(ValueError)(__test)

    for times in [[], np.linspace(0, 10, 20)]:
        for annotations in [None, ['abcde'[q] for q in np.random.randint(0, 5,
                                   size=len(times))], list('abcde')]:
                for sep in [',', '\t', ' ']:

                    if annotations is not None and len(annotations) != len(times):
                        yield __test_fail, times, annotations, sep
                    else:
                        yield __test, times, annotations, sep
