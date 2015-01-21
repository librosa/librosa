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
