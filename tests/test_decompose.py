#!/usr/bin/env python
# CREATED: 2013-10-06 22:31:29 by Dawen Liang <dl2771@columbia.edu>
# unit tests for librosa.decompose

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import numpy as np
import librosa


def test_default_decompose():
    
    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    
    (W, H) = librosa.decompose.decompose(X)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)
