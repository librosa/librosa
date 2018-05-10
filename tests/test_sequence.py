#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import librosa
import numpy as np

from nose.tools import raises
from test_core import srand

import warnings
warnings.resetwarnings()
warnings.simplefilter('always')


def test_viterbi_example():
    # Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    # States: 0 = healthy, 1 = fever
    p_init = np.asarray([0.6, 0.4])

    # state 0 = hi, state 1 = low
    transition = np.asarray([[0.7, 0.3],
                             [0.4, 0.6]])

    # emission likelihoods
    emit_p = [dict(normal=0.5, cold=0.4, dizzy=0.1),
              dict(normal=0.1, cold=0.3, dizzy=0.6)]

    obs = ['normal', 'cold', 'dizzy']

    prob = np.asarray([np.asarray([ep[o] for o in obs])
                       for ep in emit_p])

    path, logp = librosa.sequence.viterbi(prob, transition, p_init,
                                        return_logp=True)

    # True maximum likelihood state
    assert np.array_equal(path, [0, 0, 1])
    assert np.isclose(logp, np.log(0.01512))
