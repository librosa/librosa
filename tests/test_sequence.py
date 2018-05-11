#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np

import warnings
warnings.resetwarnings()
warnings.simplefilter('always')

from nose.tools import raises
from test_core import srand

import librosa


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

def test_viterbi_init():
    # Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    # States: 0 = healthy, 1 = fever
    p_init = np.asarray([0.5, 0.5])

    # state 0 = hi, state 1 = low
    transition = np.asarray([[0.7, 0.3],
                             [0.4, 0.6]])

    # emission likelihoods
    emit_p = [dict(normal=0.5, cold=0.4, dizzy=0.1),
              dict(normal=0.1, cold=0.3, dizzy=0.6)]

    obs = ['normal', 'cold', 'dizzy']

    prob = np.asarray([np.asarray([ep[o] for o in obs])
                       for ep in emit_p])

    path1, logp1 = librosa.sequence.viterbi(prob, transition, p_init,
                                            return_logp=True)

    path2, logp2 = librosa.sequence.viterbi(prob, transition,
                                            return_logp=True)

    assert np.array_equal(path1, path2)
    assert logp1 == logp2


def test_viterbi_bad_transition():
    @raises(librosa.ParameterError)
    def __bad_trans(trans, x):
        librosa.sequence.viterbi(x, trans)

    x = np.random.random(size=(3, 5))

    # transitions do not sum to 1
    trans = np.ones((3, 3), dtype=float)
    yield __bad_trans, trans, x

    # bad shape
    trans = np.ones((3, 2), dtype=float)
    yield __bad_trans, trans, x
    trans = np.ones((2, 2), dtype=float)
    yield __bad_trans, trans, x

    # sums to 1, but negative values
    trans = np.ones((3, 3), dtype=float)
    trans[:, 1] = -1
    assert np.allclose(np.sum(trans, axis=1), 1)
    yield __bad_trans, trans, x

def test_viterbi_bad_init():
    @raises(librosa.ParameterError)
    def __bad_init(init, trans, x):
        librosa.sequence.viterbi(x, trans, p_init=init)

    x = np.random.random(size=(3, 5))
    trans = np.ones((3, 3), dtype=float) / 3.

    # p_init does not sum to 1
    p_init = np.ones(3, dtype=float)
    yield __bad_init, p_init, trans, x

    # bad shape
    p_init = np.ones(4, dtype=float)
    yield __bad_init, p_init, trans, x

    # sums to 1, but negative values
    p_init = np.ones(3, dtype=float)
    p_init[1] = -1
    assert np.allclose(np.sum(p_init), 1)
    yield __bad_init, p_init, trans, x

def test_viterbi_bad_obs():
    @raises(librosa.ParameterError)
    def __bad_obs(trans, x):
        librosa.sequence.viterbi(x, trans)

    srand()

    x = np.random.random(size=(3, 5))
    trans = np.ones((3, 3), dtype=float) / 3.

    # x has values > 1
    x[1, 1] = 2
    yield __bad_obs, trans, x

    # x has values < 0
    x[1, 1] = -0.5
    yield __bad_obs, trans, x


# Transition operator constructors
def test_trans_uniform():
    def __trans(n):
        A = librosa.sequence.transition_uniform(n)
        assert A.shape == (n, n)
        assert np.allclose(A, 1./n)

    for n in range(1, 4):
        yield __trans, n

    yield raises(librosa.ParameterError)(__trans), 0
    yield raises(librosa.ParameterError)(__trans), None


def test_trans_loop():
    def __trans(n, p):
        A = librosa.sequence.transition_loop(n, p)

        # Right shape
        assert A.shape == (n, n)
        # diag is correct
        assert np.allclose(np.diag(A), p)

        # we have well-formed distributions
        assert np.all(A >= 0)
        assert np.allclose(A.sum(axis=1), 1)

    # Test with constant self-loops
    for n in range(2, 4):
        yield __trans, n, 0.5

    # Test with variable self-loops
    yield __trans, 3, [0.8, 0.7, 0.5]

    # Failure if we don't have enough states
    yield raises(librosa.ParameterError)(__trans), 1, 0.5

    # Failure if n_states is wrong
    yield raises(librosa.ParameterError)(__trans), None, 0.5

    # Failure if p is not a probability
    yield raises(librosa.ParameterError)(__trans), 3, 1.5
    yield raises(librosa.ParameterError)(__trans), 3, -0.25

    # Failure if there's a shape mismatch
    yield raises(librosa.ParameterError)(__trans), 3, [0.5, 0.2]


def test_trans_cycle():
    def __trans(n, p):
        A = librosa.sequence.transition_cycle(n, p)

        # Right shape
        assert A.shape == (n, n)
        # diag is correct
        assert np.allclose(np.diag(A), p)

        for i in range(n):
            assert A[i, np.mod(i + 1, n)] == 1 - A[i, i]

        # we have well-formed distributions
        assert np.all(A >= 0)
        assert np.allclose(A.sum(axis=1), 1)

    # Test with constant self-loops
    for n in range(2, 4):
        yield __trans, n, 0.5

    # Test with variable self-loops
    yield __trans, 3, [0.8, 0.7, 0.5]

    # Failure if we don't have enough states
    yield raises(librosa.ParameterError)(__trans), 1, 0.5

    # Failure if n_states is wrong
    yield raises(librosa.ParameterError)(__trans), None, 0.5

    # Failure if p is not a probability
    yield raises(librosa.ParameterError)(__trans), 3, 1.5
    yield raises(librosa.ParameterError)(__trans), 3, -0.25

    # Failure if there's a shape mismatch
    yield raises(librosa.ParameterError)(__trans), 3, [0.5, 0.2]

