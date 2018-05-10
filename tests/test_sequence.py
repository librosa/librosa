#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import warnings

import numpy as np

from nose.tools import raises
from test_core import srand

import librosa

warnings.resetwarnings()
warnings.simplefilter('always')


# Core viterbi tests
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

    # And check the second execution path
    path2 = librosa.sequence.viterbi(prob, transition, p_init,
                                     return_logp=False)

    assert np.array_equal(path, path2)


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


# Discriminative viterbi
def test_viterbi_discriminative_example():
    # A pre-baked example with coin tosses

    transition = np.asarray([[0.75, 0.25], [0.25, 0.75]])

    # Joint XY model
    p_joint = np.asarray([[0.25, 0.25],
                          [0.1 , 0.4 ]])

    # marginals
    p_obs_marginal = p_joint.sum(axis=0)
    p_state_marginal = p_joint.sum(axis=1)

    p_init = p_state_marginal

    # Make the Y|X distribution
    p_state_given_obs = (p_joint / p_obs_marginal).T

    # Let's make a test observation sequence
    seq = np.asarray([1, 1, 0, 1, 1, 1, 0, 0])

    # Then our conditional probability table can be constructed directly as
    prob_d = np.asarray([p_state_given_obs[i] for i in seq]).T

    path, logp = librosa.sequence.viterbi_discriminative(prob_d,
                                            transition,
                                            p_state=p_state_marginal,
                                            p_init=p_init,
                                            return_logp=True)

    # Pre-computed optimal path, determined by brute-force search
    assert np.array_equal(path, [1, 1, 1, 1, 1, 1, 0, 0])

    # And check the second code path
    path2 = librosa.sequence.viterbi_discriminative(prob_d,
                                       transition,
                                       p_state=p_state_marginal,
                                       p_init=p_init,
                                       return_logp=False)
    assert np.array_equal(path, path2)

def test_viterbi_discriminative_example_init():
    # A pre-baked example with coin tosses

    transition = np.asarray([[0.75, 0.25], [0.25, 0.75]])

    # Joint XY model
    p_joint = np.asarray([[0.25, 0.25],
                          [0.1 , 0.4 ]])

    # marginals
    p_obs_marginal = p_joint.sum(axis=0)
    p_state_marginal = p_joint.sum(axis=1)

    p_init = np.asarray([0.5, 0.5])

    # Make the Y|X distribution
    p_state_given_obs = (p_joint / p_obs_marginal).T

    # Let's make a test observation sequence
    seq = np.asarray([1, 1, 0, 1, 1, 1, 0, 0])

    # Then our conditional probability table can be constructed directly as
    prob_d = np.asarray([p_state_given_obs[i] for i in seq]).T

    path, logp = librosa.sequence.viterbi_discriminative(prob_d,
                                                         transition,
                                                         p_state=p_state_marginal,
                                                         p_init=p_init,
                                                         return_logp=True)
    path2, logp2 = librosa.sequence.viterbi_discriminative(prob_d,
                                                           transition,
                                                           p_state=p_state_marginal,
                                                           return_logp=True)
    assert np.array_equal(path, path2)
    assert np.allclose(logp, logp2)


def test_viterbi_discriminative_bad_transition():
    @raises(librosa.ParameterError)
    def __bad_trans(trans, x):
        librosa.sequence.viterbi_discriminative(x, trans)

    x = np.random.random(size=(3, 5))**2
    x /= np.sum(x, axis=0, keepdims=True)

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


def test_viterbi_discriminative_bad_init():
    @raises(librosa.ParameterError)
    def __bad_init(init, trans, x):
        librosa.sequence.viterbi_discriminative(x, trans, p_init=init)

    x = np.random.random(size=(3, 5))**2
    x /= x.sum(axis=0, keepdims=True)

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


def test_viterbi_discriminative_bad_marginal():
    @raises(librosa.ParameterError)
    def __bad_init(state, trans, x):
        librosa.sequence.viterbi_discriminative(x, trans, p_state=state)

    x = np.random.random(size=(3, 5))**2
    x /= x.sum(axis=0, keepdims=True)

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


def test_viterbi_discriminative_bad_obs():
    @raises(librosa.ParameterError)
    def __bad_obs(x, trans):
        librosa.sequence.viterbi_discriminative(x, trans)

    srand()

    trans = np.ones((3, 3), dtype=float) / 3.

    # x does not sum to 1
    x = np.zeros((3, 5), dtype=float)
    yield __bad_obs, x, trans

    x = np.ones((3, 5), dtype=float)
    yield __bad_obs, x, trans

    # x has negative values < 0
    x[1, 1] = -0.5
    yield __bad_obs, x, trans


# Multi-label viterbi
def test_viterbi_binary_example():

    # 0 stays 0,
    # 1 is uninformative
    transition = np.asarray([[0.9, 0.1], [0.5, 0.5]])

    # Initial state distribution
    p_init = np.asarray([0.25, 0.75])

    p_binary = np.asarray([0.25, 0.5, 0.75, 0.1, 0.1, 0.8, 0.9])

    p_full = np.vstack((1 - p_binary, p_binary))

    # Compute the viterbi_binary result for one class
    path, logp = librosa.sequence.viterbi_binary(p_binary, transition, p_state=p_init[1:], p_init=p_init[1:], return_logp=True)

    # And the full multi-label result
    path_c, logp_c = librosa.sequence.viterbi_binary(p_full, transition, p_state=p_init, p_init=p_init, return_logp=True)
    path_c2 = librosa.sequence.viterbi_binary(p_full, transition, p_state=p_init, p_init=p_init, return_logp=False)

    # Check that the single and multilabel cases agree
    assert np.allclose(logp, logp_c[1])
    assert np.array_equal(path[0], path_c[1])
    assert np.array_equal(path_c, path_c2)

    # And do an explicit multi-class comparison
    path_d, logp_d = librosa.sequence.viterbi_discriminative(p_full, transition, p_state=p_init, p_init=p_init, return_logp=True)
    assert np.allclose(logp[0], logp_d)
    assert np.array_equal(path[0], path_d)


def test_viterbi_binary_example_init():

    # 0 stays 0,
    # 1 is uninformative
    transition = np.asarray([[0.9, 0.1], [0.5, 0.5]])

    # Initial state distribution
    p_init = np.asarray([0.5, 0.5])

    p_binary = np.asarray([0.25, 0.5, 0.75, 0.1, 0.1, 0.8, 0.9])

    p_full = np.vstack((1 - p_binary, p_binary))

    # And the full multi-label result
    path_c, logp_c = librosa.sequence.viterbi_binary(p_full, transition, p_state=p_init, p_init=p_init, return_logp=True)
    path_c2, logp_c2 = librosa.sequence.viterbi_binary(p_full, transition, p_state=p_init, return_logp=True)

    # Check that the single and multilabel cases agree
    assert np.allclose(logp_c, logp_c2)
    assert np.array_equal(path_c, path_c2)


def test_viterbi_binary_bad_transition():
    @raises(librosa.ParameterError)
    def __bad_trans(trans, x):
        librosa.sequence.viterbi_binary(x, trans)

    x = np.random.random(size=(3, 5))**2

    # transitions do not sum to 1
    trans = np.ones((2, 2), dtype=float)
    yield __bad_trans, trans, x

    # bad shape
    trans = np.ones((3, 3), dtype=float)
    yield __bad_trans, trans, x
    trans = np.ones((3, 5, 5), dtype=float)
    yield __bad_trans, trans, x

    # sums to 1, but negative values
    trans = 2 * np.ones((2, 2), dtype=float)
    trans[:, 1] = -1
    assert np.allclose(np.sum(trans, axis=-1), 1)
    yield __bad_trans, trans, x


def test_viterbi_binary_bad_init():
    @raises(librosa.ParameterError)
    def __bad_init(init, trans, x):
        librosa.sequence.viterbi_binary(x, trans, p_init=init)

    x = np.random.random(size=(3, 5))**2

    trans = np.ones((2, 2), dtype=float) / 2.

    # p_init is too big
    p_init = 2 * np.ones(3, dtype=float)
    yield __bad_init, p_init, trans, x

    # bad shape
    p_init = np.ones(4, dtype=float)
    yield __bad_init, p_init, trans, x

    # negative values
    p_init = -np.ones(3, dtype=float)
    yield __bad_init, p_init, trans, x


def test_viterbi_binary_bad_marginal():
    @raises(librosa.ParameterError)
    def __bad_state(state, trans, x):
        librosa.sequence.viterbi_binary(x, trans, p_state=state)

    x = np.random.random(size=(3, 5))**2

    trans = np.ones((2, 2), dtype=float) / 2.

    # p_init is too big
    p_state = 2 * np.ones(3, dtype=float)
    yield __bad_state, p_state, trans, x

    # bad shape
    p_state = np.ones(4, dtype=float)
    yield __bad_state, p_state, trans, x

    # negative values
    p_state = -np.ones(3, dtype=float)
    yield __bad_state, p_state, trans, x


def test_viterbi_binary_bad_obs():
    @raises(librosa.ParameterError)
    def __bad_obs(x, trans):
        librosa.sequence.viterbi_binary(x, trans)

    srand()

    trans = np.ones((2, 2), dtype=float) / 2.

    # x is not positive
    x = -np.ones((3, 5), dtype=float)
    yield __bad_obs, x, trans

    # x is too big
    x = 2 * np.ones((3, 5), dtype=float)
    yield __bad_obs, x, trans


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


def test_trans_local_nstates_fail():

    @raises(librosa.ParameterError)
    def __test(n):
        librosa.sequence.transition_local(n, 3)

    yield __test, 1.5
    yield __test, 0


def test_trans_local_width_fail():

    @raises(librosa.ParameterError)
    def __test(width):
        librosa.sequence.transition_local(5, width)

    yield __test, -1
    yield __test, 0
    yield __test, [2, 3]

def test_trans_local_wrap_const():

    A = librosa.sequence.transition_local(5, 3, window='triangle', wrap=True)

    A_true = np.asarray([[0.5 , 0.25, 0.  , 0.  , 0.25],
                         [0.25, 0.5 , 0.25, 0.  , 0.  ],
                         [0.  , 0.25, 0.5 , 0.25, 0.  ],
                         [0.  , 0.  , 0.25, 0.5 , 0.25],
                         [0.25, 0.  , 0.  , 0.25, 0.5 ]])

    assert np.allclose(A, A_true)


def test_trans_local_nowrap_const():

    A = librosa.sequence.transition_local(5, 3, window='triangle', wrap=False)

    A_true = np.asarray([[2./3, 1./3, 0.  , 0.  , 0.],
                         [0.25, 0.5 , 0.25, 0.  , 0.  ],
                         [0.  , 0.25, 0.5 , 0.25, 0.  ],
                         [0.  , 0.  , 0.25, 0.5 , 0.25],
                         [0.  , 0.  , 0.  , 1./3, 2./3 ]])

    assert np.allclose(A, A_true)

def test_trans_local_wrap_var():

    A = librosa.sequence.transition_local(5, [2, 1, 3, 3, 2],
                                          window='ones',
                                          wrap=True)

    A_true = np.asarray([[0.5  , 0.   , 0.   , 0.   , 0.5  ],
                         [0.   , 1.   , 0.   , 0.   , 0.   ],
                         [0.   , 1./3 , 1./3 , 1./3 , 0.   ],
                         [0.   , 0.   , 1./3 , 1./3 , 1./3 ],
                         [0.   , 0.   , 0.   , 0.5  , 0.5  ]])

    assert np.allclose(A, A_true)

def test_trans_local_nowrap_var():

    A = librosa.sequence.transition_local(5, [2, 1, 3, 3, 2],
                                          window='ones',
                                          wrap=False)

    A_true = np.asarray([[1.   , 0.   , 0.   , 0.   , 0.   ],
                         [0.   , 1.   , 0.   , 0.   , 0.   ],
                         [0.   , 1./3 , 1./3 , 1./3 , 0.   ],
                         [0.   , 0.   , 1./3 , 1./3 , 1./3 ],
                         [0.   , 0.   , 0.   , 0.5  , 0.5  ]])

    assert np.allclose(A, A_true)

