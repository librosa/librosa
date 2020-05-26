#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np

import pytest
from test_core import srand

import librosa


# Core viterbi tests
def test_viterbi_example():
    # Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    # States: 0 = healthy, 1 = fever
    p_init = np.asarray([0.6, 0.4])

    # state 0 = hi, state 1 = low
    transition = np.asarray([[0.7, 0.3], [0.4, 0.6]])

    # emission likelihoods
    emit_p = [dict(normal=0.5, cold=0.4, dizzy=0.1), dict(normal=0.1, cold=0.3, dizzy=0.6)]

    obs = ["normal", "cold", "dizzy"]

    prob = np.asarray([np.asarray([ep[o] for o in obs]) for ep in emit_p])

    path, logp = librosa.sequence.viterbi(prob, transition, p_init, return_logp=True)

    # True maximum likelihood state
    assert np.array_equal(path, [0, 0, 1])
    assert np.isclose(logp, np.log(0.01512))

    # And check the second execution path
    path2 = librosa.sequence.viterbi(prob, transition, p_init, return_logp=False)

    assert np.array_equal(path, path2)


def test_viterbi_init():
    # Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    # States: 0 = healthy, 1 = fever
    p_init = np.asarray([0.5, 0.5])

    # state 0 = hi, state 1 = low
    transition = np.asarray([[0.7, 0.3], [0.4, 0.6]])

    # emission likelihoods
    emit_p = [dict(normal=0.5, cold=0.4, dizzy=0.1), dict(normal=0.1, cold=0.3, dizzy=0.6)]

    obs = ["normal", "cold", "dizzy"]

    prob = np.asarray([np.asarray([ep[o] for o in obs]) for ep in emit_p])

    path1, logp1 = librosa.sequence.viterbi(prob, transition, p_init, return_logp=True)

    path2, logp2 = librosa.sequence.viterbi(prob, transition, return_logp=True)

    assert np.array_equal(path1, path2)
    assert logp1 == logp2


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("x", [np.random.random(size=(3, 5))])
@pytest.mark.parametrize(
    "trans",
    [
        np.ones((3, 3), dtype=float),
        np.ones((3, 2), dtype=float),
        np.ones((2, 2), dtype=float),
        np.asarray([[1, 1, -1], [1, 1, -1], [1, 1, -1]], dtype=float),
    ],
    ids=["sum!=1", "not square", "too small", "negative"],
)
def test_viterbi_bad_transition(trans, x):
    librosa.sequence.viterbi(x, trans)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("x", [np.random.random(size=(3, 5))])
@pytest.mark.parametrize("trans", [np.ones((3, 3), dtype=float) / 3.0])
@pytest.mark.parametrize(
    "p_init",
    [np.ones(3, dtype=float), np.ones(4, dtype=float) / 4.0, np.asarray([1, 1, -1], dtype=float)],
    ids=["sum!=1", "wrong size", "negative"],
)
def test_viterbi_bad_init(x, trans, p_init):
    librosa.sequence.viterbi(x, trans, p_init=p_init)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("trans", [np.ones((3, 3), dtype=float) / 3])
@pytest.mark.parametrize(
    "x", [np.random.random(size=(3, 5)) + 2, np.random.random(size=(3, 5)) - 1], ids=["p>1", "p<0"]
)
def test_viterbi_bad_obs(trans, x):
    librosa.sequence.viterbi(x, trans)


# Discriminative viterbi
def test_viterbi_discriminative_example():
    # A pre-baked example with coin tosses

    transition = np.asarray([[0.75, 0.25], [0.25, 0.75]])

    # Joint XY model
    p_joint = np.asarray([[0.25, 0.25], [0.1, 0.4]])

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

    path, logp = librosa.sequence.viterbi_discriminative(
        prob_d, transition, p_state=p_state_marginal, p_init=p_init, return_logp=True
    )

    # Pre-computed optimal path, determined by brute-force search
    assert np.array_equal(path, [1, 1, 1, 1, 1, 1, 0, 0])

    # And check the second code path
    path2 = librosa.sequence.viterbi_discriminative(
        prob_d, transition, p_state=p_state_marginal, p_init=p_init, return_logp=False
    )
    assert np.array_equal(path, path2)


def test_viterbi_discriminative_example_init():
    # A pre-baked example with coin tosses

    transition = np.asarray([[0.75, 0.25], [0.25, 0.75]])

    # Joint XY model
    p_joint = np.asarray([[0.25, 0.25], [0.1, 0.4]])

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

    path, logp = librosa.sequence.viterbi_discriminative(
        prob_d, transition, p_state=p_state_marginal, p_init=p_init, return_logp=True
    )
    path2, logp2 = librosa.sequence.viterbi_discriminative(
        prob_d, transition, p_state=p_state_marginal, return_logp=True
    )
    assert np.array_equal(path, path2)
    assert np.allclose(logp, logp2)


@pytest.fixture(scope="module")
def x_disc():
    srand()
    x = np.random.random(size=(3, 5)) ** 2
    x /= x.sum(axis=0, keepdims=True)
    return x


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "trans",
    [
        np.ones((3, 3), dtype=float),
        np.ones((3, 2), dtype=float) * 0.5,
        np.ones((2, 2), dtype=float) * 0.5,
        np.asarray([[1, 1, -1], [1, 1, -1], [1, 1, -1]], dtype=float),
    ],
    ids=["sum>1", "bad shape", "too small", "negative"],
)
def test_viterbi_discriminative_bad_transition(x_disc, trans):
    librosa.sequence.viterbi_discriminative(x_disc, trans)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("trans", [np.ones((3, 3), dtype=float) / 3])
@pytest.mark.parametrize(
    "p_init",
    [np.ones(3, dtype=float), np.ones(4, dtype=float) / 4.0, np.asarray([1, 1, -1], dtype=float)],
    ids=["sum>1", "too many states", "negative"],
)
def test_viterbi_discriminative_bad_init(p_init, trans, x_disc):
    librosa.sequence.viterbi_discriminative(x_disc, trans, p_init=p_init)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("trans", [np.ones((3, 3), dtype=float) / 3])
@pytest.mark.parametrize(
    "p_state",
    [np.ones(3, dtype=float), np.ones(4, dtype=float) / 4.0, np.asarray([1, 1, -1], dtype=float)],
    ids=["sum>1", "too many states", "negative"],
)
def test_viterbi_discriminative_bad_marginal(x_disc, trans, p_state):
    librosa.sequence.viterbi_discriminative(x_disc, trans, p_state=p_state)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("trans", [np.ones((3, 3), dtype=float) / 3])
@pytest.mark.parametrize(
    "x",
    [
        np.zeros((3, 5), dtype=float),
        np.ones((3, 5), dtype=float),
        np.asarray([[1, 1, -1], [0, 0, 1], [0, 0, 0]], dtype=float),
    ],
    ids=["zeros", "ones", "neg"],
)
def test_viterbi_discriminative_bad_obs(x, trans):
    librosa.sequence.viterbi_discriminative(x, trans)


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
    path, logp = librosa.sequence.viterbi_binary(
        p_binary, transition, p_state=p_init[1:], p_init=p_init[1:], return_logp=True
    )

    # And the full multi-label result
    path_c, logp_c = librosa.sequence.viterbi_binary(
        p_full, transition, p_state=p_init, p_init=p_init, return_logp=True
    )
    path_c2 = librosa.sequence.viterbi_binary(p_full, transition, p_state=p_init, p_init=p_init, return_logp=False)

    # Check that the single and multilabel cases agree
    assert np.allclose(logp, logp_c[1])
    assert np.array_equal(path[0], path_c[1])
    assert np.array_equal(path_c, path_c2)

    # And do an explicit multi-class comparison
    path_d, logp_d = librosa.sequence.viterbi_discriminative(
        p_full, transition, p_state=p_init, p_init=p_init, return_logp=True
    )
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
    path_c, logp_c = librosa.sequence.viterbi_binary(
        p_full, transition, p_state=p_init, p_init=p_init, return_logp=True
    )
    path_c2, logp_c2 = librosa.sequence.viterbi_binary(p_full, transition, p_state=p_init, return_logp=True)

    # Check that the single and multilabel cases agree
    assert np.allclose(logp_c, logp_c2)
    assert np.array_equal(path_c, path_c2)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("x", [np.random.random(size=(3, 5)) ** 2])
@pytest.mark.parametrize(
    "trans",
    [
        np.ones((2, 2), dtype=float),
        np.ones((3, 3), dtype=float) / 3,
        np.ones((3, 5, 5), dtype=float),
        np.asarray([[2, -1], [2, -1]]),
    ],
    ids=["sum>1", "wrong size", "wrong shape", "negative"],
)
def test_viterbi_binary_bad_transition(x, trans):
    librosa.sequence.viterbi_binary(x, trans)


@pytest.mark.parametrize("x", [np.random.random(size=(3, 5)) ** 2])
@pytest.mark.parametrize("trans", [np.ones((2, 2), dtype=float) * 0.5])
@pytest.mark.parametrize(
    "p_init",
    [2 * np.ones(3, dtype=float), np.ones(4, dtype=float), -np.ones(3, dtype=float)],
    ids=["too big", "wrong shape", "negative"],
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_viterbi_binary_bad_init(x, trans, p_init):
    librosa.sequence.viterbi_binary(x, trans, p_init=p_init)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("x", [np.random.random(size=(3, 5)) ** 2])
@pytest.mark.parametrize("trans", [np.ones((2, 2), dtype=float) * 0.5])
@pytest.mark.parametrize(
    "p_state",
    [2 * np.ones(3, dtype=float), np.ones(4, dtype=float), -np.ones(3, dtype=float)],
    ids=["too big", "bad shape", "negative"],
)
def test_viterbi_binary_bad_marginal(p_state, trans, x):
    librosa.sequence.viterbi_binary(x, trans, p_state=p_state)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("trans", [np.ones((2, 2), dtype=float) * 0.5])
@pytest.mark.parametrize(
    "x", [-np.ones((3, 5), dtype=float), 2 * np.ones((3, 5), dtype=float)], ids=["non-positive", "too big"]
)
def test_viterbi_binary_bad_obs(x, trans):
    librosa.sequence.viterbi_binary(x, trans)


# Transition operator constructors
@pytest.mark.parametrize("n", range(1, 4))
def test_trans_uniform(n):
    A = librosa.sequence.transition_uniform(n)
    assert A.shape == (n, n)
    assert np.allclose(A, 1.0 / n)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("n", [0, None])
def test_trans_uniform_badshape(n):
    librosa.sequence.transition_uniform(n)


@pytest.mark.parametrize("n,p", [(2, 0.5), (3, 0.5), (3, [0.8, 0.7, 0.5])])
def test_trans_loop(n, p):
    A = librosa.sequence.transition_loop(n, p)

    # Right shape
    assert A.shape == (n, n)
    # diag is correct
    assert np.allclose(np.diag(A), p)

    # we have well-formed distributions
    assert np.all(A >= 0)
    assert np.allclose(A.sum(axis=1), 1)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "n,p",
    [(1, 0.5), (None, 0.5), (3, 1.5), (3, -0.25), (3, [0.5, 0.2])],
    ids=["missing states", "wrong states", "not probability", "neg prob", "shape mismatch"],
)
def test_trans_loop_fail(n, p):
    librosa.sequence.transition_loop(n, p)


@pytest.mark.parametrize("n,p", [(2, 0.5), (3, 0.5), (3, [0.8, 0.7, 0.5])])
def test_trans_cycle(n, p):
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


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "n,p",
    [(1, 0.5), (None, 0.5), (3, 1.5), (3, -0.25), (3, [0.5, 0.2])],
    ids=["too few states", "wrong n_states", "p>1", "p<0", "shape mismatch"],
)
def test_trans_cycle_fail(n, p):
    librosa.sequence.transition_cycle(n, p)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("n", [1.5, 0])
def test_trans_local_nstates_fail(n):
    librosa.sequence.transition_local(n, 3)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("width", [-1, 0, [2, 3]])
def test_trans_local_width_fail(width):
    librosa.sequence.transition_local(5, width)


def test_trans_local_wrap_const():

    A = librosa.sequence.transition_local(5, 3, window="triangle", wrap=True)

    A_true = np.asarray(
        [
            [0.5, 0.25, 0.0, 0.0, 0.25],
            [0.25, 0.5, 0.25, 0.0, 0.0],
            [0.0, 0.25, 0.5, 0.25, 0.0],
            [0.0, 0.0, 0.25, 0.5, 0.25],
            [0.25, 0.0, 0.0, 0.25, 0.5],
        ]
    )

    assert np.allclose(A, A_true)


def test_trans_local_nowrap_const():

    A = librosa.sequence.transition_local(5, 3, window="triangle", wrap=False)

    A_true = np.asarray(
        [
            [2.0 / 3, 1.0 / 3, 0.0, 0.0, 0.0],
            [0.25, 0.5, 0.25, 0.0, 0.0],
            [0.0, 0.25, 0.5, 0.25, 0.0],
            [0.0, 0.0, 0.25, 0.5, 0.25],
            [0.0, 0.0, 0.0, 1.0 / 3, 2.0 / 3],
        ]
    )

    assert np.allclose(A, A_true)


def test_trans_local_wrap_var():

    A = librosa.sequence.transition_local(5, [2, 1, 3, 3, 2], window="ones", wrap=True)

    A_true = np.asarray(
        [
            [0.5, 0.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0],
            [0.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ]
    )

    assert np.allclose(A, A_true)


def test_trans_local_nowrap_var():

    A = librosa.sequence.transition_local(5, [2, 1, 3, 3, 2], window="ones", wrap=False)

    A_true = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0],
            [0.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ]
    )

    assert np.allclose(A, A_true)


@pytest.mark.parametrize("gap_onset", [1, np.inf])
@pytest.mark.parametrize("gap_extend", [1, np.inf])
@pytest.mark.parametrize("knight", [False, True])
@pytest.mark.parametrize("backtrack", [False, True])
def test_rqa_edge(gap_onset, gap_extend, knight, backtrack):

    rec = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

    out = librosa.sequence.rqa(
        rec, gap_onset=gap_onset, gap_extend=gap_extend, knight_moves=knight, backtrack=backtrack
    )

    if backtrack:
        score, path = out
        __validate_rqa_results(rec, score, path, gap_onset, gap_extend, backtrack, knight)
        assert len(path) == 3
    else:
        # without backtracking, make sure the output is just the score matrix
        assert out.shape == rec.shape


@pytest.mark.parametrize("gap_onset", [1, np.inf])
@pytest.mark.parametrize("gap_extend", [1, np.inf])
@pytest.mark.parametrize("knight", [False, True])
def test_rqa_empty(gap_onset, gap_extend, knight):
    rec = np.zeros((5, 5))

    score, path = librosa.sequence.rqa(
        rec, gap_onset=gap_onset, gap_extend=gap_extend, knight_moves=knight, backtrack=True
    )

    assert score.shape == rec.shape
    assert np.allclose(score, 0)
    assert path.shape == (0, 2)


@pytest.mark.parametrize("gap_onset", [1, np.inf])
@pytest.mark.parametrize("gap_extend", [1, np.inf])
@pytest.mark.parametrize("knight", [False, True])
@pytest.mark.parametrize("backtrack", [False, True])
def test_rqa_interior(gap_onset, gap_extend, knight, backtrack):
    rec = np.asarray([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

    out = librosa.sequence.rqa(
        rec, gap_onset=gap_onset, gap_extend=gap_extend, knight_moves=knight, backtrack=backtrack
    )

    if backtrack:
        score, path = out
        __validate_rqa_results(rec, score, path, gap_onset, gap_extend, backtrack, knight)
        assert len(path) == 2
    else:
        # without backtracking, make sure the output is just the score matrix
        assert out.shape == rec.shape


@pytest.mark.parametrize("gap_onset", [1, np.inf])
@pytest.mark.parametrize("gap_extend", [1, np.inf])
def test_rqa_gaps(gap_onset, gap_extend):
    rec = np.ones((5, 5))
    librosa.sequence.rqa(rec, gap_onset=gap_onset, gap_extend=gap_extend)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_rqa_bad_onset():
    rec = np.ones((5, 5))
    librosa.sequence.rqa(rec, gap_onset=-1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_rqa_bad_extend():
    rec = np.ones((5, 5))
    librosa.sequence.rqa(rec, gap_extend=-1)


def __validate_rqa_results(rec, score, path, gap_onset, gap_extend, backtrack, knight):
    # Test maximal end-point
    assert np.all(score[tuple(path[-1])] >= score)

    # Test non-zero start point
    assert rec[tuple(path[0])] > 0

    # If we can't have gaps, then all values must be nonzero
    if not np.isfinite(gap_onset) and not np.isfinite(gap_extend):
        assert np.all([rec[tuple(i)] > 0 for i in path])

    path_diff = np.diff(path, axis=0)
    if knight:
        for d in path_diff:
            assert np.allclose(d, (1, 1)) or np.allclose(d, (1, 2)) or np.allclose(d, (2, 1))
    else:
        # Without knight moves, only diagonal steps are allowed
        assert np.allclose(path_diff, 1)
