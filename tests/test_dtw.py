#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import librosa
import numpy as np
from scipy.spatial.distance import cdist

import pytest
from test_core import srand


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_1d_input():
    X = np.array([[1], [3], [3], [8], [1]])
    Y = np.array([[2], [0], [0], [8], [7], [2]])
    librosa.sequence.dtw(X=X, Y=Y)


def test_dtw_global():
    # Example taken from:
    # Meinard Mueller, Fundamentals of Music Processing
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])

    gt_D = np.array(
        [
            [1.0, 2.0, 3.0, 10.0, 16.0, 17.0],
            [2.0, 4.0, 5.0, 8.0, 12.0, 13.0],
            [3.0, 5.0, 7.0, 10.0, 12.0, 13.0],
            [9.0, 11.0, 13.0, 7.0, 8.0, 14.0],
            [10, 10.0, 11.0, 14.0, 13.0, 9.0],
        ]
    )

    mut_D, _ = librosa.sequence.dtw(X, Y)
    assert np.array_equal(gt_D, mut_D)

    # Check that it works without backtracking
    mut_D2 = librosa.sequence.dtw(X, Y, backtrack=False)
    assert np.array_equal(mut_D, mut_D2)


def test_dtw_global_constrained():
    # Example taken from:
    # Meinard Mueller, Fundamentals of Music Processing
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])

    # With band_rad = 0.5, the GT distance array is
    gt_D = np.array(
        [
            [1.0, 2.0, 3.0, np.inf, np.inf, np.inf],
            [2.0, 4.0, 5.0, 8.0, np.inf, np.inf],
            [np.inf, 5.0, 7.0, 10.0, 12.0, np.inf],
            [np.inf, np.inf, 13.0, 7.0, 8.0, 14.0],
            [np.inf, np.inf, np.inf, 14.0, 13.0, 9.0],
        ]
    )

    mut_D = librosa.sequence.dtw(
        X, Y, backtrack=False, global_constraints=True, band_rad=0.5
    )
    assert np.array_equal(gt_D, mut_D)


def test_dtw_global_supplied_distance_matrix():
    # Example taken from:
    # Meinard Mueller, Fundamentals of Music Processing
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])

    # Precompute distance matrix.
    C = cdist(X.T, Y.T, metric="euclidean")

    gt_D = np.array(
        [
            [1.0, 2.0, 3.0, 10.0, 16.0, 17.0],
            [2.0, 4.0, 5.0, 8.0, 12.0, 13.0],
            [3.0, 5.0, 7.0, 10.0, 12.0, 13.0],
            [9.0, 11.0, 13.0, 7.0, 8.0, 14.0],
            [10, 10.0, 11.0, 14.0, 13.0, 9.0],
        ]
    )

    # Supply precomputed distance matrix and specify an invalid distance
    # metric to verify that it isn't used.
    mut_D, _ = librosa.sequence.dtw(C=C, metric="invalid")

    assert np.array_equal(gt_D, mut_D)


def test_dtw_gobal_boundary():
    # Verify that boundary condition is fulfilled for subseq=False.
    # See https://github.com/librosa/librosa/pull/920
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([1, 1, 1, 2, 4, 5, 6, 5, 5])
    gt_wp = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [4, 7], [4, 8]]
    )

    D, wp = librosa.sequence.dtw(X, Y, subseq=False)
    wp = wp[::-1]
    assert np.array_equal(gt_wp, wp)


def test_dtw_subseq_boundary():
    # Verify that boundary condition doesn't have to be fulfilled for
    # subseq=True.
    # See https://github.com/librosa/librosa/pull/920
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([1, 1, 1, 2, 4, 5, 6, 5, 5])
    gt_wp = np.array([[0, 2], [1, 3], [2, 3], [3, 4], [4, 5]])

    D, wp = librosa.sequence.dtw(X, Y, subseq=True)
    wp = wp[::-1]
    assert np.array_equal(gt_wp, wp)


def test_dtw_subseq_steps():
    # Verify that the same warping path is computed for backtracking
    # within the dtw function and manually outside by the user
    # See https://github.com/librosa/librosa/pull/1166
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([1, 1, 1, 2, 4, 5, 6, 5, 5])
    gt_wp = np.array([[0, 2], [1, 3], [2, 3], [3, 4], [4, 5]])

    D1, wp1 = librosa.sequence.dtw(X, Y, subseq=True, backtrack=True)
    wp1 = wp1[::-1]

    D2, steps = librosa.sequence.dtw(
        X, Y, subseq=True, backtrack=False, return_steps=True
    )
    start_idx = np.argmin(D2[-1, :])
    wp2 = librosa.sequence.dtw_backtracking(steps, subseq=True, start=start_idx)
    wp2 = wp2[::-1]

    assert np.array_equal(D1, D2)
    assert np.array_equal(gt_wp, wp1)
    assert np.array_equal(wp1, wp2)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_backtracking_incompatible_args_01():
    # See https://github.com/librosa/librosa/pull/1166
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([1, 1, 1, 2, 4, 5, 6, 5, 5])

    D, steps = librosa.sequence.dtw(
        X, Y, subseq=True, backtrack=False, return_steps=True
    )
    start_idx = np.argmin(D[-1, :])
    librosa.sequence.dtw_backtracking(steps, subseq=False, start=start_idx)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_incompatible_args_01():
    librosa.sequence.dtw(C=1, X=1, Y=1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_incompatible_args_02():
    librosa.sequence.dtw(C=None, X=None, Y=None)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_incompatible_sigma_add():
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])
    librosa.sequence.dtw(X=X, Y=Y, weights_add=np.arange(10))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_incompatible_sigma_mul():
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])
    librosa.sequence.dtw(X=X, Y=Y, weights_mul=np.arange(10))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_incompatible_sigma_diag():
    X = np.array([[1, 3, 3, 8, 1, 2]])
    Y = np.array([[2, 0, 0, 8, 7]])
    librosa.sequence.dtw(X=X, Y=Y, step_sizes_sigma=np.ones((1, 2), dtype=int))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_incompatible_sigma_diag_precomp():
    C = np.ones((5, 3))
    librosa.sequence.dtw(C=C, step_sizes_sigma=[[1, 1]])


def test_dtw_global_diagonal():
    # query is a linear ramp
    X = np.linspace(0.1, 1, 10)
    Y = X

    gt_wp = list(zip(list(range(10)), list(range(10))))[::-1]

    mut_D, mut_wp = librosa.sequence.dtw(
        X,
        Y,
        subseq=True,
        metric="cosine",
        step_sizes_sigma=np.array([[1, 1]]),
        weights_mul=np.array([1]),
    )

    assert np.array_equal(np.asarray(gt_wp), np.asarray(mut_wp))


def test_dtw_subseq():
    srand()

    # query is a linear ramp
    X = np.linspace(0, 1, 100)

    # database is query surrounded by noise
    noise_len = 200
    noise = np.random.rand(noise_len)
    Y = np.concatenate((noise, noise, X, noise))

    _, mut_wp = librosa.sequence.dtw(X, Y, subseq=True)

    # estimated sequence has to match original sequence
    # note the +1 due to python indexing
    mut_X = Y[mut_wp[-1][1] : mut_wp[0][1] + 1]
    assert np.array_equal(X, mut_X)


def test_dtw_subseq_supplied_distance_matrix():
    X = np.array([[0], [1], [2]])
    Y = np.array([[1], [2], [3], [4]])
    C = cdist(X, Y)

    costs0, path0 = librosa.sequence.dtw(X.T, Y.T, subseq=True)
    costs1, path1 = librosa.sequence.dtw(C=C, subseq=True)

    assert np.array_equal(costs0, costs1)
    assert np.array_equal(path0, path1)


def test_dtw_subseq_sym():
    Y = np.array([10.0, 10.0, 0.0, 1.0, 2.0, 3.0, 10.0, 10.0])
    X = np.arange(4)

    gt_wp_XY = np.array([[3, 5], [2, 4], [1, 3], [0, 2]])
    gt_wp_YX = np.array([[5, 3], [4, 2], [3, 1], [2, 0]])

    _, mut_wp_XY = librosa.sequence.dtw(X, Y, subseq=True)
    _, mut_wp_YX = librosa.sequence.dtw(Y, X, subseq=True)

    assert np.array_equal(gt_wp_XY, mut_wp_XY)
    assert np.array_equal(gt_wp_YX, mut_wp_YX)


def test_dtw_global_constraint_destructive():

    # Issue #1029, dtw with global constraints inserts nans
    # into the cost matrix.  This is fine when the cost is computed
    # locally, but if passed by reference, it's destructive.
    # This test checks that the cost matrix is unmodified.
    C1 = np.ones((20, 20))
    C2 = np.copy(C1)
    librosa.sequence.dtw(C=C1, global_constraints=True)
    assert np.array_equal(C1, C2)


def test_dtw_global_inf():
    # What should we do if backtracking fails in full sequence mode?
    # This will happen if the inner loop of bt aborts prematurely
    # by walking off the edge of the cost array instead of
    # path-following to (0, 0)

    # Construct a cost matrix where full alignment is impossible
    C = np.zeros((4, 4), dtype=np.float)
    C[-1, -1] = np.inf
    with pytest.raises(librosa.ParameterError):
        librosa.sequence.dtw(C=C, subseq=False)


def test_dtw_subseq_inf():
    # Construct a cost matrix where partial alignment is impossible
    C = np.zeros((4, 4), dtype=np.float)
    C[-1, :] = np.inf

    with pytest.raises(librosa.ParameterError):
        librosa.sequence.dtw(C=C, subseq=True)


def test_dtw_subseq_pass():
    # Construct a cost matrix where partial alignment is possible
    C = np.zeros((4, 4), dtype=np.float)
    C[-1, 2:] = np.inf
    librosa.sequence.dtw(C=C, subseq=True)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_dtw_nan_fail():
    C = np.ones((10, 10))
    C[4, 6] = np.nan
    librosa.sequence.dtw(C=C)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "steps", [np.array([[1, -1]]), np.array([[-1, 1]]), np.array([[-1, -1]])]
)
def test_dtw_negative_steps(steps):
    C = np.ones((10, 10))
    librosa.sequence.dtw(C=C, step_sizes_sigma=steps)
