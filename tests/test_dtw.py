#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import librosa
import numpy as np
from scipy.spatial.distance import cdist

from nose.tools import raises
from test_core import srand

import warnings
warnings.resetwarnings()
warnings.simplefilter('always')


@raises(librosa.ParameterError)
def test_1d_input():
    X = np.array([[1], [3], [3], [8], [1]])
    Y = np.array([[2], [0], [0], [8], [7], [2]])
    librosa.sequence.dtw(X=X, Y=Y)


def test_dtw_global():
    # Example taken from:
    # Meinard Mueller, Fundamentals of Music Processing
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])

    gt_D = np.array([[1., 2., 3., 10., 16., 17.],
                     [2., 4., 5., 8., 12., 13.],
                     [3., 5., 7., 10., 12., 13.],
                     [9., 11., 13., 7., 8., 14.],
                     [10, 10., 11., 14., 13., 9.]])

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
    gt_D = np.array([[1., 2., 3., np.inf, np.inf, np.inf],
                     [2., 4., 5., 8., np.inf, np.inf],
                     [np.inf, 5., 7., 10., 12., np.inf],
                     [np.inf, np.inf, 13., 7., 8., 14.],
                     [np.inf, np.inf, np.inf, 14., 13., 9.]])

    mut_D = librosa.sequence.dtw(X, Y, backtrack=False, global_constraints=True, band_rad=0.5)
    assert np.array_equal(gt_D, mut_D)


def test_dtw_global_supplied_distance_matrix():
    # Example taken from:
    # Meinard Mueller, Fundamentals of Music Processing
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])

    # Precompute distance matrix.
    C = cdist(X.T, Y.T, metric='euclidean')

    gt_D = np.array([[1., 2., 3., 10., 16., 17.],
                     [2., 4., 5., 8., 12., 13.],
                     [3., 5., 7., 10., 12., 13.],
                     [9., 11., 13., 7., 8., 14.],
                     [10, 10., 11., 14., 13., 9.]])

    # Supply precomputed distance matrix and specify an invalid distance
    # metric to verify that it isn't used.
    mut_D, _ = librosa.sequence.dtw(C=C, metric='invalid')

    assert np.array_equal(gt_D, mut_D)


@raises(librosa.ParameterError)
def test_dtw_incompatible_args_01():
    librosa.sequence.dtw(C=1, X=1, Y=1)


@raises(librosa.ParameterError)
def test_dtw_incompatible_args_02():
    librosa.sequence.dtw(C=None, X=None, Y=None)



@raises(librosa.ParameterError)
def test_dtw_incompatible_sigma_add():
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])
    librosa.sequence.dtw(X=X, Y=Y, weights_add=np.arange(10))


@raises(librosa.ParameterError)
def test_dtw_incompatible_sigma_mul():
    X = np.array([[1, 3, 3, 8, 1]])
    Y = np.array([[2, 0, 0, 8, 7, 2]])
    librosa.sequence.dtw(X=X, Y=Y, weights_mul=np.arange(10))


@raises(librosa.ParameterError)
def test_dtw_incompatible_sigma_diag():
    X = np.array([[1, 3, 3, 8, 1, 2]])
    Y = np.array([[2, 0, 0, 8, 7]])
    librosa.sequence.dtw(X=X, Y=Y, step_sizes_sigma=np.ones((1, 2), dtype=int))


def test_dtw_global_diagonal():
    # query is a linear ramp
    X = np.linspace(0.1, 1, 10)
    Y = X

    gt_wp = list(zip(list(range(10)), list(range(10))))[::-1]

    mut_D, mut_wp = librosa.sequence.dtw(X, Y, subseq=True, metric='cosine',
                                step_sizes_sigma=np.array([[1, 1]]),
                                weights_mul=np.array([1, ]))

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
    mut_X = Y[mut_wp[-1][1]:mut_wp[0][1] + 1]
    assert np.array_equal(X, mut_X)


def test_dtw_subseq_sym():
    Y = np.array([10., 10., 0., 1., 2., 3., 10., 10.])
    X = np.arange(4)

    gt_wp_XY = np.array([[3, 5], [2, 4], [1, 3], [0, 2]])
    gt_wp_YX = np.array([[5, 3], [4, 2], [3, 1], [2, 0]])

    _, mut_wp_XY = librosa.sequence.dtw(X, Y, subseq=True)
    _, mut_wp_YX = librosa.sequence.dtw(Y, X, subseq=True)

    assert np.array_equal(gt_wp_XY, mut_wp_XY)
    assert np.array_equal(gt_wp_YX, mut_wp_YX)
