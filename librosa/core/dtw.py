#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence Alignment with Dynamic Time Warping."""

import numpy as np
import numba
from scipy.spatial.distance import cdist

__all__ = ['dtw']


def dtw(X, Y,
        dist='euclidean',
        step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]),
        weights_add=np.array([0, 0, 0]),
        weights_mul=np.array([1, 1, 1]),
        subseq=False):
    '''Dynamic time warping (DTW).

    This function performs a DTW and path backtracking on two sequences.


    Parameters
    ----------
    X : np.ndarray [shape=(K, N)]
        audio feature matrix (e.g., chroma features)

    Y : np.ndarray [shape=(K, M)]
        audio feature matrix (e.g., chroma features)

    dist : str
        Identifier for the cost-function as documented
        in scipy.spatial.cdist()

    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.

    weights_add : np.ndarray [shape=[n, ]]
        Additive weights to penalize certain step sizes.

    weights_mul : np.ndarray [shape=[n, ]]
        Multiplicative weights to penalize certain step sizes.

    subseq : binary
        Enable subsequence DTW, e.g., for retrieval tasks.

    Returns
    -------
    D : np.ndarray [shape=(N,M)]
        accumulated cost matrix.
        D[N,M] is the total alignment cost.
        When doing subsequence DTW, D[N,:] indicates a matching function.

    wp : list [shape=(N,)]
        Warping path with index pairs.
        Each list entry contains an index pair
        (n,m) as a tuple

    See Also
    --------

    '''
    max_0 = step_sizes_sigma[:, 0].max()
    max_1 = step_sizes_sigma[:, 1].max()

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    # calculate pair-wise distances
    C = cdist(X.T, Y.T, dist)

    # initialize whole matrix with infinity values
    D = np.ones(C.shape + np.array([max_0, max_1])) * np.inf

    # set starting point to C[0, 0]
    D[max_0, max_1] = C[0, 0]

    if subseq:
        D[max_0, max_1:] = C[0, :]

    D_steps = np.empty(D.shape, dtype=np.int)

    # calculate accumulated cost matrix
    D, D_steps = calc_accu_cost(C, D, D_steps,
                                step_sizes_sigma,
                                weights_mul, weights_add,
                                max_0, max_1)

    # delete infinity rows and columns
    D = D[max_0:, max_1:]
    D_steps = D_steps[max_0:, max_1:]

    if subseq is False:
        # perform warping path backtracking
        wp = backtracking(D_steps, step_sizes_sigma)
    elif subseq is True:
        # search for global minimum in last row of D-matrix
        wp_end_idx = np.argmin(D[-1, :]) + 1
        # import matplotlib.pyplot as plt
        # plt.plot(D[-1, :])
        # plt.show()
        wp = backtracking(D_steps[:, :wp_end_idx], step_sizes_sigma)

    return D, wp


@numba.jit(nopython=True)
def calc_accu_cost(C, D, D_steps, step_sizes_sigma, weights_mul, weights_add, max_0, max_1):

    for cur_n in range(max_0, D.shape[0]):
        for cur_m in range(max_1, D.shape[1]):
            # loop over all step sizes
            for cur_step_idx in range(step_sizes_sigma.shape[0]):
                cur_D = D[cur_n-step_sizes_sigma[cur_step_idx, 0], cur_m-step_sizes_sigma[cur_step_idx, 1]]
                cur_C = weights_mul[cur_step_idx] * C[cur_n-max_0, cur_m-max_1]
                cur_C += weights_add[cur_step_idx]
                cur_cost = cur_D + cur_C

                # check if cur_cost is smaller than the one stored in D
                if cur_cost < D[cur_n, cur_m]:
                    D[cur_n, cur_m] = cur_cost

                    # save step-index
                    D_steps[cur_n, cur_m] = cur_step_idx
                # D[cur_n, cur_m] = min(cur_cost, D[cur_n, cur_m])
    return D, D_steps


@numba.jit(nopython=True)
def backtracking(D_steps, step_sizes_sigma):
    '''Backtrack optimal warping path.

    Uses the saved step sizes from the cost accumulation
    step to backtrack the index pairs for an optimal
    warping path.


    Parameters
    ----------
    D_steps : np.ndarray [shape=(N, M)]
        Saved indices of the used steps used in the calculation of D.

    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.

    Returns
    -------
    wp : list [shape=(N,)]
        Warping path with index pairs.
        Each list entry contains an index pair
        (n,m) as a tuple

    See Also
    --------

    '''
    wp = []
    # Set starting point D(N,M) and append it to the path
    cur_idx = (D_steps.shape[0]-1, D_steps.shape[1]-1)
    wp.append(cur_idx)

    while True:
        cur_n = cur_idx[0]
        cur_m = cur_idx[1]

        cur_step_idx = D_steps[cur_n, cur_m]

        # save tuple with minimal acc. cost in path
        cur_idx = (cur_n-step_sizes_sigma[cur_step_idx][0],
                   cur_m-step_sizes_sigma[cur_step_idx][1])
        wp.append(cur_idx)

        # set stop criteria
        # Setting it to (0, 0) does not work for the subsequence dtw
        if cur_idx[0] == 0:
            break

    return wp
