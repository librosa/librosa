#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence Alignment with Dynamic Time Warping."""

import numpy as np
import numba
from scipy.spatial.distance import cdist


def dtw(X, Y,
        dist='euclidean',
        step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]),
        weights_add=np.array([0, 0, 0]),
        weights_mul=np.array([1, 1, 1]),
        subseq=False):

    max_0 = step_sizes_sigma[:, 0].max()
    max_1 = step_sizes_sigma[:, 1].max()

    # calculate pair-wise distances
    C = cdist(X.T, Y.T, dist)

    # initialize whole matrix with infinity values
    D = np.ones(C.shape + np.array([max_0, max_1])) * np.inf

    # set starting point to C[0, 0]
    D[max_0, max_1] = C[0, 0]

    if subseq:
        D[max_0, max_1:] = C[0, :]

    # calculate accumulated cost matrix
    # will result in
    D = calc_accu_cost(C, D,
                       step_sizes_sigma, weights_mul,
                       max_0, max_1)

    # delete infinity rows and columns
    D = D[max_0:, max_1:]

    return D


@numba.jit(nopython=True)
def calc_accu_cost(C, D, step_sizes_sigma, weights_mul, max_0, max_1):
    for cur_n in range(max_0, D.shape[0]):
        for cur_m in range(max_1, D.shape[1]):
            # loop over all step sizes
            for cur_step_idx in range(step_sizes_sigma.shape[0]):
                cur_D = D[cur_n-step_sizes_sigma[cur_step_idx, 0], cur_m-step_sizes_sigma[cur_step_idx, 1]]
                cur_C = weights_mul[cur_step_idx] * C[cur_n-max_0, cur_m-max_1]
                cur_cost = cur_D + cur_C

                # check if cur_cost is smaller than the one stored in D
                D[cur_n, cur_m] = min(cur_cost, D[cur_n, cur_m])
    return D
