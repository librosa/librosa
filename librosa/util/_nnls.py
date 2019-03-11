#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Non-negative least squares'''

# The scipy library provides an nnls solver, but it does
# not generalize efficiently to matrix-valued problems.
# We therefore provide an alternate solver here.
#
# The NNLS solver we implement below is based on the ADMM
# algorithm for convex optimization.  Instead of directly
# solving:
#   min_{X>=0} |AX - B|^2
# we solve a related problem:
#   min_{X, Y>=0} |AX - B|^2 + I[X = Y]
# where the last term is the indicator that X and Y are
# identical.
#
# The X,Y problem can be solved using augmented lagrangian
# methods, which results in a loop of the following updates:
#
# 1. X <- min_X |AX - B|^2 + r |X - (Y - W)|^2
# 2. Y <- min_{Y>=0} |X - (Y - W)|^2 = max(0, X + W)
# 3. W <- W + X - Y
#
# where W is the matrix of lagrangian dual variables for
# the equality constraint, and r is a step size parameter.
# Steps 2 and 3 are trivial, and step 1 can be viewed as
# a generalized Tikhonov regularization problem.
#
# The optimal solution to step 1 is given by
#   X <- Y - W + (A'A + rI)^-1 (A'B - A'A(Y - W))
#
# This is efficient to compute if A is tall (so A'A is small),
# but for wide A, we can use the Woodbury matrix identity to
# compute it more efficiently. The implementation given below
# dynamically selects the more efficient method at run-time.
#
# A couple of additional optimizations are provided to improve
# stability:
#   1. The solver is warm-started by X = leastsq(A, B) and Y = X_+
#   2. The step-size r (rho in the code) is tuned by the effective
#      condition number of A.  This is inspired by Nishihara et al.,
#      which assumes strong convexity of the objective (which we
#      don't generally have).
#   3. If the target B is a vector (not a matrix), we fall back on the
#      scipy solver.


import numpy as np
import scipy.optimize
from numba import jit

__all__ = ['nnls']


@jit(nopython=True)
def _nnls(A, B, rho, eps_abs=1e-6, eps_rel=1e-4, max_iter=500):
    '''Compute a non-negative least-squares solution to
    min_{X>=0} \|AX - B\|_F^2
    '''
    # X* = Z + r^-1 * (I - A'(r I - AA')^-1 A) (A'B - A'A Z)
    #
    # Say L = r^-1 * (I - A'(rI - AA')^-1 A)
    #
    # then X* <= Z + LA'B - LA'AZ

    # Can we infer rho from the spectrum of A?
    #   nishihara'15 say to use
    #       rho* = sqrt(lambda_min(A'A) * lambda_max(A'A))
    #
    #   but this assumes strong convexity on f, i.e., A is full-rank.
    #   that's not the case for us, but we do have strong convexity over A's column space
    #   so instead, we'll initialze using the (nonzero) singular values of A:
    #       rho* = sigma_min(A) * sigma_max(A)

    n, m = A.shape
    _, N = B.shape

    # identiy matrices with dtype matching A
    Im = np.eye(m, m, 0, A.dtype)

    if n <= m:
        # This will be a small matrix if A is wide
        # Say L = r^-1 * (I - A'(rI - AA')^-1 A)
        In = np.eye(n, n, 0, A.dtype)
        L = Im - np.dot(A.T, np.linalg.solve(rho * In + np.dot(A, A.T), A))
        L /= rho

        # L is m by m
        # A' is m by n  (m >> n)
        # B is n by N

        LAt = np.dot(L, A.T)
    else:
        LAt = np.linalg.solve(np.dot(A.T, A) + rho * Im, A.T)

    LAtB = np.dot(LAt, B)
    LAtApI = np.dot(LAt, A) - Im

    # Initialize X and Y with the (thresholded) least squares solution
    # This puts our initial iterate X into the column space of A
    # so that we have strong (local) convexity
    X = np.linalg.lstsq(A, B)[0]
    Y = np.zeros(X.shape, dtype=A.dtype)
    np.maximum(X, 0.0, Y)
    W = X - Y

    residual = W.copy()

    for _ in range(max_iter):
        # Primal update 1: generalized tikhonov solve
        X[:] = LAtB - np.dot(LAtApI, Y - W)

        # Primal update 2: projection onto feasible set
        np.maximum(X + W, 0, Y)

        # Dual update
        residual[:] = X - Y
        W += residual

        # Convergence criteria:
        #    dual residual:   res_dual = rho * (W - W_prev)
        #    primal residual: res_primal = X - Y
        #                          but W - W_prev = X - Y
        #    so res_dual = rho * rho_primal
        # boyd et al suggest for stopping criteria:
        #    |res_dual|_F <= sqrt(n) * eps_absolute + eps_relative * rho * norm(W)
        #    |res_primal|_F <= sqrt(n) * eps_absolute + eps_relative * max(norm(X), norm(Y))
        #    |res_primal| <= sqrt(n) * eps_absolute / rho + eps_relative * norm(W)
        #    
        # so convergence is when
        #    |X - Y| <= min(t1, t2)
        #    where t1 = sqrt(n) * eps_absolute + eps_relative * max(norm(X), norm(Y))
        #          t2 = sqrt(n) * eps_absolute / rho + eps_relative * norm(W)

        # Convergence thresholds for primal and dual variables
        t_primal = np.sqrt(X.size) * eps_abs + eps_rel * np.sqrt(max(np.sum(X**2), np.sum(Y**2)))
        t_dual = np.sqrt(X.size) * eps_abs / rho + eps_rel * np.sqrt(np.sum(W**2))

        if np.sum(residual**2) <= min(t_primal, t_dual)**2:
            break

    # Y is the feasible point that we've found
    return Y


def nnls(A, B, eps_abs=1e-6, eps_rel=1e-4, max_iter=500):
    '''Non-negative least squares.

    Given two matrices A and B, find a non-negative matrix X
    that minimizes the sum squared error:

        err(X) = sum_i,j ((AX)[i,j] - B[i, j])^2

    Parameters
    ----------
    A : np.ndarray [shape=(m, n)]
        The basis matrix

    B : np.ndarray [shape=(m, N)]
        The target matrix.

    eps_abs : number > 0
        The absolute precision threshold

    eps_rel : number > 0
        The relative precision threshold

    max_iter : int > 0
        The maximum number of iterations for the solver


    Returns
    -------
    X : np.ndarray [shape=(n, N), non-negative]
        A minimizing solution to |AX - B|^2

    See Also
    --------
    scipy.optimize.nnls

    Examples
    --------
    Approximate a magnitude spectrum from its mel spectrogram

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=10)
    >>> S = np.abs(librosa.stft(y, n_fft=2048))
    >>> M = librosa.feature.melspectrogram(S=S, sr=sr, power=1)
    >>> mel_basis = librosa.filters.mel(sr, n_fft=2048, n_mels=M.shape[0])
    >>> S_recover = librosa.util.nnls(mel_basis, M)

    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2,1,1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Original spectrogram')
    >>> plt.subplot(2,1,2)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S_recover, ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Reconstructed spectrogram')
    >>> plt.tight_layout()
    '''

    # If B is a single vector, punt up to the scipy method
    if B.ndim == 1:
        return scipy.optimize.nnls(A, B)[0]

    if B.size > A.size:
        A = A.astype(B.dtype)
    elif B.size < A.size:
        B = B.astype(A.dtype)

    # Otherwise, initialize our step size
    svds = np.linalg.svd(A, compute_uv=False)

    # Explicitly cast to float so that numba isn't confused
    rho = np.asanyarray(0.5 * svds.max() * svds.min(), dtype=A.dtype)

    return _nnls(A, B,
                 rho=rho,
                 eps_abs=eps_abs,
                 eps_rel=eps_rel,
                 max_iter=max_iter)
