#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Sequential modeling
===================

Dynamic time warping
--------------------
.. autosummary::
    :toctree: generated/

    dtw

Viterbi decoding
----------------
.. autosummary::
    :toctree: generated/

    viterbi
    viterbi_discriminative
    viterbi_binary

Transition matrices
-------------------
.. autosummary::
    :toctree: generated/

    transition_uniform
    transition_loop
    transition_cycle
    transition_local
'''

import numpy as np
from scipy.spatial.distance import cdist
import six
from numba import jit
from .util import pad_center, fill_off_diagonal
from .util.exceptions import ParameterError
from .filters import get_window

__all__ = ['dtw',
           'viterbi',
           'viterbi_discriminative',
           'viterbi_binary',
           'transition_uniform',
           'transition_loop',
           'transition_cycle',
           'transition_local']


def dtw(X=None, Y=None, C=None, metric='euclidean', step_sizes_sigma=None,
        weights_add=None, weights_mul=None, subseq=False, backtrack=True,
        global_constraints=False, band_rad=0.25):
    '''Dynamic time warping (DTW).

    This function performs a DTW and path backtracking on two sequences.
    We follow the nomenclature and algorithmic approach as described in [1]_.

    .. [1] Meinard Mueller
           Fundamentals of Music Processing — Audio, Analysis, Algorithms, Applications
           Springer Verlag, ISBN: 978-3-319-21944-8, 2015.

    Parameters
    ----------
    X : np.ndarray [shape=(K, N)]
        audio feature matrix (e.g., chroma features)

    Y : np.ndarray [shape=(K, M)]
        audio feature matrix (e.g., chroma features)

    C : np.ndarray [shape=(N, M)]
        Precomputed distance matrix. If supplied, X and Y must not be supplied and
        ``metric`` will be ignored.

    metric : str
        Identifier for the cost-function as documented
        in `scipy.spatial.cdist()`

    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.

    weights_add : np.ndarray [shape=[n, ]]
        Additive weights to penalize certain step sizes.

    weights_mul : np.ndarray [shape=[n, ]]
        Multiplicative weights to penalize certain step sizes.

    subseq : binary
        Enable subsequence DTW, e.g., for retrieval tasks.

    backtrack : binary
        Enable backtracking in accumulated cost matrix.

    global_constraints : binary
        Applies global constraints to the cost matrix ``C`` (Sakoe-Chiba band).

    band_rad : float
        The Sakoe-Chiba band radius (1/2 of the width) will be
        ``int(radius*min(C.shape))``.

    Returns
    -------
    D : np.ndarray [shape=(N,M)]
        accumulated cost matrix.
        D[N,M] is the total alignment cost.
        When doing subsequence DTW, D[N,:] indicates a matching function.

    wp : np.ndarray [shape=(N,2)]
        Warping path with index pairs.
        Each row of the array contains an index pair n,m).
        Only returned when ``backtrack`` is True.

    Raises
    ------
    ParameterError
        If you are doing diagonal matching and Y is shorter than X or if an incompatible
        combination of X, Y, and C are supplied.
        If your input dimensions are incompatible.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), offset=10, duration=15)
    >>> X = librosa.feature.chroma_cens(y=y, sr=sr)
    >>> noise = np.random.rand(X.shape[0], 200)
    >>> Y = np.concatenate((noise, noise, X, noise), axis=1)
    >>> D, wp = librosa.sequence.dtw(X, Y, subseq=True)
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(D, x_axis='frames', y_axis='frames')
    >>> plt.title('Database excerpt')
    >>> plt.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    >>> plt.legend()
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(D[-1, :] / wp.shape[0])
    >>> plt.xlim([0, Y.shape[1]])
    >>> plt.ylim([0, 2])
    >>> plt.title('Matching cost function')
    >>> plt.tight_layout()
    '''
    # Default Parameters
    if step_sizes_sigma is None:
        step_sizes_sigma = np.array([[1, 1], [0, 1], [1, 0]])
    if weights_add is None:
        weights_add = np.zeros(len(step_sizes_sigma))
    if weights_mul is None:
        weights_mul = np.ones(len(step_sizes_sigma))

    if len(step_sizes_sigma) != len(weights_add):
        raise ParameterError('len(weights_add) must be equal to len(step_sizes_sigma)')
    if len(step_sizes_sigma) != len(weights_mul):
        raise ParameterError('len(weights_mul) must be equal to len(step_sizes_sigma)')

    if C is None and (X is None or Y is None):
        raise ParameterError('If C is not supplied, both X and Y must be supplied')
    if C is not None and (X is not None or Y is not None):
        raise ParameterError('If C is supplied, both X and Y must not be supplied')

    # calculate pair-wise distances, unless already supplied.
    if C is None:
        # take care of dimensions
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        try:
            C = cdist(X.T, Y.T, metric=metric)
        except ValueError as e:
            msg = ('scipy.spatial.distance.cdist returned an error.\n'
                   'Please provide your input in the form X.shape=(K, N) and Y.shape=(K, M).\n'
                   '1-dimensional sequences should be reshaped to X.shape=(1, N) and Y.shape=(1, M).')
            six.reraise(ParameterError, ParameterError(msg))

        # for subsequence matching:
        # if N > M, Y can be a subsequence of X
        if subseq and (X.shape[1] > Y.shape[1]):
            C = C.T

    C = np.atleast_2d(C)

    # if diagonal matching, Y has to be longer than X
    # (X simply cannot be contained in Y)
    if np.array_equal(step_sizes_sigma, np.array([[1, 1]])) and (C.shape[0] > C.shape[1]):
        raise ParameterError('For diagonal matching: Y.shape[1] >= X.shape[1] '
                             '(C.shape[1] >= C.shape[0])')

    max_0 = step_sizes_sigma[:, 0].max()
    max_1 = step_sizes_sigma[:, 1].max()

    if global_constraints:
        # Apply global constraints to the cost matrix
        fill_off_diagonal(C, band_rad, value=np.inf)

    # initialize whole matrix with infinity values
    D = np.ones(C.shape + np.array([max_0, max_1])) * np.inf

    # set starting point to C[0, 0]
    D[max_0, max_1] = C[0, 0]

    if subseq:
        D[max_0, max_1:] = C[0, :]

    # initialize step matrix with -1
    # will be filled in calc_accu_cost() with indices from step_sizes_sigma
    D_steps = -1 * np.ones(D.shape, dtype=np.int)

    # calculate accumulated cost matrix
    D, D_steps = __dtw_calc_accu_cost(C, D, D_steps,
                                      step_sizes_sigma,
                                      weights_mul, weights_add,
                                      max_0, max_1)

    # delete infinity rows and columns
    D = D[max_0:, max_1:]
    D_steps = D_steps[max_0:, max_1:]

    if backtrack:
        if subseq:
            # search for global minimum in last row of D-matrix
            wp_end_idx = np.argmin(D[-1, :]) + 1
            wp = __dtw_backtracking(D_steps[:, :wp_end_idx], step_sizes_sigma)
        else:
            # perform warping path backtracking
            wp = __dtw_backtracking(D_steps, step_sizes_sigma)

        wp = np.asarray(wp, dtype=int)

        # since we transposed in the beginning, we have to adjust the index pairs back
        if subseq and (X.shape[1] > Y.shape[1]):
            wp = np.fliplr(wp)

        return D, wp
    else:
        return D


@jit(nopython=True)
def __dtw_calc_accu_cost(C, D, D_steps, step_sizes_sigma,
                         weights_mul, weights_add, max_0, max_1):  # pragma: no cover
    '''Calculate the accumulated cost matrix D.

    Use dynamic programming to calculate the accumulated costs.

    Parameters
    ----------
    C : np.ndarray [shape=(N, M)]
        pre-computed cost matrix

    D : np.ndarray [shape=(N, M)]
        accumulated cost matrix

    D_steps : np.ndarray [shape=(N, M)]
        steps which were used for calculating D

    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.

    weights_add : np.ndarray [shape=[n, ]]
        Additive weights to penalize certain step sizes.

    weights_mul : np.ndarray [shape=[n, ]]
        Multiplicative weights to penalize certain step sizes.

    max_0 : int
        maximum number of steps in step_sizes_sigma in dim 0.

    max_1 : int
        maximum number of steps in step_sizes_sigma in dim 1.

    Returns
    -------
    D : np.ndarray [shape=(N,M)]
        accumulated cost matrix.
        D[N,M] is the total alignment cost.
        When doing subsequence DTW, D[N,:] indicates a matching function.

    D_steps : np.ndarray [shape=(N,M)]
        steps which were used for calculating D.

    See Also
    --------
    dtw
    '''
    for cur_n in range(max_0, D.shape[0]):
        for cur_m in range(max_1, D.shape[1]):
            # accumulate costs
            for cur_step_idx, cur_w_add, cur_w_mul in zip(range(step_sizes_sigma.shape[0]),
                                                          weights_add, weights_mul):
                cur_D = D[cur_n - step_sizes_sigma[cur_step_idx, 0],
                          cur_m - step_sizes_sigma[cur_step_idx, 1]]
                cur_C = cur_w_mul * C[cur_n - max_0, cur_m - max_1]
                cur_C += cur_w_add
                cur_cost = cur_D + cur_C

                # check if cur_cost is smaller than the one stored in D
                if cur_cost < D[cur_n, cur_m]:
                    D[cur_n, cur_m] = cur_cost

                    # save step-index
                    D_steps[cur_n, cur_m] = cur_step_idx

    return D, D_steps


@jit(nopython=True)
def __dtw_backtracking(D_steps, step_sizes_sigma):  # pragma: no cover
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
    dtw
    '''
    wp = []
    # Set starting point D(N,M) and append it to the path
    cur_idx = (D_steps.shape[0] - 1, D_steps.shape[1] - 1)
    wp.append((cur_idx[0], cur_idx[1]))

    # Loop backwards.
    # Stop criteria:
    # Setting it to (0, 0) does not work for the subsequence dtw,
    # so we only ask to reach the first row of the matrix.
    while cur_idx[0] > 0:
        cur_step_idx = D_steps[(cur_idx[0], cur_idx[1])]

        # save tuple with minimal acc. cost in path
        cur_idx = (cur_idx[0] - step_sizes_sigma[cur_step_idx][0],
                   cur_idx[1] - step_sizes_sigma[cur_step_idx][1])

        # append to warping path
        wp.append((cur_idx[0], cur_idx[1]))

    return wp


@jit(nopython=True)
def _viterbi(log_prob, log_trans, log_p_init, state, value, ptr):  # pragma: no cover
    '''Core Viterbi algorithm.

    This is intended for internal use only.

    Parameters
    ----------
    log_prob : np.ndarray [shape=(T, m)]
        `log_prob[t, s]` is the conditional log-likelihood
        log P[X = X(t) | State(t) = s]

    log_trans : np.ndarray [shape=(m, m)]
        The log transition matrix
        `log_trans[i, j]` = log P[State(t+1) = j | State(t) = i]

    log_p_init : np.ndarray [shape=(m,)]
        log of the initial state distribution

    state : np.ndarray [shape=(T,), dtype=int]
        Pre-allocated state index array

    value : np.ndarray [shape=(T, m)] float
        Pre-allocated value array

    ptr : np.ndarray [shape=(T, m), dtype=int]
        Pre-allocated pointer array

    Returns
    -------
    None
        All computations are performed in-place on `state, value, ptr`.
    '''
    n_steps, n_states = log_prob.shape

    # factor in initial state distribution
    value[0] = log_prob[0] + log_p_init

    for t in range(1, n_steps):
        # Want V[t, j] <- p[t, j] * max_k V[t-1, k] * A[k, j]
        #    assume at time t-1 we were in state k
        #    transition k -> j

        # Broadcast over rows:
        #    Tout[k, j] = V[t-1, k] * A[k, j]
        #    then take the max over columns
        # We'll do this in log-space for stability

        trans_out = value[t - 1] + log_trans.T

        # Unroll the max/argmax loop to enable numba support
        for j in range(n_states):
            ptr[t, j] = np.argmax(trans_out[j])
            # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]

    # Now roll backward

    # Get the last state
    state[-1] = np.argmax(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t+1, state[t+1]]
    # Done.


def viterbi(prob, transition, p_init=None, return_logp=False):
    '''Viterbi decoding from observation likelihoods.

    Given a sequence of observation likelihoods `prob[s, t]`,
    indicating the conditional likelihood of seeing the observation
    at time `t` from state `s`, and a transition matrix
    `transition[i, j]` which encodes the conditional probability of
    moving from state `i` to state `j`, the Viterbi algorithm [1]_ computes
    the most likely sequence of states from the observations.

    .. [1] Viterbi, Andrew. "Error bounds for convolutional codes and an
        asymptotically optimum decoding algorithm."
        IEEE transactions on Information Theory 13.2 (1967): 260-269.

    Parameters
    ----------
    prob : np.ndarray [shape=(n_states, n_steps), non-negative]
        `prob[s, t]` is the probability of observation at time `t`
        being generated by state `s`.

    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        `transition[i, j]` is the probability of a transition from i->j.
        Each row must sum to 1.

    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, a uniform distribution is assumed.

    return_logp : bool
        If `True`, return the log-likelihood of the state sequence.

    Returns
    -------
    Either `states` or `(states, logp)`:

    states : np.ndarray [shape=(n_steps,)]
        The most likely state sequence.

    logp : scalar [float]
        If `return_logp=True`, the log probability of `states` given
        the observations.

    See Also
    --------
    viterbi_discriminative : Viterbi decoding from state likelihoods


    Examples
    --------
    Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    In this example, we have two states ``healthy`` and ``fever``, with
    initial probabilities 60% and 40%.

    We have three observation possibilities: ``normal``, ``cold``, and
    ``dizzy``, whose probabilities given each state are:

    ``healthy => {normal: 50%, cold: 40%, dizzy: 10%}`` and
    ``fever => {normal: 10%, cold: 30%, dizzy: 60%}``

    Finally, we have transition probabilities:

    ``healthy => healthy (70%)`` and
    ``fever => fever (60%)``.

    Over three days, we observe the sequence ``[normal, cold, dizzy]``,
    and wish to know the maximum likelihood assignment of states for the
    corresponding days, which we compute with the Viterbi algorithm below.

    >>> p_init = np.array([0.6, 0.4])
    >>> p_emit = np.array([[0.5, 0.4, 0.1],
    ...                    [0.1, 0.3, 0.6]])
    >>> p_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> path, logp = librosa.sequence.viterbi(p_emit, p_trans, p_init,
    ...                                       return_logp=True)
    >>> print(logp, path)
    -4.19173690823075 [0 0 1]
    '''

    n_states, n_steps = prob.shape

    if transition.shape != (n_states, n_states):
        raise ParameterError('transition.shape={}, must be '
                             '(n_states, n_states)={}'.format(transition.shape,
                                                              (n_states, n_states)))

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError('Invalid transition matrix: must be non-negative '
                             'and sum to 1 on each row.')

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError('Invalid probability values: must be between 0 and 1.')

    states = np.zeros(n_steps, dtype=int)
    values = np.zeros((n_steps, n_states), dtype=float)
    ptr = np.zeros((n_steps, n_states), dtype=int)

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = np.finfo(prob.dtype).tiny
    log_trans = np.log(transition + epsilon)
    log_prob = np.log(prob.T + epsilon)

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1./n_states)
    elif np.any(p_init < 0) or not np.allclose(p_init.sum(), 1):
        raise ParameterError('Invalid initial state distribution: '
                             'p_init={}'.format(p_init))

    log_p_init = np.log(p_init + epsilon)

    _viterbi(log_prob, log_trans, log_p_init, states, values, ptr)

    if return_logp:
        return states, values[-1, states[-1]]

    return states


def viterbi_discriminative(prob, transition, p_state=None, p_init=None, return_logp=False):
    '''Viterbi decoding from discriminative state predictions.

    Given a sequence of conditional state predictions `prob[s, t]`,
    indicating the conditional likelihood of state `s` given the
    observation at time `t`, and a transition matrix `transition[i, j]`
    which encodes the conditional probability of moving from state `i`
    to state `j`, the Viterbi algorithm computes the most likely sequence
    of states from the observations.

    This implementation uses the standard Viterbi decoding algorithm
    for observation likelihood sequences, under the assumption that
    `P[Obs(t) | State(t) = s]` is proportional to
    `P[State(t) = s | Obs(t)] / P[State(t) = s]`, where the denominator
    is the marginal probability of state `s` occurring as given by `p_state`.

    Parameters
    ----------
    prob : np.ndarray [shape=(n_states, n_steps), non-negative]
        `prob[s, t]` is the probability of state `s` conditional on
        the observation at time `t`.
        Must be non-negative and sum to 1 along each column.

    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        `transition[i, j]` is the probability of a transition from i->j.
        Each row must sum to 1.

    p_state : np.ndarray [shape=(n_states,)]
        Optional: marginal probability distribution over states,
        must be non-negative and sum to 1.
        If not provided, a uniform distribution is assumed.

    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, it is assumed to be uniform.

    return_logp : bool
        If `True`, return the log-likelihood of the state sequence.

    Returns
    -------
    Either `states` or `(states, logp)`:

    states : np.ndarray [shape=(n_steps,)]
        The most likely state sequence.

    logp : scalar [float]
        If `return_logp=True`, the log probability of `states` given
        the observations.

    See Also
    --------
    viterbi : Viterbi decoding from observation likelihoods
    viterbi_binary: Viterbi decoding for multi-label, conditional state likelihoods

    Examples
    --------
    This example constructs a simple, template-based discriminative chord estimator,
    using CENS chroma as input features.

    .. note:: this chord model is not accurate enough to use in practice. It is only
            intended to demonstrate how to use discriminative Viterbi decoding.

    >>> # Create templates for major, minor, and no-chord qualities
    >>> maj_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0])
    >>> min_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0])
    >>> N_template   = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 4.
    >>> # Generate the weighting matrix that maps chroma to labels
    >>> weights = np.zeros((25, 12), dtype=float)
    >>> labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
    ...           'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
    ...           'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
    ...           'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
    ...           'N']
    >>> for c in range(12):
    ...     weights[c, :] = np.roll(maj_template, c) # c:maj
    ...     weights[c + 12, :] = np.roll(min_template, c)  # c:min
    >>> weights[-1] = N_template  # the last row is the no-chord class
    >>> # Make a self-loop transition matrix over 25 states
    >>> trans = librosa.sequence.transition_loop(25, 0.9)

    >>> # Load in audio and make features
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> chroma = librosa.feature.chroma_cens(y=y, sr=sr, bins_per_octave=36)
    >>> # Map chroma (observations) to class (state) likelihoods
    >>> probs = np.exp(weights.dot(chroma))  # P[class | chroma] proportional to exp(template' chroma)
    >>> probs /= probs.sum(axis=0, keepdims=True)  # probabilities must sum to 1 in each column
    >>> # Compute independent frame-wise estimates
    >>> chords_ind = np.argmax(probs, axis=0)
    >>> # And viterbi estimates
    >>> chords_vit = librosa.sequence.viterbi_discriminative(probs, trans)

    >>> # Plot the features and prediction map
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> plt.subplot(2,1,1)
    >>> librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
    >>> plt.colorbar()
    >>> plt.subplot(2,1,2)
    >>> librosa.display.specshow(weights, x_axis='chroma')
    >>> plt.yticks(np.arange(25) + 0.5, labels)
    >>> plt.ylabel('Chord')
    >>> plt.colorbar()
    >>> plt.tight_layout()

    >>> # And plot the results
    >>> plt.figure(figsize=(10, 4))
    >>> librosa.display.specshow(probs, x_axis='time', cmap='gray')
    >>> plt.colorbar()
    >>> times = librosa.frames_to_time(np.arange(len(chords_vit)))
    >>> plt.scatter(times, chords_ind + 0.75, color='lime', alpha=0.5, marker='+', s=15, label='Independent')
    >>> plt.scatter(times, chords_vit + 0.25, color='deeppink', alpha=0.5, marker='o', s=15, label='Viterbi')
    >>> plt.yticks(0.5 + np.unique(chords_vit), [labels[i] for i in np.unique(chords_vit)], va='center')
    >>> plt.legend(loc='best')
    >>> plt.tight_layout()

    '''

    n_states, n_steps = prob.shape

    if transition.shape != (n_states, n_states):
        raise ParameterError('transition.shape={}, must be '
                             '(n_states, n_states)={}'.format(transition.shape,
                                                              (n_states, n_states)))

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError('Invalid transition matrix: must be non-negative '
                             'and sum to 1 on each row.')

    if np.any(prob < 0) or not np.allclose(prob.sum(axis=0), 1):
        raise ParameterError('Invalid probability values: each column must '
                             'sum to 1 and be non-negative')

    states = np.zeros(n_steps, dtype=int)
    values = np.zeros((n_steps, n_states), dtype=float)
    ptr = np.zeros((n_steps, n_states), dtype=int)

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = np.finfo(prob.dtype).tiny

    # Compute marginal log probabilities while avoiding underflow
    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(1./n_states)
    elif p_state.shape != (n_states,):
        raise ParameterError('Marginal distribution p_state must have shape (n_states,). '
                             'Got p_state.shape={}'.format(p_state.shape))
    elif np.any(p_state < 0) or not np.allclose(p_state.sum(axis=-1), 1):
        raise ParameterError('Invalid marginal state distribution: '
                             'p_state={}'.format(p_state))

    log_trans = np.log(transition + epsilon)
    log_marginal = np.log(p_state + epsilon)

    # By Bayes' rule, P[X | Y] * P[Y] = P[Y | X] * P[X]
    # P[X] is constant for the sake of maximum likelihood inference
    # and P[Y] is given by the marginal distribution p_state.
    #
    # So we have P[X | y] \propto P[Y | x] / P[Y]
    # if X = observation and Y = states, this can be done in log space as
    # log P[X | y] \propto \log P[Y | x] - \log P[Y]
    log_prob = np.log(prob.T + epsilon) - log_marginal

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1./n_states)
    elif np.any(p_init < 0) or not np.allclose(p_init.sum(), 1):
        raise ParameterError('Invalid initial state distribution: '
                             'p_init={}'.format(p_init))

    log_p_init = np.log(p_init + epsilon)

    _viterbi(log_prob, log_trans, log_p_init, states, values, ptr)

    if return_logp:
        return states, values[-1, states[-1]]

    return states


def viterbi_binary(prob, transition, p_state=None, p_init=None, return_logp=False):
    '''Viterbi decoding from binary (multi-label), discriminative state predictions.

    Given a sequence of conditional state predictions `prob[s, t]`,
    indicating the conditional likelihood of state `s` being active
    conditional on observation at time `t`, and a 2*2 transition matrix
    `transition` which encodes the conditional probability of moving from
    state `s` to state `~s` (not-`s`), the Viterbi algorithm computes the
    most likely sequence of states from the observations.

    This function differs from `viterbi_discriminative` in that it does not assume the
    states to be mutually exclusive.  `viterbi_binary` is implemented by
    transforming the multi-label decoding problem to a collection
    of binary Viterbi problems (one for each *state* or label).

    The output is a binary matrix `states[s, t]` indicating whether each
    state `s` is active at time `t`.

    Parameters
    ----------
    prob : np.ndarray [shape=(n_steps,) or (n_states, n_steps)], non-negative
        `prob[s, t]` is the probability of state `s` being active
        conditional on the observation at time `t`.
        Must be non-negative and less than 1.

        If `prob` is 1-dimensional, it is expanded to shape `(1, n_steps)`.

    transition : np.ndarray [shape=(2, 2) or (n_states, 2, 2)], non-negative
        If 2-dimensional, the same transition matrix is applied to each sub-problem.
        `transition[0, i]` is the probability of the state going from inactive to `i`,
        `transition[1, i]` is the probability of the state going from active to `i`.
        Each row must sum to 1.

        If 3-dimensional, `transition[s]` is interpreted as the 2x2 transition matrix
        for state label `s`.

    p_state : np.ndarray [shape=(n_states,)]
        Optional: marginal probability for each state (between [0,1]).
        If not provided, a uniform distribution (0.5 for each state)
        is assumed.

    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, it is assumed to be uniform.

    return_logp : bool
        If `True`, return the log-likelihood of the state sequence.

    Returns
    -------
    Either `states` or `(states, logp)`:

    states : np.ndarray [shape=(n_states, n_steps)]
        The most likely state sequence.

    logp : np.ndarray [shape=(n_states,)]
        If `return_logp=True`, the log probability of each state activation
        sequence `states`

    See Also
    --------
    viterbi : Viterbi decoding from observation likelihoods
    viterbi_discriminative : Viterbi decoding for discriminative (mutually exclusive) state predictions

    Examples
    --------
    In this example, we have a sequence of binary state likelihoods that we want to de-noise
    under the assumption that state changes are relatively uncommon.  Positive predictions
    should only be retained if they persist for multiple steps, and any transient predictions
    should be considered as errors.  This use case arises frequently in problems such as
    instrument recognition, where state activations tend to be stable over time, but subject
    to abrupt changes (e.g., when an instrument joins the mix).

    We assume that the 0 state has a self-transition probability of 90%, and the 1 state
    has a self-transition probability of 70%.  We assume the marginal and initial
    probability of either state is 50%.

    >>> trans = np.array([[0.9, 0.1], [0.3, 0.7]])
    >>> prob = np.array([0.1, 0.7, 0.4, 0.3, 0.8, 0.9, 0.8, 0.2, 0.6, 0.3])
    >>> librosa.sequence.viterbi_binary(prob, trans, p_state=0.5, p_init=0.5)
    array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])
    '''

    prob = np.atleast_2d(prob)

    n_states, n_steps = prob.shape

    if transition.shape == (2, 2):
        transition = np.tile(transition, (n_states, 1, 1))
    elif transition.shape != (n_states, 2, 2):
        raise ParameterError('transition.shape={}, must be (2,2) or '
                             '(n_states, 2, 2)={}'.format(transition.shape, (n_states)))

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=-1), 1):
        raise ParameterError('Invalid transition matrix: must be non-negative '
                             'and sum to 1 on each row.')

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError('Invalid probability values: prob must be between [0, 1]')

    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(0.5)
    else:
        p_state = np.atleast_1d(p_state)

    if p_state.shape != (n_states,) or np.any(p_state < 0) or np.any(p_state > 1):
        raise ParameterError('Invalid marginal state distributions: p_state={}'.format(p_state))

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(0.5)
    else:
        p_init = np.atleast_1d(p_init)

    if p_init.shape != (n_states,) or np.any(p_init < 0) or np.any(p_init > 1):
        raise ParameterError('Invalid initial state distributions: p_init={}'.format(p_init))

    states = np.empty((n_states, n_steps), dtype=int)
    logp = np.empty(n_states)

    prob_binary = np.empty((2, n_steps))
    p_state_binary = np.empty(2)
    p_init_binary = np.empty(2)

    for state in range(n_states):
        prob_binary[0] = 1 - prob[state]
        prob_binary[1] = prob[state]

        p_state_binary[0] = 1 - p_state[state]
        p_state_binary[1] = p_state[state]

        p_init_binary[0] = 1 - p_init[state]
        p_init_binary[1] = p_init[state]

        states[state, :], logp[state] = viterbi_discriminative(prob_binary,
                                                               transition[state],
                                                               p_state=p_state_binary,
                                                               p_init=p_init_binary,
                                                               return_logp=True)

    if return_logp:
        return states, logp

    return states


def transition_uniform(n_states):
    '''Construct a uniform transition matrix over `n_states`.

    Parameters
    ----------
    n_states : int > 0
        The number of states

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        `transition[i, j] = 1./n_states`

    Examples
    --------

    >>> librosa.sequence.transition_uniform(3)
    array([[0.333, 0.333, 0.333],
           [0.333, 0.333, 0.333],
           [0.333, 0.333, 0.333]])
    '''

    if not isinstance(n_states, int) or n_states <= 0:
        raise ParameterError('n_states={} must be a positive integer')

    transition = np.empty((n_states, n_states), dtype=np.float)
    transition.fill(1./n_states)
    return transition


def transition_loop(n_states, prob):
    '''Construct a self-loop transition matrix over `n_states`.

    The transition matrix will have the following properties:

        - `transition[i, i] = p` for all i
        - `transition[i, j] = (1 - p) / (n_states - 1)` for all `j != i`

    This type of transition matrix is appropriate when states tend to be
    locally stable, and there is no additional structure between different
    states.  This is primarily useful for de-noising frame-wise predictions.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length `n_states`, `p[i]` is the probability of state `i`'s self-transition.

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    >>> librosa.sequence.transition_loop(3, 0.5)
    array([[0.5 , 0.25, 0.25],
           [0.25, 0.5 , 0.25],
           [0.25, 0.25, 0.5 ]])

    >>> librosa.sequence.transition_loop(3, [0.8, 0.5, 0.25])
    array([[0.8  , 0.1  , 0.1  ],
           [0.25 , 0.5  , 0.25 ],
           [0.375, 0.375, 0.25 ]])
    '''

    if not isinstance(n_states, int) or n_states <= 1:
        raise ParameterError('n_states={} must be a positive integer > 1')

    transition = np.empty((n_states, n_states), dtype=np.float)

    # if it's a float, make it a vector
    prob = np.asarray(prob, dtype=np.float)

    if prob.ndim == 0:
        prob = np.tile(prob, n_states)

    if prob.shape != (n_states,):
        raise ParameterError('prob={} must have length equal to n_states={}'.format(prob, n_states))

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError('prob={} must have values in the range [0, 1]'.format(prob))

    for i, prob_i in enumerate(prob):
        transition[i] = (1. - prob_i) / (n_states - 1)
        transition[i, i] = prob_i

    return transition


def transition_cycle(n_states, prob):
    '''Construct a cyclic transition matrix over `n_states`.

    The transition matrix will have the following properties:

        - `transition[i, i] = p`
        - `transition[i, i + 1] = (1 - p)`

    This type of transition matrix is appropriate for state spaces
    with cyclical structure, such as metrical position within a bar.
    For example, a song in 4/4 time has state transitions of the form

        1->{1, 2}, 2->{2, 3}, 3->{3, 4}, 4->{4, 1}.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length `n_states`, `p[i]` is the probability of state
        `i`'s self-transition.

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    >>> librosa.sequence.transition_cycle(4, 0.9)
    array([[0.9, 0.1, 0. , 0. ],
           [0. , 0.9, 0.1, 0. ],
           [0. , 0. , 0.9, 0.1],
           [0.1, 0. , 0. , 0.9]])
    '''

    if not isinstance(n_states, int) or n_states <= 1:
        raise ParameterError('n_states={} must be a positive integer > 1')

    transition = np.zeros((n_states, n_states), dtype=np.float)

    # if it's a float, make it a vector
    prob = np.asarray(prob, dtype=np.float)

    if prob.ndim == 0:
        prob = np.tile(prob, n_states)

    if prob.shape != (n_states,):
        raise ParameterError('prob={} must have length equal to n_states={}'.format(prob, n_states))

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError('prob={} must have values in the range [0, 1]'.format(prob))

    for i, prob_i in enumerate(prob):
        transition[i, np.mod(i + 1, n_states)] = 1. - prob_i
        transition[i, i] = prob_i

    return transition


def transition_local(n_states, width, window='triangle', wrap=False):
    '''Construct a localized transition matrix.

    The transition matrix will have the following properties:

        - `transition[i, j] = 0` if `|i - j| > width`
        - `transition[i, i]` is maximal
        - `transition[i, i - width//2 : i + width//2]` has shape `window`

    This type of transition matrix is appropriate for state spaces
    that discretely approximate continuous variables, such as in fundamental
    frequency estimation.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    width : int >= 1 or iterable
        The maximum number of states to treat as "local".
        If iterable, it should have length equal to `n_states`,
        and specify the width independently for each state.

    window : str, callable, or window specification
        The window function to determine the shape of the "local" distribution.

        Any window specification supported by `filters.get_window` will work here.

        .. note:: Certain windows (e.g., 'hann') are identically 0 at the boundaries,
            so and effectively have `width-2` non-zero values.  You may have to expand
            `width` to get the desired behavior.


    wrap : bool
        If `True`, then state locality `|i - j|` is computed modulo `n_states`.
        If `False` (default), then locality is absolute.

    See Also
    --------
    filters.get_window

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------

    Triangular distributions with and without wrapping

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=False)
    array([[0.667, 0.333, 0.   , 0.   , 0.   ],
           [0.25 , 0.5  , 0.25 , 0.   , 0.   ],
           [0.   , 0.25 , 0.5  , 0.25 , 0.   ],
           [0.   , 0.   , 0.25 , 0.5  , 0.25 ],
           [0.   , 0.   , 0.   , 0.333, 0.667]])

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=True)
    array([[0.5 , 0.25, 0.  , 0.  , 0.25],
           [0.25, 0.5 , 0.25, 0.  , 0.  ],
           [0.  , 0.25, 0.5 , 0.25, 0.  ],
           [0.  , 0.  , 0.25, 0.5 , 0.25],
           [0.25, 0.  , 0.  , 0.25, 0.5 ]])

    Uniform local distributions with variable widths and no wrapping

    >>> librosa.sequence.transition_local(5, [1, 2, 3, 3, 1], window='ones', wrap=False)
    array([[1.   , 0.   , 0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.   , 0.   , 0.   ],
           [0.   , 0.333, 0.333, 0.333, 0.   ],
           [0.   , 0.   , 0.333, 0.333, 0.333],
           [0.   , 0.   , 0.   , 0.   , 1.   ]])
    '''

    if not isinstance(n_states, int) or n_states <= 1:
        raise ParameterError('n_states={} must be a positive integer > 1')

    width = np.asarray(width, dtype=int)
    if width.ndim == 0:
        width = np.tile(width, n_states)

    if width.shape != (n_states,):
        raise ParameterError('width={} must have length equal to n_states={}'.format(width, n_states))

    if np.any(width < 1):
        raise ParameterError('width={} must be at least 1')

    transition = np.zeros((n_states, n_states), dtype=np.float)

    # Fill in the widths.  This is inefficient, but simple
    for i, width_i in enumerate(width):
        trans_row = pad_center(get_window(window, width_i, fftbins=False), n_states)
        trans_row = np.roll(trans_row, n_states//2 + i + 1)

        if not wrap:
            # Knock out the off-diagonal-band elements
            trans_row[min(n_states, i + width_i//2 + 1):] = 0
            trans_row[:max(0, i - width_i//2)] = 0

        transition[i] = trans_row

    # Row-normalize
    transition /= transition.sum(axis=1, keepdims=True)

    return transition
