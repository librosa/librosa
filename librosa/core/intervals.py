#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Functions for interval construction"""

from importlib import resources
from typing import Collection, Dict, List, Union, overload, Iterable
from typing_extensions import Literal
import msgpack
import numpy as np
from numpy.typing import ArrayLike
from .._cache import cache
from .._typing import _FloatLike_co


with resources.path("librosa.core", "intervals.msgpack") as imsgpack:
    with imsgpack.open("rb") as _fdesc:
        # We use floats for dictionary keys, so strict mapping is disabled
        INTERVALS = msgpack.load(_fdesc, strict_map_key=False)


@cache(level=10)
def interval_frequencies(
    n_bins: int,
    *,
    fmin: _FloatLike_co,
    intervals: Union[str, Collection[float]],
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    sort: bool = True
) -> np.ndarray:
    """Construct a set of frequencies from an interval set

    Parameters
    ----------
    n_bins : int
        The number of frequencies to generate

    fmin : float > 0
        The minimum frequency

    intervals : str or array of floats in [1, 2)
        If `str`, must be one of the following:
        - `'equal'` - equal temperament
        - `'pythagorean'` - Pythagorean intervals
        - `'ji3'` - 3-limit just intonation
        - `'ji5'` - 5-limit just intonation
        - `'ji7'` - 7-limit just intonation

        Otherwise, an array of intervals in the range [1, 2) can be provided.

    bins_per_octave : int > 0
        If `intervals` is a string specification, how many bins to
        generate per octave.
        If `intervals` is an array, then this parameter is ignored.

    tuning : float
        Deviation from A440 tuning in fractional bins.
        This is only used when `intervals == 'equal'`

    sort : bool
        Sort the intervals in ascending order.

    Returns
    -------
    frequencies : array of float
        The frequencies

    Examples
    --------
    Generate two octaves of Pythagorean intervals starting at 55Hz

    >>> librosa.interval_frequencies(24, fmin=55, intervals="pythagorean", bins_per_octave=12)
    array([ 55.   ,  58.733,  61.875,  66.075,  69.609,  74.334,  78.311,
            82.5  ,  88.099,  92.812,  99.112, 104.414, 110.   , 117.466,
           123.75 , 132.149, 139.219, 148.668, 156.621, 165.   , 176.199,
           185.625, 198.224, 208.828])

    Generate two octaves of 5-limit intervals starting at 55Hz

    >>> librosa.interval_frequencies(24, fmin=55, intervals="ji5", bins_per_octave=12)
    array([ 55.   ,  58.667,  61.875,  66.   ,  68.75 ,  73.333,  77.344,
            82.5  ,  88.   ,  91.667,  99.   , 103.125, 110.   , 117.333,
           123.75 , 132.   , 137.5  , 146.667, 154.687, 165.   , 176.   ,
           183.333, 198.   , 206.25 ])

    Generate three octaves using only three intervals

    >>> intervals = [1, 4/3, 3/2]
    >>> librosa.interval_frequencies(9, fmin=55, intervals=intervals)
    array([ 55.   ,  73.333,  82.5  , 110.   , 146.667, 165.   , 220.   ,
       293.333, 330.   ])
    """
    if isinstance(intervals, str):
        if intervals == "equal":
            # Maybe include tuning here?
            ratios = 2.0 ** (
                (tuning + np.arange(0, bins_per_octave, dtype=float)) / bins_per_octave
            )
        elif intervals == "pythagorean":
            ratios = pythagorean_intervals(bins_per_octave=bins_per_octave, sort=sort)
        elif intervals == "ji3":
            ratios = plimit_intervals(
                primes=[3], bins_per_octave=bins_per_octave, sort=sort
            )
        elif intervals == "ji5":
            ratios = plimit_intervals(
                primes=[3, 5], bins_per_octave=bins_per_octave, sort=sort
            )
        elif intervals == "ji7":
            ratios = plimit_intervals(
                primes=[3, 5, 7], bins_per_octave=bins_per_octave, sort=sort
            )
    else:
        ratios = np.array(intervals)
        bins_per_octave = len(ratios)

    # We have one octave of ratios, tile it up to however many we need
    # and trim back to the right number of bins
    n_octaves = np.ceil(n_bins / bins_per_octave)
    all_ratios = np.multiply.outer(2.0 ** np.arange(n_octaves), ratios).flatten()[
        :n_bins
    ]

    if sort:
        all_ratios = np.sort(all_ratios)

    return all_ratios * fmin


@overload
def pythagorean_intervals(
    *,
    bins_per_octave: int = ...,
    sort: bool = ...,
    return_factors: Literal[False] = ...
) -> np.ndarray:
    ...


@overload
def pythagorean_intervals(
    *, bins_per_octave: int = ..., sort: bool = ..., return_factors: Literal[True]
) -> List[Dict[int, int]]:
    ...


@overload
def pythagorean_intervals(
    *, bins_per_octave: int = ..., sort: bool = ..., return_factors: bool = ...
) -> Union[np.ndarray, List[Dict[int, int]]]:
    ...


@cache(level=10)
def pythagorean_intervals(
    *, bins_per_octave: int = 12, sort: bool = True, return_factors: bool = False
) -> Union[np.ndarray, List[Dict[int, int]]]:
    """Pythagorean intervals

    Intervals are constructed by stacking ratios of 3/2 (i.e.,
    just perfect fifths) and folding down to a single octave::

        1, 3/2, 9/8, 27/16, 81/64, ...

    Note that this differs from 3-limit just intonation intervals
    in that Pythagorean intervals only use positive powers of 3
    (ascending fifths) while 3-limit intervals use both positive
    and negative powers (descending fifths).

    Parameters
    ----------
    bins_per_octave : int
        The number of intervals to generate
    sort : bool
        If `True` then intervals are returned in ascending order.
        If `False`, then intervals are returned in circle-of-fifths order.
    return_factors : bool
        If `True` then return a list of dictionaries encoding the prime factorization
        of each interval as `{2: p2, 3: p3}` (meaning `3**p3 * 2**p2`).
        If `False` (default), return intervals as an array of floating point numbers.

    Returns
    -------
    intervals : np.ndarray or list of dictionaries
        The constructed interval set. All intervals are mapped
        to the range [1, 2).

    See Also
    --------
    plimit_intervals

    Examples
    --------
    Generate the first 12 intervals

    >>> librosa.pythagorean_intervals(bins_per_octave=12)
    array([1.      , 1.067871, 1.125   , 1.201355, 1.265625, 1.351524,
           1.423828, 1.5     , 1.601807, 1.6875  , 1.802032, 1.898437])
    >>> # Compare to the 12-tone equal temperament intervals:
    >>> 2**(np.arange(12)/12)
    array([1.      , 1.059463, 1.122462, 1.189207, 1.259921, 1.33484 ,
           1.414214, 1.498307, 1.587401, 1.681793, 1.781797, 1.887749])

    Or the first 7, in circle-of-fifths order

    >>> librosa.pythagorean_intervals(bins_per_octave=7, sort=False)
    array([1.      , 1.5     , 1.125   , 1.6875  , 1.265625, 1.898437,
           1.423828])

    Generate the first 7, in circle-of-fifths other and factored form

    >>> librosa.pythagorean_intervals(bins_per_octave=7, sort=False, return_factors=True)
    [
        {2: 0, 3: 0},
        {2: -1, 3: 1},
        {2: -3, 3: 2},
        {2: -4, 3: 3},
        {2: -6, 3: 4},
        {2: -7, 3: 5},
        {2: -9, 3: 6}
    ]
    """
    # Generate all powers of 3 in log space
    pow3 = np.arange(bins_per_octave)

    # Using modf here to quickly get the fractional part of the log,
    # accounting for whatever power of 2 is necessary to get 3**k
    # within the octave.
    log_ratios: np.ndarray
    pow2: np.ndarray
    log_ratios, pow2 = np.modf(pow3 * np.log2(3))

    # If the fractional part is negative, add
    # one more power of two to get it into the range [0, 1).
    too_small = log_ratios < 0
    log_ratios[too_small] += 1
    pow2[too_small] += 1

    # Convert powers of 2 to integer
    pow2 = pow2.astype(int)

    idx: Iterable[int]

    if sort:
        # Order the intervals
        idx = np.argsort(log_ratios)
        log_ratios = log_ratios[idx]
    else:
        # If not sorting, we'll take powers in order
        idx = range(bins_per_octave)

    if return_factors:
        return list({2: -pow2[i], 3: pow3[i]} for i in idx)

    return np.power(2, log_ratios)


def __harmonic_distance(logs, a, b):
    """Compute the harmonic distance between ratios a and b.

    Harmonic distance is defined as `log2(a * b) - 2*log2(gcd(a, b))` [#]_.

    Here we are expressing a and b as prime factorization exponents,
    and the prime basis are provided in their log2 form.

    .. [#] Tenney, James.
        "On ‘Crystal Growth’ in harmonic space (1993–1998)."
        Contemporary Music Review 27.1 (2008): 47-56.
    """
    a = np.array(a)
    b = np.array(b)

    # numerator = positive exponents
    a_num = np.maximum(a, 0)
    # denominator = negative exponents
    a_den = a_num - a

    b_num = np.maximum(b, 0)
    b_den = b_num - b

    # log2(ab / gcd(a,b)**2) = log(a) + log(b) - 2 * log(gcd(a,b))
    # gcd(a,b) for rationals: gcd(a_num, b_num) / lcm(a_den, b_den)
    # gcd = minimum(a_num, b_num) and lcm = maximum(a_den, b_den)
    gcd = np.minimum(a_num, b_num) - np.maximum(a_den, b_den)

    # Rounding this to 6 decimals to avoid floating point weirdness
    return np.around(logs.dot(a + b - 2 * gcd), 6)


def _crystal_tie_break(a, b, logs):
    """Given two tuples of prime powers, break ties."""
    return logs.dot(np.abs(a)) < logs.dot(np.abs(b))


@overload
def plimit_intervals(
    *,
    primes: ArrayLike,
    bins_per_octave: int = ...,
    sort: bool = ...,
    return_factors: Literal[False] = ...
) -> np.ndarray:
    ...


@overload
def plimit_intervals(
    *,
    primes: ArrayLike,
    bins_per_octave: int = ...,
    sort: bool = ...,
    return_factors: Literal[True]
) -> List[Dict[int, int]]:
    ...


@overload
def plimit_intervals(
    *,
    primes: ArrayLike,
    bins_per_octave: int = ...,
    sort: bool = ...,
    return_factors: bool = ...
) -> Union[np.ndarray, List[Dict[int, int]]]:
    ...


@cache(level=10)
def plimit_intervals(
    *,
    primes: ArrayLike,
    bins_per_octave: int = 12,
    sort: bool = True,
    return_factors: bool = False
) -> Union[np.ndarray, List[Dict[int, int]]]:
    """Construct p-limit intervals for a given set of prime factors.

    This function is based on the "harmonic crystal growth" algorithm
    of [#1]_ [#2]_.

    .. [#1] Tenney, James.
        "On ‘Crystal Growth’ in harmonic space (1993–1998)."
        Contemporary Music Review 27.1 (2008): 47-56.

    .. [#2] Sabat, Marc, and James Tenney.
        "Three crystal growth algorithms in 23-limit constrained harmonic space."
        Contemporary Music Review 27, no. 1 (2008): 57-78.

    Parameters
    ----------
    primes : array of odd primes
        Which prime factors are to be used
    bins_per_octave : int
        The number of intervals to construct
    sort : bool
        If `True` then intervals are returned in ascending order.
        If `False`, then intervals are returned in crystal growth order.
    return_factors : bool
        If `True` then return a list of dictionaries encoding the prime factorization
        of each interval as `{2: p2, 3: p3, ...}` (meaning `3**p3 * 2**p2`).
        If `False` (default), return intervals as an array of floating point numbers.

    Returns
    -------
    intervals : np.ndarray or list of dictionaries
        The constructed interval set. All intervals are mapped
        to the range [1, 2).

    See Also
    --------
    pythagorean_intervals

    Examples
    --------
    Compare 3-limit tuning to Pythagorean tuning and 12-TET

    >>> librosa.plimit_intervals(primes=[3], bins_per_octave=12)
    array([1.        , 1.05349794, 1.125     , 1.18518519, 1.265625  ,
           1.33333333, 1.40466392, 1.5       , 1.58024691, 1.6875    ,
           1.77777778, 1.8984375 ])
    >>> # Pythagorean intervals:
    >>> librosa.pythagorean_intervals(bins_per_octave=12)
    array([1.        , 1.06787109, 1.125     , 1.20135498, 1.265625  ,
           1.35152435, 1.42382812, 1.5       , 1.60180664, 1.6875    ,
           1.80203247, 1.8984375 ])
    >>> # 12-TET intervals:
    >>> 2**(np.arange(12)/12)
    array([1.        , 1.05946309, 1.12246205, 1.18920712, 1.25992105,
           1.33483985, 1.41421356, 1.49830708, 1.58740105, 1.68179283,
           1.78179744, 1.88774863])

    Create a 7-bin, 5-limit interval set

    >>> librosa.plimit_intervals(primes=[3, 5], bins_per_octave=7)
    array([1.        , 1.125     , 1.25      , 1.33333333, 1.5       ,
           1.66666667, 1.875     ])

    The same example, but now in factored form

    >>> librosa.plimit_intervals(primes=[3, 5], bins_per_octave=7,
    ...                          return_factors=True)
    [
        {},
        {2: -3, 3: 2},
        {2: -2, 5: 1},
        {2: 2, 3: -1},
        {2: -1, 3: 1},
        {3: -1, 5: 1},
        {2: -3, 3: 1, 5: 1}
    ]
    """
    primes = np.atleast_1d(primes)
    logs = np.log2(primes, dtype=np.float64)

    # The seed set are primes and their reciprocals
    # These are the values that we can use to expand our
    # interval set.  These are expressed in terms of the
    # prime factorization exponents
    seeds = []
    for i in range(len(primes)):
        # Add the prime
        seed = [0] * len(primes)
        seed[i] = 1
        seeds.append(tuple(seed))
        # Add the inverse
        seed[i] = -1
        seeds.append(tuple(seed))

    # The frontier is the set of candidate intervals for inclusion
    frontier = seeds.copy()

    # The distances table will let us keep track of the harmonic
    # distances between all selected intervals
    distances = dict()

    # Initialize the interval set with the root (1)
    intervals = list()
    root = tuple([0] * len(primes))
    intervals.append(root)

    while len(intervals) < bins_per_octave:
        # Find the element on the frontier that minimizes the total
        # harmonic distance to the existing set
        score = np.inf
        best_f = 0
        for f, point in enumerate(frontier):
            # Compute harmonic distance (HD) to each selected interval
            HD = 0.0

            for s in intervals:
                if (s, point) not in distances:
                    distances[s, point] = __harmonic_distance(logs, point, s)
                    distances[point, s] = distances[s, point]

                HD += distances[s, point]

            if HD < score or (
                np.isclose(HD, score)
                and _crystal_tie_break(point, frontier[best_f], logs)
            ):
                score = HD
                best_f = f

        new_point = frontier.pop(best_f)
        intervals.append(new_point)

        for _ in seeds:
            new_seed = tuple(np.array(new_point) + np.array(_))
            if new_seed not in intervals and new_seed not in frontier:
                frontier.append(new_seed)

    pows = np.array(list(intervals), dtype=float)

    log_ratios: np.ndarray
    pow2: np.ndarray
    log_ratios, pow2 = np.modf(pows.dot(logs))

    # If the fractional part is negative, add
    # one more power of two to get it into the range [0, 1).
    too_small = log_ratios < 0
    log_ratios[too_small] += 1
    pow2[too_small] -= 1

    # Convert powers of 2 to integer
    pow2 = pow2.astype(int)

    idx: Iterable[int]
    if sort:
        # Order the intervals
        idx = np.argsort(log_ratios)
        log_ratios = log_ratios[idx]
    else:
        # If not sorting, we'll take powers in order
        idx = range(bins_per_octave)

    if return_factors:
        # Collect the factorized intervals into a list
        factors = []
        for i in idx:
            v = dict()
            if pow2[i] != 0:
                v[2] = -pow2[i]

            v.update({p: int(power) for p, power in zip(primes, pows[i]) if power != 0})

            factors.append(v)
        return factors

    # Otherwise, just return intervals as floats
    return np.power(2, log_ratios)
