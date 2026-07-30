#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2022-08-11 07:26:21 by Brian McFee <brian.mcfee@nyu.edu>
"""Construct the interval cache for just intonation systems.

This creates the data file intervals.json, which maps truncated floating point
representations of intervals to their prime factorizations.

This script is primarily intended for developer use.

Usage:

    python create_intervals.py

The output will be stored in intervals.pickle
"""

import msgpack
import numpy as np
import librosa


def main():

    # Get the intervals
    intervals_pythagorean = librosa.pythagorean_intervals(
        bins_per_octave=72, sort=False, return_factors=True
    )
    intervals_3lim = librosa.plimit_intervals(
        primes=[3],
        bins_per_octave=72,
        sort=False,
        return_factors=True,
    )
    intervals_5lim = librosa.plimit_intervals(
        primes=[3, 5],
        bins_per_octave=72,
        sort=False,
        return_factors=True,
    )
    intervals_7lim = librosa.plimit_intervals(
        primes=[3, 5, 7],
        bins_per_octave=72,
        sort=False,
        return_factors=True,
    )
    intervals_23lim = librosa.plimit_intervals(
        primes=[3, 5, 7, 11, 13, 17, 19, 23],
        bins_per_octave=190,
        sort=False,
        return_factors=True,
    )

    all_intervals = np.concatenate(
        (
            intervals_pythagorean,
            intervals_3lim,
            intervals_5lim,
            intervals_7lim,
            intervals_23lim,
        )
    )

    # Factorize the rationals and cache them, keyed by truncated float

    factorized = dict()
    for interval in all_intervals:
        # Compute the interval
        log_value = 0
        for p in interval:
            log_value += np.log2(p) * interval[p]
        value = np.around(np.power(2.0, log_value), 6)

        factorized[float(value)] = {
            int(p): int(interval[p]) for p in interval if interval[p] != 0
        }

    with open("intervals.msgpack", "wb") as fdesc:
        msgpack.dump(factorized, fdesc)


if __name__ == "__main__":
    main()
