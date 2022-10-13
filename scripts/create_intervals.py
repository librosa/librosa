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
import sympy
import numpy as np
import librosa


def fraction(x):
    """Wrapper to find a rational approximation to floating point numbers"""

    return sympy.nsimplify(np.array(x), rational=True, full=True, tolerance=1e-4)


def main():

    # Get the intervals
    intervals_pythagorean = librosa.pythagorean_intervals(
        bins_per_octave=72, sort=False
    )
    intervals_3lim = librosa.plimit_intervals(
        primes=[3], bins_per_octave=72, sort=False
    )
    intervals_5lim = librosa.plimit_intervals(
        primes=[3, 5], bins_per_octave=72, sort=False
    )
    intervals_7lim = librosa.plimit_intervals(
        primes=[3, 5, 7], bins_per_octave=72, sort=False
    )
    intervals_23lim = librosa.plimit_intervals(
        primes=[3, 5, 7, 11, 13, 17, 19, 23], bins_per_octave=190, sort=False
    )

    all_intervals = np.concatenate(
        (intervals_pythagorean, intervals_3lim, intervals_5lim, intervals_7lim, intervals_23lim)
    )
    # Convert to rationals
    fractions = fraction(all_intervals)

    # Factorize the rationals and cache them, keyed by truncated float
    factorized = {
        np.around(interval, 6): sympy.factorrat(fraction)
        for (interval, fraction) in zip(all_intervals, fractions)
    }

    with open("intervals.msgpack", "wb") as fdesc:
        msgpack.dump(factorized, fdesc)


if __name__ == "__main__":
    main()
