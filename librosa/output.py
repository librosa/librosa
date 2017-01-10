#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Output
======

Text output
-----------
.. autosummary::
    :toctree: generated/

    annotation
    times_csv

Audio output
------------
.. autosummary::
    :toctree: generated/

    write_wav
"""

import csv

import numpy as np
import scipy
import scipy.io.wavfile

from . import util
from .util.exceptions import ParameterError


__all__ = ['annotation', 'times_csv', 'write_wav']


def annotation(path, intervals, annotations=None, delimiter=',', fmt='%0.3f'):
    r'''Save annotations in a 3-column format::

        intervals[0, 0],intervals[0, 1],annotations[0]\n
        intervals[1, 0],intervals[1, 1],annotations[1]\n
        intervals[2, 0],intervals[2, 1],annotations[2]\n
        ...

    This can be used for segment or chord annotations.

    Parameters
    ----------
    path : str
        path to save the output CSV file

    intervals : np.ndarray [shape=(n, 2)]
        array of interval start and end-times.

        `intervals[i, 0]` marks the start time of interval `i`

        `intervals[i, 1]` marks the end time of interval `i`

    annotations : None or list-like [shape=(n,)]
        optional list of annotation strings. `annotations[i]` applies
        to the time range `intervals[i, 0]` to `intervals[i, 1]`

    delimiter : str
        character to separate fields

    fmt : str
        format-string for rendering time data

    Raises
    ------
    ParameterError
        if `annotations` is not `None` and length does
        not match `intervals`

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> data = librosa.feature.mfcc(y=y, sr=sr, hop_length=512)

    Detect segment boundaries

    >>> boundaries = librosa.segment.agglomerative(data, k=10)

    Convert to time

    >>> boundary_times = librosa.frames_to_time(boundaries, sr=sr,
    ...                                         hop_length=512)

    Convert events boundaries to intervals

    >>> intervals = np.hstack([boundary_times[:-1, np.newaxis],
    ...                        boundary_times[1:, np.newaxis]])


    Make some fake annotations

    >>> labels = ['Seg #{:03d}'.format(i) for i in range(len(intervals))]

    Save the output

    >>> librosa.output.annotation('segments.csv', intervals,
    ...                           annotations=labels)
    '''

    util.valid_intervals(intervals)

    if annotations is not None and len(annotations) != len(intervals):
        raise ParameterError('len(annotations) != len(intervals)')

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)

        if annotations is None:
            for t_int in intervals:
                writer.writerow([fmt % t_int[0], fmt % t_int[1]])
        else:
            for t_int, lab in zip(intervals, annotations):
                writer.writerow([fmt % t_int[0], fmt % t_int[1], lab])


def times_csv(path, times, annotations=None, delimiter=',', fmt='%0.3f'):
    r"""Save time steps as in CSV format.  This can be used to store the output
    of a beat-tracker or segmentation algorihtm.

    If only `times` are provided, the file will contain each value
    of `times` on a row::

        times[0]\n
        times[1]\n
        times[2]\n
        ...

    If `annotations` are also provided, the file will contain
    delimiter-separated values::

        times[0],annotations[0]\n
        times[1],annotations[1]\n
        times[2],annotations[2]\n
        ...


    Parameters
    ----------
    path : string
        path to save the output CSV file

    times : list-like of floats
        list of frame numbers for beat events

    annotations : None or list-like
        optional annotations for each time step

    delimiter : str
        character to separate fields

    fmt : str
        format-string for rendering time

    Raises
    ------
    ParameterError
        if `annotations` is not `None` and length does not
        match `times`

    Examples
    --------
    Write beat-tracker time to CSV

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, units='time')
    >>> librosa.output.times_csv('beat_times.csv', beats)
    """

    if annotations is not None and len(annotations) != len(times):
        raise ParameterError('len(annotations) != len(times)')

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)

        if annotations is None:
            for t in times:
                writer.writerow([fmt % t])
        else:
            for t, lab in zip(times, annotations):
                writer.writerow([(fmt % t), lab])


def write_wav(path, y, sr, norm=False):
    """Output a time series as a .wav file

    Parameters
    ----------
    path : str
        path to save the output wav file

    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)

    sr : int > 0 [scalar]
        sampling rate of `y`

    norm : boolean [scalar]
        enable amplitude normalization.
        For floating point `y`, scale the data to the range [-1, +1].

    Examples
    --------
    Trim a signal to 5 seconds and save it back

    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=5.0)
    >>> librosa.output.write_wav('file_trim_5s.wav', y, sr)

    """

    # Validate the buffer.  Stereo is okay here.
    util.valid_audio(y, mono=False)

    # normalize
    if norm and np.issubdtype(y.dtype, np.float):
        wav = util.normalize(y, norm=np.inf, axis=None)
    else:
        wav = y

    # Check for stereo
    if wav.ndim > 1 and wav.shape[0] == 2:
        wav = wav.T

    # Save
    scipy.io.wavfile.write(path, sr, wav)
