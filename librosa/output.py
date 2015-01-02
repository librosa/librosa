#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Output routines for audio and analysis"""

import csv

import numpy as np
import scipy
import scipy.io.wavfile

import librosa.core
from . import util


def annotation(path, intervals, annotations=None, delimiter=',', fmt='%0.3f'):
    r'''Save annotations in a 3-column format::

        intervals[0, 0],intervals[0, 1],annotations[0]\n
        intervals[1, 0],intervals[1, 1],annotations[1]\n
        intervals[2, 0],intervals[2, 1],annotations[2]\n
        ...

    This can be used for segment or chord annotations.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> data = librosa.feature.mfcc(y=y, sr=sr, hop_length=512)
        >>> # Detect segment boundaries
        >>> boundaries = librosa.segment.agglomerative(data, k=10)
        >>> # Convert to time
        >>> boundary_times = librosa.frames_to_time(boundaries, sr=sr,
                                                    hop_length=512)
        >>> # Make some fake annotations
        >>> labels = ['Seg #{:03d}'.format(i) for i in range(len(time_start))]
        >>> # Save the output
        >>> librosa.output.annotation('segments.csv', boundary_times,
                                      annotations=annotations)

    :parameters:
      - path : str
          path to save the output CSV file

      - intervals : np.ndarray [shape=(n, 2)]
          array of interval start and end-times.

          - ``intervals[i, 0]`` marks the start time of interval ``i``

          - ``intervals[i, 1]`` marks the endtime of interval ``i``

      - annotations : None or list-like [shape=(n,)]
          optional list of annotation strings. ``annotations[i]`` applies
          to the time range ``intervals[i, 0]`` to ``intervals[i, 1]``

      - delimiter : str
          character to separate fields

      - fmt : str
          format-string for rendering time data

    :raises:
      - ValueError
          if ``annotations`` is not ``None`` and length does
          not match ``intervals``
    '''

    if annotations is not None and len(annotations) != len(intervals):
        raise ValueError('len(annotations) != len(intervals)')

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)

        if annotations is None:
            for t_int in intervals:
                writer.writerow([fmt % t_int[0], fmt % t_int[1]])
        else:
            for t_int, lab in zip(intervals, annotations):
                writer.writerow([fmt % t_int[0], fmt % t_int[1], lab])


def frames_csv(path, frames, sr=22050, hop_length=512, **kwargs):
    """Convert frames to time and store tbrycehe output in CSV format.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=64)
        >>> librosa.output.frames_csv('beat_times.csv', beats,
                                      sr=sr, hop_length=64)

    :parameters:
      - path : string
          path to save the output CSV file

      - frames : list-like of ints
          list of frame numbers for beat events

      - sr : int > 0 [scalar]
          audio sampling rate

      - hop_length : int > 0 [scalar]
          number of samples between success frames

      - *kwargs*
          additional keyword arguments.  See :func:`librosa.output.times_csv`
    """

    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    times_csv(path, times, **kwargs)


def times_csv(path, times, annotations=None, delimiter=',', fmt='%0.3f'):
    r"""Save time steps as in CSV format.  This can be used to store the output
    of a beat-tracker or segmentation algorihtm.

    If only ``times`` are provided, the file will contain each value
    of ``times`` on a row::

        times[0]\n
        times[1]\n
        times[2]\n
        ...

    If ``annotations`` are also provided, the file will contain
    delimiter-separated values::

        times[0],annotations[0]\n
        times[1],annotations[1]\n
        times[2],annotations[2]\n
        ...


    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=64)
        >>> times = librosa.frames_to_time(beats, sr=sr, hop_length=64)
        >>> librosa.output.times_csv('beat_times.csv', times)

    :parameters:
      - path : string
          path to save the output CSV file

      - times : list-like of floats
          list of frame numbers for beat events

      - annotations : None or list-like
          optional annotations for each time step

      - delimiter : str
          character to separate fields

      - fmt : str
          format-string for rendering time

    :raises:
      - ValueError
          if ``annotations`` is not ``None`` and length does not
          match ``times``
    """

    if annotations is not None and len(annotations) != len(times):
        raise ValueError('len(annotations) != len(times)')

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)

        if annotations is None:
            for t in times:
                writer.writerow([fmt % t])
        else:
            for t, lab in zip(times, annotations):
                writer.writerow([(fmt % t), lab])


def write_wav(path, y, sr, norm=True):
    """Output a time series as a .wav file

    :usage:
        >>> # Trim a signal to 5 seconds and save it back
        >>> y, sr = librosa.load(librosa.util.example_audio_file(),
                                 duration=5.0)
        >>> librosa.output.write_wav('file_trim_5s.wav', y, sr)

    :parameters:
      - path : str
          path to save the output wav file

      - y : np.ndarray [shape=(n,) or (2,n)]
          audio time series (mono or stereo)

      - sr : int > 0 [scalar]
          sampling rate of ``y``

      - norm : boolean [scalar]
          enable amplitude normalization
    """

    # Validate the buffer.  Stereo is okay here.
    util.valid_audio(y, mono=False)

    # normalize
    if norm:
        wav = util.normalize(y, norm=np.inf, axis=None)
    else:
        wav = y

    # Convert to 16bit int
    wav = util.buf_to_int(wav)

    # Check for stereo
    if wav.ndim > 1 and wav.shape[0] == 2:
        wav = wav.T

    # Save
    scipy.io.wavfile.write(path, sr, wav)
