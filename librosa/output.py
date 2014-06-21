#!/usr/bin/env python
"""Output routines for audio and analysis"""

import csv

import numpy as np
import scipy
import scipy.io.wavfile

import librosa.core

def annotation(path, time_start, time_end, annotations=None, delimiter=',', fmt='%0.3f'):
    r'''Save annotations in a 3-column format::

        time_start[0],time_end[0],annotations[0]\n
        time_start[1],time_end[1],annotations[1]\n
        time_start[2],time_end[2],annotations[2]\n
        ...

    This can be used for segment or chord annotations.

    :usage:
        >>> # Detect segment boundaries
        >>> boundaries = librosa.segment.agglomerative(data, k=10)
        >>> # Convert to time
        >>> boundary_times = librosa.frames_to_time(boundaries, sr=sr,
                                                    hop_length=hop_length)
        >>> # Convert boundaries to start-ends
        >>> time_start, time_end = boundaries[:-1], boundaries[1:]
        >>> # Make some fake annotations
        >>> labels = ['Segment #%03d' % i for i in range(len(time_start))]
        >>> # Save the output
        >>> librosa.output.annotation('segments.csv', time_start, time_end,
                                      annotations=annotations)

    :parameters:
      - path : str
        path to save the output CSV file

      - time_start : list-like
        array of starting times for annotations

      - time_end : list-like
        array of ending times for annotations

      - annotations : None or list-like
        optional list of annotation strings. ``annotations[i]`` applies to the time
        range ``time_start[i]`` to ``time_end[i]``

      - delimiter : str
          character to separate fields

      - fmt : str
          format-string for rendering time data

    :raises:
      - ValueError
          if ``time_start`` and ``time_end`` have different length
      - ValueError
          if ``annotations`` is not ``None`` and length does not match ``time_start``
    '''

    if len(time_start) != len(time_end):
        raise ValueError('len(time_start) != len(time_end)')

    if annotations is not None and len(annotations) != len(time_start):
        raise ValueError('len(annotations) != len(time_start)')

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)

        if annotations is None:
            for t_s, t_e in zip(time_start, time_end):
                writer.writerow([fmt % t_s, fmt % t_e])
        else:
            for t_s, t_e, lab in zip(time_start, time_end, annotations):
                writer.writerow([fmt % t_s, fmt % t_e, lab])

def frames_csv(path, frames, sr=22050, hop_length=512, **kwargs):
    """Convert frames to time and store the output in CSV format.

    :usage:
        >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=64)
        >>> librosa.output.frames_csv('beat_times.csv', frames, sr=sr, hop_length=64)

    :parameters:
      - path : string
          path to save the output CSV file

      - frames : list-like of ints
          list of frame numbers for beat events

      - sr : int > 0
          audio sampling rate

      - hop_length : int > 0
          number of samples between success frames

      - kwargs
          additional keyword arguments.  See ``librosa.output.times_csv``
    """

    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    times_csv(path, times, **kwargs)

def times_csv(path, times, annotations=None, delimiter=',', fmt='%0.3f'):
    r"""Save time steps as in CSV format.  This can be used to store the output
    of a beat-tracker or segmentation algorihtm.

    If only ``times`` are provided, the file will contain each value of ``times`` on a row::

        times[0]\n
        times[1]\n
        times[2]\n
        ...

    If ``annotations`` are also provided, the file will contain delimiter-separated values::

        times[0],annotations[0]\n
        times[1],annotations[1]\n
        times[2],annotations[2]\n
        ...


    :usage:
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
          if ``annotations`` is not ``None`` and length does not match ``times``
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

def write_wav(path, y, sr, normalize=True):
    """Output a time series as a .wav file

    :usage:
        >>> # Trim a signal to 5 seconds and save it back
        >>> y, sr = librosa.load('file.wav', duration=5)
        >>> librosa.output.write_wav('file_trim_5s.wav', y, sr)

    :parameters:
      - path : str
          path to save the output wav file

      - y : np.ndarray
          audio time series

      - sr : int > 0
          sampling rate of ``y``

      - normalize : boolean
          enable amplitude normalization
    """

    # normalize
    if normalize:
        wav = y / np.max(np.abs(y))
    else:
        wav = y

    # Scale up to pcm range
    #wav = (wav - wav.min()) * (1<<15) - (1<<15)
    wav = wav * 32767

    # Convert to 16bit int
    wav = wav.astype('<i2')

    # Save
    scipy.io.wavfile.write(path, sr, wav)

