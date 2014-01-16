#!/usr/bin/env python
"""Output routines for audio and analysis"""

import csv

import numpy as np
import scipy
import scipy.io.wavfile

import librosa.core

def frames_csv(path, frames, sr=22050, hop_length=512, **kwargs):
    """Save beat tracker or segmentation output in CSV format.

    :usage:
        >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=64)
        >>> librosa.output.frames_csv('beat_times.csv', frames, sr=sr, hop_length=64)

    :parameters:
      - path : string
          path to save the output CSV file

      - frames : list-like of ints
          list of frame numbers for beat events
      
      - sr : int
          audio sampling rate
    
      - hop_length : int
          number of samples between success frames

      - kwargs 
          additional keyword arguments.  See ``librosa.output.times_csv``
    """

    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    times_csv(path, times, **kwargs)

def times_csv(path, times, annotations=None, delimiter=',', fmt='%0.3f'):
    """Save time steps as in CSV format.

    :usage:
        >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=64)
        >>> times = librosa.frames_to_time(beats,sr=sr, hop_length=64)
        >>> librosa.output.times_csv('beat_times.csv', times)

    :parameters:
      - path : string
          path to save the output CSV file

      - times : list-like of floats
          list of frame numbers for beat events
      
      - annotations : None or list-like
          optional annotations for each time step

    :raises:
      - ValueError
          if annotations is not None and length does not match `times`
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

def write_wav(path, y, sr):
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

      - sr : int
          sampling rate of ``y``

    """

    # normalize
    wav = y / np.max(np.abs(y))
    
    # Scale up to pcm range
    wav = (wav - wav.min()) * (1<<15) - (1<<15)

    # Convert to 16bit int
    wav = wav.astype('<i2')

    # Save
    scipy.io.wavfile.write(path, sr, wav)
