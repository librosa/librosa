#!/usr/bin/env python
"""Output routines for audio and analysis"""

import csv

import numpy as np
import scipy
import scipy.io.wavfile

import librosa.core

def frames_csv(path, frames, sr=22050, hop_length=64):
    """Save beat tracker or segmentation output in CSV format.

    :parameters:
      - path : string
          path to save the output CSV file

      - frames : list of ints
          list of frame numbers for beat events
      
      - sr : int
          audio sample rate
    
      - hop_length : int
          hop length
      
      - save_bpm : boolean
          add a BPM annotation column

    """

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file)

        for t_new in librosa.core.frames_to_time(frames,
                                            sr=sr, hop_length=hop_length):
            writer.writerow([t_new])

def write_wav(path, y, sr):
    """Output a time series as a .wav file

    :parameters:
      - path : str 
          path to save the output wav file

      - y : np.ndarray    
          time-series audio data

      - sr : int
          audio sample rate

    """

    wav = y / np.max(np.abs(y))
    scipy.io.wavfile.write(path, sr, (wav * 32768.0).astype('<i2'))
