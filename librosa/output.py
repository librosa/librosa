#!/usr/bin/env python
"""Output routines for audio and analysis"""

import csv

import numpy as np
import scipy
import scipy.io.wavfile

import librosa.core

def segment_csv(path, segments, sr, hop_length, save_bpm=False):
    """Save beat tracker or segmentation output in CSV format.

    :parameters:
      - path : string
          path to save the output CSV file

      - segments : list of ints
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

        time = 0.0
        for t_new in librosa.core.frames_to_time(segments,
                                            sr=sr, hop_length=hop_length):
            if save_bpm:
                writer.writerow([t_new, '%.2f BPM' % (60.0 / (t_new - time))])
            else:
                writer.writerow([t_new])

            time = t_new
#-- --#

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
#-- --#
