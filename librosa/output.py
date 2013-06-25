#!/usr/bin/env python
"""Utility functions for analysis output, eg:

  - sonic visualizer output for clustering/beat tracking
  - wav file output

CREATED:2013-01-23 09:26:25 by Brian McFee <brm2132@columbia.edu>
"""

import csv

import numpy as np
import scipy
import scipy.io.wavfile

import librosa

def segment_csv(path, segments, sr, hop_length, save_bpm=False):
    """Save beat tracker or segmentation output in CSV format

    Arguments:
      path          -- (string)  path to the output CSV file
      segments      -- (list)    list of frame numbers for beat events
      sr            -- (int)     sample rate
      hop_length    -- (int)     hop length
      save_bpm      -- (boolean) add a BPM annotation column

    """

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file)

        time = 0.0
        for t_new in librosa.frames_to_time(segments,
                                            sr=sr, hop_length=hop_length):
            if save_bpm:
                writer.writerow([t_new, '%.2f BPM' % (60.0 / (t_new - time))])
            else:
                writer.writerow([t_new])

            time = t_new
#-- --#

def write_wav(path, y, sr):
    """Output a time series as a .wav file

    Arguments:
      path  -- (string)     path to output the wav file
      y     -- (ndarray)    time-series audio data (float)
      sr    -- (int)        sample rate

    """

    wav = y / np.max(np.abs(y))
    scipy.io.wavfile.write(path, sr, (wav * 32768.0).astype('<i2'))
#-- --#
