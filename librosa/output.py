#!/usr/bin/env python
'''
CREATED:2013-01-23 09:26:25 by Brian McFee <brm2132@columbia.edu>

Utility functions for analysis output, eg:

    - sonic visualizer output for clustering/beat tracking
    - ???

'''

import librosa
import numpy as np
import scipy
import scipy.io.wavfile
import csv

def segment_csv(path, segments, sr, hop_length):
    '''
    Save beat tracker or segmentation output in CSV format

    Input:
        path:               path to the output file
        segments:           1-by-n list of frame numbers for beat events
        sr:                 sample rate of the beat detector (eg 8000)
        hop_length:         hop length of the beat detector (32)
    '''

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file)

        time = 0.0
        for t_new in librosa.frames_to_time(segments,
                                            sr=sr, hop_length=hop_length):
            writer.writerow([t_new, '%.2f BPM' % (60.0 / (t_new - time))])
            time = t_new
#-- --#

def write_wav(path, y, sr):
    '''
    Output a time series as a .wav file

    Input:
        path:       path to save the wav file. 
        y:          time-series audio data (floating point) (mono)
        sr:         sample rate of y

    '''

    wav = y / np.max(np.abs(y))
    scipy.io.wavfile.write(path, sr, (wav * 32768.0).astype('<i2'))
#-- --#
