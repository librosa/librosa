#!/usr/bin/env python
'''
CREATED:2013-01-23 09:26:25 by Brian McFee <brm2132@columbia.edu>

Utility functions for analysis output, eg:

    - sonic visualizer output for clustering/beat tracking
    - ???

'''

import librosa
import numpy, scipy
import scipy.io.wavfile
import csv

def segment_csv(outfile, segments, sr, hop_length):
    '''
    Save beat tracker or segmentation output in CSV format

    Input:
        outfile:            path to the output file
        segments:           1-by-n list of frame numbers for beat events
        sr:                 sample rate of the beat detector (eg 8000)
        hop_length:         hop length of the beat detector (32)
    '''

    with open(outfile, 'w') as f:
        CW = csv.writer(f)

        t = 0.0
        for t_new in librosa.frames_to_time(segments, sr=sr, hop_length=hop_length):
            CW.writerow([t_new, '%.2f BPM' % (60.0 / (t_new - t))])
            t = t_new
            pass
        pass
    pass


def write_wav(y, sr, filename):
    '''
    Output a time series as a .wav file

    Input:
        y:          time-series audio data (floating point) (mono)
        sr:         sample rate of y
        filename:   path to save the wav file. 

    '''

    w = y / numpy.max(numpy.abs(y))
    scipy.io.wavfile.write(filename, sr, (w * 32768.0).astype('<i2'))
    pass
