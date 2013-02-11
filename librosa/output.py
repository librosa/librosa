#!/usr/bin/env python
'''
CREATED:2013-01-23 09:26:25 by Brian McFee <brm2132@columbia.edu>

Utility functions for analysis output, eg:

    - sonic visualizer output for clustering/beat tracking
    - ???

'''

import librosa
import numpy, scipy
import csv

def segment_csv(outfile, N, sr, hop_length):
    '''
    Save a segmentation output in CSV format

    Input:
        outfile:            path to the output file
        N:                  1-by-n list of frame counts
        sr:                 sample rate of the analyzer (eg 11025)
        hop_length:         hop length (in frames) of the analyzer (eg 32)
    '''

    with open(outfile, 'w') as f:
        t = 0.0
        CW = csv.writer(f)

        for segment in N:
            t_segment = librosa.frames_to_time(segment, sr=sr, hop_length=hop_length)
            CW.writerow([t, '%d frames' % segment])
            t += t_segment
            pass
        pass
    pass

def beat_csv(outfile, beats, sr, hop_length):
    '''
    Save beat tracker output in CSV format

    Input:
        outfile:            path to the output file
        beats:              1-by-n list of frame numbers for beat events
        sr:                 sample rate of the beat detector (eg 8000)
        hop_length:         hop length of the beat detector (32)
    '''

    with open(outfile, 'w') as f:
        CW = csv.writer(f)

        t = 0.0
        for t_new in librosa.frames_to_time(beats, sr=sr, hop_length=hop_length):
            CW.writerow([t_new, '%.2f BPM' % (60.0 / (t_new - t))])
            t = t_new
            pass
        pass
    pass
