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
            t_segment = librosa.beat.frames_to_time(segment, sr=sr, hop_length=hop_length)
            CW.writerow([t, '%d frames' % segment])
            t += t_segment
            pass
        pass
    pass
