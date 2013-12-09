#!/usr/bin/env python
'''
CREATED:2013-12-09 00:02:54 by Brian McFee <brm2132@columbia.edu>
 
Estimate the tuning (deviation from A440) of a recording.

Usage:

./tuning.py input_file
'''

import sys, librosa
import numpy as np

N_FFT       = 2048
HOP_LENGTH  = N_FFT / 4

print 'Loading ', sys.argv[1], '... ',
y, sr = librosa.load(sys.argv[1])
print 'done.'

# Get the ifptrack
print 'Tracking pitches... ',
pitches, magnitudes, D = librosa.feature.ifptrack(  y, sr, 
                                                    n_fft=N_FFT, 
                                                    hop_length=HOP_LENGTH)
print 'done.'

# Just track the pitches associated with high magnitude
print 'Estimating tuning ... ',
pitches = pitches[magnitudes > np.median(magnitudes)]

tuning = librosa.feature.estimate_tuning(pitches)
print 'done.'

print sys.argv[1], ': %+0.2f cents' % (100 * tuning)
   
