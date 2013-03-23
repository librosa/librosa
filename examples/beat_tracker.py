#!/usr/bin/env python
'''
CREATED:2013-02-11 18:37:30 by Brian McFee <brm2132@columbia.edu>

Do beat tracking on an audio file

Usage:

./beat_tracker.py   input_file.mp3    output_beats.csv
'''

import sys, librosa

# 1. load the wav file and resample to 22.050 KHz
print 'Loading ', sys.argv[1], '... ',
(y, sr)         = librosa.load(sys.argv[1], target_sr=22050)
print 'done.'

# 2. extract beats

# Use a default hop size of 64 frames ~= 11.6ms
hop_length = 64
print 'Extracting beats... ',
(bpm, beats)    = librosa.beat.beat_track(y, sr, hop_length=hop_length)
print 'done.    Estimated bpm: %.2f' % bpm

# 3. save output
print 'Saving output... ',
librosa.output.segment_csv(sys.argv[2], beats, sr, hop_length)
print 'done.'
