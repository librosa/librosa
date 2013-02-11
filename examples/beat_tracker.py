#!/usr/bin/env python
'''
CREATED:2013-02-11 18:37:30 by Brian McFee <brm2132@columbia.edu>

Do beat tracking on an audio file

Usage:

./beat_tracker.py   input_file.mp3    output_beats.csv
'''

import sys, librosa

# 1. load the wav file
(y, sr) = librosa.load(sys.argv[1])

# 2. extract beats
hop_length = 64
(bpm, beats) = librosa.beat.beat_track(y, sr, hop_length=hop_length)

# 3. save output
librosa.output.beat_csv(sys.argv[2], beats, sr, hop_length)
