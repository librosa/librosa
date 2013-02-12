#!/usr/bin/env python
'''
CREATED:2013-02-12 16:33:40 by Brian McFee <brm2132@columbia.edu>

Beat tracking with HPSS filtering

Usage:

./hpss_beats.py input_audio.mp3 output_beats.csv

'''

import sys
import librosa

SR  = 8000
hop = 128
FFT = 1024

# Load the file
print 'Loading file ... ',
(y, sr) = librosa.load(sys.argv[1], target_sr=SR)
print 'done.'

# Construct log-amplitude spectrogram
print 'Harmonic-percussive separation ... ',
S = librosa.logamplitude(librosa.melspectrogram(y, sr, FFT, hop_length=hop, mel_channels=128))

# Do HPSS
(H, P) = librosa.hpss.hpss_median(S, p=2.0)
print 'done.'

# Construct onset envelope from percussive component
print 'Beat tracking ... ',
O = librosa.beat.onset_strength(y, sr, window_length=FFT, hop_length=hop, S=P)

# Track the beats
(bpm, beats) = librosa.beat.beat_track(y, sr, hop_length=hop, onsets=O)
print 'done.'

# Save the output
librosa.output.segment_csv(sys.argv[2], beats, sr, hop)
