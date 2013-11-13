#!/usr/bin/env python
'''
CREATED:2013-02-12 16:33:40 by Brian McFee <brm2132@columbia.edu>

Beat tracking with HPSS filtering

Usage:

./hpss_beats.py input_audio.mp3 output_beats.csv

'''

import sys
import numpy as np

import librosa


SR  = 22050
HOP = 64
N_FFT = 2048

# Load the file
print 'Loading file ... ',
(y, sr) = librosa.load(sys.argv[1], sr=SR)
print 'done.'

# Construct log-amplitude spectrogram
print 'Generating STFT ... ', 
D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)).astype(np.float32)
print 'done.'

# Do HPSS
print 'Harmonic-percussive separation ... ',
(H, P) = librosa.decompose.hpss(D, p=2.0)
print 'done.'

# Construct onset envelope from percussive component
print 'Beat tracking ... ',

S = librosa.feature.melspectrogram(S=P, sr=sr, n_mels=128)

O = librosa.onset.onset_strength(S=librosa.logamplitude(S))

# Track the beats
(bpm, beats) = librosa.beat.beat_track(onsets=O, sr=sr, hop_length=HOP, n_fft=N_FFT)
print 'done.'

# Save the output
librosa.output.segment_csv(sys.argv[2], beats, sr, HOP)
