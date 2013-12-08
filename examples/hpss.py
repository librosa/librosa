#!/usr/bin/env python
'''
CREATED:2013-12-08 14:28:34 by Brian McFee <brm2132@columbia.edu>
 
Demonstration of harmonic-percussive source separation

Usage:

./hpss.py  input_file.mp3  output_harmonic.wav  output_percussive.wav

'''

import sys, librosa
N_FFT = 2048
HOP   = N_FFT /4

# 1. Load the wav file, resample
print 'Loading ', sys.argv[1], '... ',

y, sr = librosa.load(sys.argv[1])
print 'done.'

# 2. generate STFT @ 2048 samples
print 'Computing short-time fourier transform... ',
D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
print 'done.'

# 3. HPSS.  The default kernel size isn't necessarily optimal, but works okay enough
print 'Separating harmonics and percussives... ',
H, P = librosa.decompose.hpss(D)
print 'done.'

# 4. Invert STFT
print 'Inverting harmonics and percussives... ',
y_h = librosa.istft(H, hop_length=HOP)
y_p = librosa.istft(P, hop_length=HOP)
print 'done.'

# 5. Save the results
print 'Saving results to: ', sys.argv[2:]
librosa.output.write_wav(sys.argv[2], y_h, sr)
librosa.output.write_wav(sys.argv[3], y_p, sr)

