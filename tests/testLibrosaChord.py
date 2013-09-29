# Compute chroma, predict chords

# Libraries
import matplotlib.pylab as plt

import os, sys

import numpy as np

# Path for audio read
import sys
if "/Users/mattmcvicar/Desktop/Work/audioread" not in sys.path:
  sys.path.append("/Users/mattmcvicar/Desktop/Work/audioread")

# Path for libROSA
if "/Users/mattmcvicar/Desktop/Work/libROSA_chorddev/librosa" not in sys.path:
  sys.path.append("/Users/mattmcvicar/Desktop/Work/libROSA_chorddev/librosa")
import librosa

# Path for HPA
if "/Users/mattmcvicar/Desktop/Work/libROSA_chorddev/librosa/HPA" not in sys.path:
  sys.path.append("/Users/mattmcvicar/Desktop/Work/libROSA_chorddev/librosa/HPA")
import HPA_extract_chroma as HPA_chroma

# Audio folder
audio_folder = "/Users/mattmcvicar/Desktop/Work/onsets_draft"

# Audio example
song = "titanium.wav"

# Load audio
x_all, sr = librosa.load(os.path.join(audio_folder,song))
x = x_all[:sr*30]

# STFT and chroma
reload(librosa)
fftlen = 1024
hoplen = fftlen/2.0

X = librosa.core.stft(x, n_fft=fftlen, hop_length=hoplen)

Chroma_Ellis = librosa.feature.chromagram(x, sr, method='Ellis', norm=None)
Chroma_McV_HPA = HPA_chroma.extract_chroma(os.path.join(audio_folder,song))

# Now integrated. First get beats
[bpm,beats] = librosa.beat.beat_track(x, sr=sr, onsets=None, hop_length=hoplen, n_fft=fftlen)

# Convert beats to seconds
beats_s = librosa.frames_to_time(beats, sr, hoplen)

Chroma_McV = librosa.feature.chromagram(x, sr, method='McVicar', norm=None, beat_times=beats_s)

# Normalise the other chroma and beatsynch
# Ellis
Ellis_BS = librosa.feature.sync(Chroma_Ellis, beats)

# Normalise
Ellis_BS_Norm = np.zeros(Ellis_BS.shape)
for t in range(Ellis_BS_Norm.shape[1]):
  maxval = np.max(Ellis_BS[:,t])
  Ellis_BS_Norm[:,t] = Ellis_BS[:,t] / maxval


subplot(311)
plt.imshow(Ellis_BS_Norm,aspect='auto',interpolation='None')
plt.title('Ellis Method')
plt.show()

subplot(312)
plt.imshow(Chroma_McV_HPA[1],aspect='auto',interpolation='None')
plt.title('McVicar HPA')
plt.show()

subplot(313)
plt.imshow(Chroma_McV,aspect='auto',interpolation='None')
plt.title('McVicar Integrated')
plt.show()