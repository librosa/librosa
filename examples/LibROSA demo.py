# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# This notebook demonstrates some of the basic functionality of librosa version 0.2.
# 
# Following through this example, you'll learn how to:
# 
# * Load audio input
# * Compute mel spectrogram, MFCC, delta features, chroma
# * Locate beat events
# * Compute beat-synchronous features
# * Display features
# * Save beat tracker output to a CSV file
# 
# In order to run this example, you will need to start IPython with the `--pylab` option, such as:
# 
# ```
#     ipython notebook --pylab line
# ```

# <codecell>

import numpy as np
from matplotlib import pyplot as plt
import librosa

# <codecell>

# For this example, I'll use a track from the SMC beat tracking dataset:
# http://smc.inescporto.pt/research/data-2/

y, sr = librosa.load('/home/bmcfee/data/SMC_Mirex/SMC_MIREX_Audio/SMC_001.wav')

# <markdowncell>

# By default, librosa will resample the signal to 22050Hz.
# You can change this behavior by setting the `sr` parameter of `librosa.loadload()`, or disable resampling entirely by setting `sr=None`.

# <codecell>

# Let's make and display a mel-scaled power (energy-squared) spectrogram
# We use a small hop length of 64 here so that the frames line up with the beat tracker example below.
S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=64, n_mels=128, fmax=8000)

# Normalize by peak power
S = S / S.max()  

# Convert to log scale: instead of power, use dB relative to the peak power.
log_S = librosa.logamplitude(S)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, hop_length=64, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar()

# Make the figure layout compact
plt.tight_layout()

# <codecell>

# How about we get a chromagram as well?
# We'll use a longer FFT window here to better resolve low frequencies
C = librosa.feature.chromagram(y=y, sr=sr, n_fft=4096, hop_length=64)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the chromagram: the energy in each chromatic pitch class as a function of time
# To make sure that the colors span the full range of chroma values, set vmin and vmax
librosa.display.specshow(C, sr=sr, hop_length=64, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

plt.title('Chromagram')
plt.colorbar()

plt.tight_layout()

# <markdowncell>

# In the above examples, Mel power spectrogram is negative-valued, and `specshow()` defaults to a purple->white color gradient.
# The chromagram example is positive-valued, and `specshow()` will default to a white->red color gradient.
# If the input data has both positive and negative values, as in the MFCC example below, then a purple->white->orange diverging color gradient will be used.
# 
# These defaults have been selected to ensure readability in print (grayscale) and are color-blind friendly.
# 
# Just as in `pyplot.imshow()`, the color map can be overriden by setting the `cmap` keyword argument.

# <codecell>

# Next, we'll extract the top 20 Mel-frequency cepstral coefficients (MFCCs)
mfcc        = librosa.feature.mfcc(log_S, n_mfcc=20)

# Let's pad on the first and second deltas while we're at it
delta_mfcc  = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

# How do they look?
plt.figure(figsize=(12, 4))
librosa.display.specshow(M, sr=sr, hop_length=64, x_axis='time')

plt.title('MFCC + $\Delta$ + $\Delta^2$')
plt.colorbar()

plt.tight_layout()

# <codecell>

# Now, let's run the beat tracker
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=64, n_fft=2048)

# Let's re-draw the spectrogram, but this time, overlay the detected beats
plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, hop_length=64, x_axis='time', y_axis='mel')

# Let's draw lines with a drop shadow on the beat events
plt.vlines(beats, 0, log_S.shape[0], colors='k', linestyles='-', linewidth=2.5)
plt.vlines(beats, 0, log_S.shape[0], colors='w', linestyles='-', linewidth=1.5)
plt.axis('tight')

plt.tight_layout()

# <markdowncell>

# By default, the beat tracker will trim away any leading or trailing beats that don't appear strong enough.  
# 
# To disable this behavior, call `beat_track()` with `trim=False`.

# <codecell>

print 'Estimated tempo:      %.2f BPM' % tempo

print 'First 5 beat frames: ', beats[:5]

# Frame numbers are great and all, but when do those beats occur?
print 'First 5 beat times:  ', librosa.frames_to_time(beats[:5], sr=sr, hop_length=64)

# We could also get frame numbers from times by librosa.time_to_frames()

# <codecell>

# Now, let's use the beats to make beat-synchronous timbre features

# feature.sync will summarize each beat event by the mean feature vector within that beat
# this can be useful for reducing dimensionality
M_sync = librosa.feature.sync(M, beats)

plt.figure(figsize=(12,6))

# Let's plot the original and beat-synchronous features against each other
plt.subplot(2,1,1)
librosa.display.specshow(M, sr=sr, hop_length=64, x_axis='time')
plt.title('MFCC-$\Delta$-$\Delta^2$')

# We can also use pyplot *ticks directly
# Let's mark off the raw MFCC and the delta features
plt.yticks(np.arange(0, M_sync.shape[0], 20), ['MFCC', '$\Delta$', '$\Delta^2$'])
plt.colorbar()

plt.subplot(2,1,2)
plt.librosa.display.specshow(M_sync)

# librosa can generate axis ticks from arbitrary timestamps and beat events also
librosa.display.time_ticks(librosa.frames_to_time(beats, sr=sr, hop_length=64))

plt.yticks(np.arange(0, M_sync.shape[0], 20), ['MFCC', '$\Delta$', '$\Delta^2$'])             
plt.title('Beat-synchronous MFCC-$\Delta$-$\Delta^2$')
plt.colorbar()

plt.tight_layout()

# <codecell>

# Beat synchronization is flexible.
# Instead of computing the mean delta-MFCC within each beat, let's do beat-synchronous chroma
# We can replace the mean with any statistical aggregation function, such as min, max, or median.

C_sync = librosa.feature.sync(C, beats, aggregate=np.median)

figure(figsize=(12,6))

subplot(2, 1, 1)
librosa.display.specshow(C, sr=sr, hop_length=64, y_axis='chroma', vmin=0.0, vmax=1.0)
title('Chroma')
colorbar()

subplot(2, 1, 2)
librosa.display.specshow(C_sync, y_axis='chroma', vmin=0.0, vmax=1.0)

librosa.display.time_ticks(librosa.frames_to_time(beats, sr=sr, hop_length=64))

title('Beat-synchronous Chroma (median aggregation)')

colorbar()
tight_layout()

# <codecell>

# Let's save the beat times off as a CSV file
librosa.output.frames_csv('beat_times.csv', beats, sr=sr, hop_length=64)

