# -*- coding: utf-8 -*-
"""
================
Rainbowgrams
================
This notebook demonstrates how to use "Rainbowgrams" to simultaneously 
visualize amplitude and (unwrapped) phase (differential).
Our working example will be the problem of silence/non-silence detection.
"""
# %%
# Code source: Brian McFee
# License: ISC
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

#############################################
# %% 
# Construsct a sine-sweep signal.
sr = 22050
y = librosa.chirp(fmin=32, fmax=32 * 2**5, sr=sr, duration=10, linear=True)
D = librosa.stft(y)
mag, phase = librosa.magphase(D)

freqs = librosa.fft_frequencies()
times = librosa.times_like(D)

phase_exp = 2*np.pi*np.multiply.outer(freqs,times)

# %%
plt.close('all')
fig, ax = plt.subplots()

img = librosa.display.specshow(np.diff(np.unwrap(np.angle(phase), axis=1), axis=1, prepend=0),
                         cmap='hsv', 
                         alpha=librosa.amplitude_to_db(mag, ref=np.max)/80 + 1,
                         ax=ax,
                         y_axis='log', 
                         x_axis='time')
ax.set_facecolor('#000')

cbar = fig.colorbar(img, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.ax.set(yticklabels=['-π', '-π/2', 0, 'π/2', 'π']);

# %%
fig, ax = plt.subplots()
img = librosa.display.specshow(np.diff(np.unwrap(np.angle(phase), axis=1), axis=1, prepend=0),
                         cmap='twilight', 
                         alpha=librosa.amplitude_to_db(mag, ref=np.max)/80 + 1,
                         ax=ax,
                         y_axis='log', 
                         x_axis='time')
ax.set_facecolor('#888')

# %%
