# coding: utf-8
"""
===============================================
Music Synchronization with Dynamic Time Warping
===============================================

In this short tutorial, we demonstrate the use of dynamic time warping (DTW) for music synchronization
which is implemented in `librosa`.

We assume that you are familiar with the algorithm and focus on the application. Further information about
the algorithm can be found in the literature, e. g. [1].

Our example consists of two recordings of the first bars of the famous
brass section lick in Stevie Wonder's rendition of "Sir Duke".
Due to differences in tempo, the first recording lasts for ca. 7 seconds and the second recording for ca. 5 seconds.
Our objective is now to find an alignment between these two recordings by using DTW.

"""

# Code source: Stefan Balke
# License: ISC
# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display


############################################################
# ---------------------
# Load Audio Recordings
# ---------------------
# First, let's load a first version of our audio recordings.
x_1, fs = librosa.load('audio/sir_duke_slow.mp3')
# And a second version, slightly faster.
x_2, fs = librosa.load('audio/sir_duke_fast.mp3')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
librosa.display.waveplot(x_1, sr=fs, ax=ax[0])
ax[0].set(title='Slower Version $X_1$')
ax[0].label_outer()

librosa.display.waveplot(x_2, sr=fs, ax=ax[1])
ax[1].set(title='Faster Version $X_2$')

#########################
# -----------------------
# Extract Chroma Features
# -----------------------
hop_length = 1024

x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs,
                                         hop_length=hop_length)
x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs,
                                         hop_length=hop_length)

fig, ax = plt.subplots(nrows=2, sharey=True)
img = librosa.display.specshow(x_1_chroma, x_axis='time',
                               y_axis='chroma',
                               hop_length=hop_length, ax=ax[0])
ax[0].set(title='Chroma Representation of $X_1$')
librosa.display.specshow(x_2_chroma, x_axis='time',
                         y_axis='chroma',
                         hop_length=hop_length, ax=ax[1])
ax[1].set(title='Chroma Representation of $X_2$')
fig.colorbar(img, ax=ax)


########################
# ----------------------
# Align Chroma Sequences
# ----------------------
D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=fs,
                               cmap='gray_r', hop_length=hop_length, ax=ax)
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
ax.set(title='Warping Path on Acc. Cost Matrix $D$',
       xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
fig.colorbar(img, ax=ax)

##############################################
# --------------------------------------------
# Alternative Visualization in the Time Domain
# --------------------------------------------
#
# We can also visualize the warping path directly on our time domain signals.
# Red lines connect corresponding time positions in the input signals.
# (Thanks to F. Zalkow for the nice visualization.)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True)

# Plot x_1
librosa.display.waveplot(x_1, sr=fs, ax=ax1)
ax1.set(title='Slower Version $X_1$')

# Plot x_2
librosa.display.waveplot(x_2, sr=fs, ax=ax2)
ax2.set(title='Slower Version $X_2$')


trans_figure = fig.transFigure.inverted()
lines = []
arrows = 30
points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))

# for tp1, tp2 in zip((wp[points_idx, 0]) * hop_length, (wp[points_idx, 1]) * hop_length):
for tp1, tp2 in wp_s[points_idx]:
    # get position on axis for a given index-pair
    coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
    coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

    # draw a line
    line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                   (coord1[1], coord2[1]),
                                   transform=fig.transFigure,
                                   color='r')
    lines.append(line)

fig.lines = lines

###########################################################
# -------------
# Next steps...
# -------------
#
# Alright, you might ask where to go from here.
# Once we have the warping path between our two signals,
# we could realize different applications.
# One example is a player which enables you to navigate between
# different recordings of the same piece of music,
# e.g. one of Wagner's symphonies played by an orchestra or in a piano-reduced version.
#
# Another example is that you could apply time scale modification algorithms,
# e.g. speed up the slower signal to the tempo of the faster one.
#
# ----------
# Literature
# ----------
#
# [1] Meinard Müller, Fundamentals of Music Processing — Audio, Analysis, Algorithms, Applications.
# Springer Verlag, 2015.
