# coding: utf-8
"""
===============
Onset detection
===============

This section introduces the problem of detecting musical onsets, and the
underlying methods for computing an onset strength envelope.
Beyond the direct application of onset detection, the methods introduced
here form the basis for rhythm analysis generally, which will be covered
in the next section.
"""

# %%
# .. _tutorial-intro-onsets:
#
# Onsets
# ------
# As a starting point, we'll first look at the problem of identifying when each
# musical event (e.g., a note sounding) occurs in the input signal.
# We will focus on identifying the beginning of the note events, known as *onsets*,
# as this is often the first stage of processing for more sophisticated analyses
# such as tempo estimation or beat tracking, which we will see below.
#
# First, we will load in a simple monophonic example recording, and display its waveform
# and spectrogram.

# sphinx_gallery_thumbnail_number = 4

import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, HTML

# Load the audio
y, sr = librosa.loadx("trumpet")
HTML(librosa.util.example_info("trumpet", html=True))

# %%
# Compute the STFT
S = librosa.stft(y)

# Generate a plot of waveform and spectrogram
fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=(1, 4))
librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label="Waveform")
img = librosa.display.specshow(S, vscale="dBFS", x_axis="time", y_axis="log", sr=sr)
librosa.display.colorbar_db(img, label="dBFS")
ax[0].label_outer()
ax[0].legend()

# And an audio playback object so we can listen to it
Audio(url=librosa.example("trumpet", url=True))

# %%
# Try listening to the example above while following the visual display.
# With some practice, you should be able to draw a correspondence between
# when each note is sounding, visually detectable changes in the spectrogram
# and waveform.
#
# The spans of time where the acoustic content is stable, e.g., the final
# sustained F note beginning at around 2.5 seconds, there are no abrupt
# changes in the spectrogram from one vertical slice to the next.
# Put another way, there are sustained horizontal patterns which correspond
# to sustained tones.
#
# When we are looking for new note onsets, we are interested in exactly the
# cases where the spectrogram is *not* constant in time.
# There are many ways to formalize this idea, but a core principle is the idea
# of detecting change or *novelty* by comparing each time step of a spectrogram
# to the preceding time, i.e., `S[:, t]  - S[:, t-1]`.
#
# Typically we are not so much interested in the complex spectrogram values
# as the magnitudes, which are proportional to energy (as displayed above).
# Additionally, working in log magnitudes (or decibels) provides some robustness
# to overall changes in loudness.
#
# Finally, we are generally more interested only in places where energy is
# *increasing*, since decreasing energy does not generally indicate the onset
# of a new event.
#
# We can put these ideas together as follows.

# Map the magnitude abs(S) to decibels
logS = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# Compute the first-order difference logS[:, t] - logS[:, t-1]
# along the time direction.
# We'll pad the differencing operation with the first column of
# logS to prevent a spike in the first step.
# This also ensures that the output has the same number of frames as the
# input (`logS`), since the differencing operation would otherwise
# discard the first frame.
diffS = np.diff(logS, axis=-1, prepend=logS[:, :1])

# We'll threshold out any negative values as these correspond to
# falling energy
diffS_thresh = np.maximum(diffS, 0)

# Visualize the results
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
i1 = librosa.display.specshow(S, vscale="dBFS", x_axis="time", y_axis="log", ax=ax[0], sr=sr)
i2 = librosa.display.specshow(diffS, x_axis="time", y_axis="log", ax=ax[1], sr=sr)
i3 = librosa.display.specshow(diffS_thresh, x_axis="time", y_axis="log", ax=ax[2], sr=sr, norm=i2.norm, cmap=i2.cmap)

librosa.display.colorbar_db(i1, label="dBFS")
librosa.display.colorbar_db(i2, label="Δ dB")
librosa.display.colorbar_db(i3, label="Δ dB")
ax[0].label_outer()
ax[1].label_outer()
ax[0].set(ylabel="STFT")
ax[1].set(ylabel="Difference")
ax[2].set(ylabel="Thresholded diff")


# %%
# In the middle plot above, we can see that constant regions map to a neutral
# color (gray), while regions where the difference (often denoted by delta, Δ)
# is positive (increasing energy) are encoded in red, while negative differences
# (decreasing energy) are encoded in blue.
#
# The bottom plot discards the negative regions, retaining only time-frequency
# positions where energy is increasing.
#
# Finally, we are usually not interested in changes at each individual frequency,
# but rather the aggregated change across all frequencies at each time.
# A simple way to aggregate is by averaging across frequencies, resulting in
# what is usually called an *onset strength envelope* or a *novelty curve*.
#

# Average across frequencies to get the onset strength envelope
onset_env = np.mean(diffS_thresh, axis=0)

# Plot the waveform, spectrogram, and onset envelope together

fig, ax = plt.subplots(nrows=3, sharex=True, height_ratios=(1, 1, 4))
librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label="Waveform")
img = librosa.display.specshow(S, vscale="dBFS", x_axis="time", y_axis="log", ax=ax[2], sr=sr)
librosa.display.colorbar_db(img, label="dBFS")
times = librosa.times_like(onset_env, sr=sr)
ax[1].plot(times, onset_env, label="Onset envelope", color="C1")
ax[1].legend()
ax[0].legend()
ax[0].label_outer()
ax[1].label_outer()

# %%
# So far, we have computed the onset strength envelope manually.
# Because this is such a common operation, librosa provides `librosa.onset.onset_strength`,
# along with several variations on the same idea.

# Equivalent to the above
onset_env = librosa.onset.onset_strength(S=logS)

# %%
# It's worth emphasizing here that the onset strength envelope does not make decisions about
# whether or not a new event occurs at each time.
# Rather, it should be taken as a soft representation that something new *might* be happening.
# Still, onset strength envelopes can be useful objects on their own, and often form an intermediate
# signal representation that is passed into a subsequent stage of processing.

# %%
# Onset detection
# ---------------
# To actually detect onsets, we need to make a decision about
# whether each frame in the onset strength envelope contains a new event or not.
#
# A simple heuristic is to simply take the peak positions of the onset strength envelope,
# i.e., local maxima where `o[t] > o[t-1]` and `o[t] > o[t+1]`.
# This codifies the intuition that peaks of the onset envelope correspond to
# the steepest increase of energy, and should therefore align with the perception
# of a new event.
# This can be implemented simply using the `librosa.util.localmax` utility function:

onset_peaks = librosa.util.localmax(onset_env)

fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=(3, 1))

librosa.display.waveshow(y=y, sr=sr, ax=ax[1], label="Waveform")
ax[1].legend()
ax[0].plot(times, onset_env, label="Onset envelope", color="C1")
ax[0].scatter(times[onset_peaks], onset_env[onset_peaks], marker="^", color="k", label="Peaks")
ax[0].legend()
ax[0].label_outer()

# %%
# A naive approach is to treat local maxima of the onset envelope as onsets.
# In practice, this is too sensitive: it ignores both the overall magnitude of
# the envelope and the fact that physically plausible onsets should be separated in time.
# As illustrated above, this leads to a highly sensitive
# detector that produces far more events than actually occur in the signal.
#
# Librosa therefore implements a heuristic peak-picking algorithm which seeks
# to select peaks which are sufficiently separated and sufficiently high in
# value relative to their surroundings.

onset_detect = librosa.onset.onset_detect(onset_envelope=onset_env)

fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=(3, 1))

librosa.display.waveshow(y=y, sr=sr, ax=ax[1], label="Waveform")
ax[1].legend()
ax[0].plot(times, onset_env, label="Onset envelope", color="C1")
ax[0].scatter(times[onset_peaks], onset_env[onset_peaks], marker="^", color="k", label="Localmax Peaks")
ax[0].scatter(times[onset_detect], onset_env[onset_detect], marker="o",
              edgecolor="C2", facecolor="none", label="onset_detect")
ax[0].legend()
ax[0].label_outer()

# %%
# .. note:: `librosa.onset.onset_detect` does not necessarily select *peaks* of the
#  onset envelope, and may favor earlier points on the rising edge of a peak.
#  The underlying `librosa.util.peak_pick` function also has several parameters that
#  can be adjusted to control this behavior.

# %%
# We can also sonify these detected events to hear how they align with the onset of new notes.

# Generate a click track from the detected frames
# and match the length to the original input signal
clicks = librosa.clicks(frames=onset_detect, length=len(y), sr=sr)

# Sonify the result
Audio(data=y + clicks, rate=sr)

# %%
# Output units
# ------------
# By default, the `onset_detect` function returns the frame indices of
# the detected onsets.  These can be converted to time values using the
# `frames_to_time`:

onset_times = librosa.frames_to_time(onset_detect, sr=sr)

# %%
# (Note that we are implicitly using the default hop length of 512
# samples per frame here, which can be overridden by setting `hop_length`
# in the `onset_detect` and `frames_to_time` functions.)
#
# This is a common enough use case that the `onset_detect` function can
# also return the times directly, by setting the `units` argument:

onset_times = librosa.onset.onset_detect(onset_envelope=onset_env,
                                         units="time")

print(onset_times)

# %%
# The `librosa.clicks` function can also accept time values directly by
# specifying the `times` argument instead of `frames` like above:

clicks = librosa.clicks(times=onset_times, length=len(y), sr=sr)

# %%
# In addition to `frames` and `times`, these functions also can work in
# units of `samples`, though this is less commonly used than the other
# two modes.
#
# .. tip:: As a general rule, we recommend using `frames` (or `samples)`
#   for intermediate processing.  This is because `frames` and `samples`
#   are represented as integers and are exact. `times` are represented as
#   floating point numbers, and may be subject to rounding errors.
#   This is usually not a problem if `times` are the final result of the
#   analysis, but rounding errors can accumulate if the outputs are
#   subjected to further processing.

# %%
# Summary
# -------
# This section introduced the concept of onset envelopes, and methods for
# detecting and working with onsets.
# As noted above, this kind of analysis often forms the first stage of
# processing for tempo estimation, beat tracking, and rhythm analysis.
#
# If you are interested in learning more about onset detection and
# novelty functions, we recommend the following tutorial article:
#
# - *A Basic Tutorial on Novelty and Activation Functions for Music
#   Signal Processing* (Müller and Chiu, 2024)
#   https://transactions.ismir.net/articles/10.5334/tismir.202
