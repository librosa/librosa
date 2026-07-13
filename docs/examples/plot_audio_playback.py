# coding: utf-8
"""
==============
Audio playback
==============

This notebook demonstrates how to use IPython's audio playback
to play audio signals through your web browser.

"""

# Code source: Brian McFee
# License: ISC

# %%
# We'll need numpy and matplotlib for this example
import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_path = '_static/playback-thumbnail.png'

import librosa

# We'll need IPython.display's Audio widget
from IPython.display import Audio

# We'll also use `mir_eval` to synthesize a signal for us
import mir_eval.sonify

# %%
# Playing a synthetic sound
# -------------------------
# The IPython Audio widget accepts raw numpy data as
# audio signals.  This means we can synthesize signals
# directly and play them back in the browser.
#
# For example, we can make a sine sweep from C3 to C5:

sr = 22050

y_sweep = librosa.chirp(fmin=librosa.note_to_hz('C3'),
                        fmax=librosa.note_to_hz('C5'),
                        sr=sr,
                        duration=1)

Audio(data=y_sweep, rate=sr)

# %%
# Playing a real sound
# --------------------
# Of course, we can also play back real recorded sounds
# in the same way.
#
y, sr = librosa.load(librosa.ex('trumpet'))

Audio(data=y, rate=sr)


# %%
# Sonifying pitch estimates
# -------------------------
# As a slightly more advanced example, we can
# use sonification to directly observe the output of a
# fundamental frequency estimator.
#
# We'll do this using `librosa.pyin` for analysis,
# and `mir_eval.sonify.pitch_contour` for synthesis.

# Using fill_na=None retains the best-guess f0 at unvoiced frames
f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             sr=sr,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'),
                                             fill_na=None)

# To synthesize the f0, we'll need sample times
times = librosa.times_like(f0)

# %%
# mir_eval's synthesizer uses negative f0 values to indicate
# unvoiced regions.
#
# We'll make an array vneg which is 1 for voiced frames, and
# -1 for unvoiced frames.
# This way, `f0 * vneg` will leave voiced estimates unchanged,
# and negate the frequency for unvoiced frames.
vneg = (-1)**(~voiced_flag)

# And sonify the f0 using mir_eval
y_f0 = mir_eval.sonify.pitch_contour(times, f0 * vneg, sr)

Audio(data=y_f0, rate=sr)

# %%
# Sonifying mixtures
# ------------------
# Finally, we can also use the Audio widget to listen
# to combinations of signals.
#
# This example runs the onset detector over the original
# test clip, and then synthesizes a click at each detection.
#
# We can then overlay the click track on the original signal
# and hear them both.
#
# For this to work, we need to ensure that both the synthesized
# click track and the original signal are of the same length.

# Compute the onset strength envelope, using a max filter of 5 frequency bins
# to cut down on false positives
onset_env = librosa.onset.onset_strength(y=y, sr=sr, max_size=5)

# Detect onset times from the strength envelope
onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')

# Sonify onset times as clicks
y_clicks = librosa.clicks(times=onset_times, length=len(y), sr=sr)

Audio(data=y+y_clicks, rate=sr)


# %%
# Caveats
# -------
# Finally, some important things to note when using interactive playback:
#
#   - `IPython.display.Audio` works by serializing the entire audio signal and
#     sending it to the browser in a UUEncoded stream.  This may be inefficient for
#     long signals.
#   - `IPython.display.Audio` can also work directly with filenames and URLs.  If
#     you're working with long signals, or do not want to load the signal into python
#     directly, it may be better to use one of these modes.
#   - Audio playback, by default, will normalize the amplitude of the signal being
#     played.  Most of the time this is what you will want, but sometimes it may not
#     be, so be aware that normalization can be disabled.
#   - If you're working in a Jupyter notebook and want to show multiple Audio widgets
#     in the same cell, you can use
#     ``IPython.display.display(IPython.display.Audio(...))`` to explicitly render
#     each widget.  This is helpful when playing back multiple related signals.
