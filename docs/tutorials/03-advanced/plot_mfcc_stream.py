# coding: utf-8
# License: ISC
"""
===============
MFCC Streaming
===============

This notebook demonstrates how to compute MFCCs incrementally on a stream of
audio, and why the ``top_db`` parameter matters when you do.

`librosa.feature.mfcc` converts the mel spectrogram to decibels with
`librosa.power_to_db`, which by default thresholds the result at ``top_db``
below its peak.  That peak is taken over the entire array it is given,
including the time axis.  When you process a whole file at once, the peak is
the loudest moment in the file.  When you process the same audio block by
block, each block gets its own peak.

The consequence is that MFCCs computed on a stream do not match MFCCs computed
on the whole file, and neither is a function of the individual frame.  Passing
``top_db=None`` disables the threshold and makes the computation frame-local,
which is what you want when the features must not depend on their context.
"""

##################################################
# We'll need numpy and matplotlib for this example
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

import librosa

######################################################################
# We'll use a recording with a wide dynamic range, so that the loud
# passages sit far enough above the quiet ones for the threshold to
# engage.
filename = librosa.ex("humpback")
HTML(librosa.util.example_info("humpback", html=True))

#####################################################################
# Set up the block reader.  ``center=False`` is important here: it keeps
# each block's frames aligned with the frames of the whole-file
# computation, so the two are directly comparable.

n_fft = 2048
hop_length = 512
n_mfcc = 13

sr = librosa.get_samplerate(filename)


def streamed_mfcc(**kwargs):
    """Compute MFCCs block by block and concatenate along time."""
    stream = librosa.stream(
        filename,
        block_length=16,
        frame_length=n_fft,
        hop_length=hop_length,
        mono=True,
        fill_value=0,
    )
    blocks = [
        librosa.feature.mfcc(
            y=y_block,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            center=False,
            **kwargs,
        )
        for y_block in stream
    ]
    return np.hstack(blocks)


#####################################################################
# For comparison, load the whole file and compute the MFCCs in one pass.
y, sr = librosa.load(filename, sr=sr)

whole_default = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, center=False
)
stream_default = streamed_mfcc()

#####################################################################
# With the default ``top_db=80``, the two disagree.  Each block was
# thresholded against its own loudest frame rather than against the
# loudest frame in the file.
n = min(whole_default.shape[1], stream_default.shape[1])
default_error = np.max(np.abs(whole_default[:, :n] - stream_default[:, :n]))
print(f"default top_db=80: max difference = {default_error:.4g}")

#####################################################################
# Passing ``top_db=None`` removes the threshold.  Now every frame is
# computed from its own samples alone, so the streamed result matches
# the whole-file result to floating point precision.
whole_none = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
    center=False, top_db=None,
)
stream_none = streamed_mfcc(top_db=None)

n = min(whole_none.shape[1], stream_none.shape[1])
none_error = np.max(np.abs(whole_none[:, :n] - stream_none[:, :n]))
print(f"top_db=None:       max difference = {none_error:.4g}")

#####################################################################
# Plotting the difference makes the structure visible.  The error is not
# spread evenly: it appears wherever a block's own peak differs from the
# peak of the whole file, so it shows up as sharp bands at the block
# boundaries on either side of a loud event.  Frames inside a block that
# contains a loud passage are thresholded against that passage; the same
# frames in a block without one are not.
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 8))

librosa.display.specshow(
    whole_default[:, :n], x_axis="time", sr=sr, hop_length=hop_length, ax=ax[0]
)
ax[0].set(title="MFCC, whole file, default top_db", ylabel="MFCC")
ax[0].label_outer()

librosa.display.specshow(
    stream_default[:, :n], x_axis="time", sr=sr, hop_length=hop_length, ax=ax[1]
)
ax[1].set(title="MFCC, streamed, default top_db", ylabel="MFCC")
ax[1].label_outer()

img = librosa.display.specshow(
    np.abs(whole_default[:, :n] - stream_default[:, :n]),
    x_axis="time", sr=sr, hop_length=hop_length, ax=ax[2],
)
ax[2].set(title="Absolute difference", ylabel="MFCC")
fig.colorbar(img, ax=ax[2])

#####################################################################
# A note on which setting to use.
#
# The threshold exists to stop the logarithm from amplifying near-silent
# bins into large negative values, and for a single fixed-length input it
# does that well.  It becomes a problem when the same audio has to yield
# the same features in more than one context: streaming versus batch,
# a clip versus the file it came from, or one file versus the same file
# with a loud event appended.
#
# If you are building a feature pipeline whose outputs must be
# comparable across those situations, pass ``top_db=None`` and, if you
# need a floor, apply one you control with `librosa.power_to_db` and an
# explicit ``ref``.
#
# The same parameter is available on `librosa.feature.spectral_contrast`,
# `librosa.onset.onset_strength` and `librosa.onset.onset_strength_multi`,
# which route through `librosa.power_to_db` in the same way.
