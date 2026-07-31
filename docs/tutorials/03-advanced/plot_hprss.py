# -*- coding: utf-8 -*-
"""
=====================================
Harmonic-percussive source separation
=====================================

This notebook illustrates how to separate an audio signal into
its harmonic and percussive components.

We'll compare the original median-filtering based approach of
`Fitzgerald, 2010 <https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1078&context=argcon>`_ [1]_
and its margin-based extension due to `Driedger, Müller and Disch, 2014 <https://zenodo.org/record/1415226>`_ [2]_.

.. [1] Fitzgerald, D. (2010)
    Harmonic/Percussive Separation using Median Filtering.
    13th International Conference on Digital Audio Effects (DAFX10), Graz, Austria, 2010.

.. [2] Jonathan Driedger, Meinard Müller & Sascha Disch. (2014).
    Extending Harmonic-Percussive Separation of Audio Signals.
    Proceedings of the 15th International Society for Music Information Retrieval Conference, 611--616. https://doi.org/10.5281/zenodo.1415226
"""

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Audio, HTML

import librosa

########################
# Load an example clip with harmonics and percussives
y, sr = librosa.loadx("fishin", duration=5, offset=10)
HTML(librosa.util.example_info("fishin", html=True))

# %%
#

Audio(data=y, rate=sr)

###############################################
# Compute the short-time Fourier transform of y
D = librosa.stft(y)

#####################################################
# Decompose D into harmonic and percussive components
#
# :math:`D = D_\text{harmonic} + D_\text{percussive}`
D_harmonic, D_percussive = librosa.decompose.hpss(D)


####################################################################
# We can plot the two components along with the original spectrogram

# Pre-compute a global reference power from the input spectrum
rp = np.max(np.abs(D))

fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)

imgs = librosa.display.multiplot("specshow", D, D_harmonic, D_percussive,
                                 vscale=f"dB[{rp}]", y_axis="log", x_axis="time",
                                 titles=["Full spectrogram", "Harmonic spectrogram", "Percussive spectrogram"],
                                 axes=ax)
librosa.display.colorbar_db(imgs[0], ax=ax)

#########################################################################
# We can also invert the separated spectrograms to play back the audio.
# First the harmonic signal:

y_harmonic = librosa.istft(D_harmonic, length=len(y))
Audio(data=y_harmonic, rate=sr)

#################################
# And next the percussive signal:

y_percussive = librosa.istft(D_percussive, length=len(y))
Audio(data=y_percussive, rate=sr)


#################################################################################
# The default HPSS above assigns energy to each time-frequency bin according to
# whether a horizontal (harmonic) or vertical (percussive) filter responds higher
# at that position.
#
# This assumes that all energy belongs to either a harmonic or percussive source,
# but does not handle "noise" well.  Noise energy ends up getting spread between
# D_harmonic and D_percussive.  Unfortunately, this often also includes vocals
# and other sounds that are not purely harmonic or percussive.
#
# If we instead require that the horizontal filter responds more than the vertical
# filter *by at least some margin*, and vice versa, then noise can be removed
# from both components.
#
# Note: the default (above) corresponds to margin=1

# Let's compute separations for a few different margins and compare the results below
D_harmonic2, D_percussive2 = librosa.decompose.hpss(D, margin=2)
D_harmonic4, D_percussive4 = librosa.decompose.hpss(D, margin=4)
D_harmonic8, D_percussive8 = librosa.decompose.hpss(D, margin=8)
D_harmonic16, D_percussive16 = librosa.decompose.hpss(D, margin=16)


#############################################################################
# In the plots below, note that vibrato has been suppressed from the harmonic
# components, and vocals have been suppressed in the percussive components.
fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
librosa.display.multiplot("specshow", D_harmonic, D_harmonic2, D_harmonic4, D_harmonic8, D_harmonic16,
                          vscale=f"dB[{rp}]", y_axis="log", x_axis="time", axes=ax[:, 0])
librosa.display.multiplot("specshow", D_percussive, D_percussive2, D_percussive4, D_percussive8, D_percussive16,
                          vscale=f"dB[{rp}]", y_axis="log", x_axis="time", axes=ax[:, 1])
ax[0, 0].set(title="Harmonic")
ax[0, 1].set(title="Percussive")
for i in range(5):
    ax[i, 0].set(ylabel="margin={:d}".format(2**i))


################################################################################
# In the plots above, it looks like margins of 4 or greater are sufficient to
# produce strictly harmonic and percussive components.
#
# We can invert and play those components back just as before.
# Again, starting with the harmonic component:

y_harmonic4 = librosa.istft(D_harmonic4, length=len(y))
Audio(data=y_harmonic4, rate=sr)

##############################################################
# And the percussive component:

y_percussive4 = librosa.istft(D_percussive4, length=len(y))
Audio(data=y_percussive4, rate=sr)
