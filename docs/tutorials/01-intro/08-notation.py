# coding: utf-8
"""
==============
Music notation
==============

This section introduces tools for working with various music notational systems, and
converting between them.
"""
# %%
# .. _tutorial-intro-notation:
#
# Pitches, frequency, and MIDI numbers
# ------------------------------------
# Earlier (:ref:`tutorial-f0`), we estimated fundamental frequency in Hz.
# In this section, we look more carefully at how librosa converts between frequency, pitch notation, and MIDI.
#
# It helps to first clarify some definitions:
#
# - **Frequency** is a physical property, measured in Hertz (cycles per
#   second, abbreviated Hz).  This is something that we can observe directly in a
#   recorded signal.
#
# - **Pitch** is a perceptual concept relating frequency to what we understand as
#   musical tones.  In librosa, we adopt the `Scientific Pitch Notation (SPN)
#   <https://en.wikipedia.org/wiki/Scientific_pitch_notation>`_ standard for representing pitch in
#   Western notation (*C*, *C♯*, *D♭*, etc.).  SPN represents pitches as a combination of note name (*C*,
#   *D*, *E*, etc.), accidentals (*♯*, *♭*, *♮*, etc.), and an octave number.  Librosa extends this
#   slightly to additionally encode cent deviations from the underlying equal temperament grid.
#
#   .. admonition:: Example
#
#       15¢ above middle C can be represented as *C4+15*, which is equivalent to 263.902 Hz.
#       The same frequency could be represented equivalently as either *C♯4-85* or *D♭4-85*.
#
#       SPN assumes 12-tone equal temperament (12TET) and A440 tuning.
#
# - **MIDI** (Musical Instrument Digital Interface) standard assigns integer values 0-127 to pitches
#   following the conventions described above (12TET, A440).
#   A MIDI note number *n* can be converted to a frequency via the equation
#   :math:`f = 440 · 2^{(n-69)/12}`.  Librosa also supports fractional MIDI note
#   numbers, which allow for conversion of frequencies between those on the 12TET grid.
#
# The table below illustrates the relationships between the three systems described above.
#
# +------+-------------+----------------+
# | MIDI | Pitch (SPN) | Frequency (Hz) |
# +======+=============+================+
# | 0    | C-1         | 8.1758         |
# +------+-------------+----------------+
# | 1    | C♯-1        | 8.6619         |
# +------+-------------+----------------+
# | 2    | D-1         | 9.1770         |
# +------+-------------+----------------+
# | ...  | ...         | ...            |
# +------+-------------+----------------+
# | 60   | C4          | 261.626        |
# +------+-------------+----------------+
# | 61   | C♯4         | 277.183        |
# +------+-------------+----------------+
# | 62   | D4          | 293.665        |
# +------+-------------+----------------+
# | ...  | ...         | ...            |
# +------+-------------+----------------+
# | 127  | G9          | 12543.854      |
# +------+-------------+----------------+
#
# Librosa provides functions to convert between any pair of these representations, as
# illustrated in the example code below.
import numpy as np
import librosa
# sphinx_gallery_thumbnail_path = '_static/bass_clef.png'

# Generate one octave of MIDI notes, starting at middle C (MIDI 60):
midi = np.arange(60, 72)
print(midi)

# %%
# We can then convert these MIDI note numbers to frequencies in Hz:
frequencies = librosa.midi_to_hz(midi)
print(frequencies)

# %%
# ... or to pitches in SPN:
pitches = librosa.midi_to_note(midi)
print(pitches)

# %%
# The conversion can be done in the other direction as well:
midi_from_pitch = librosa.note_to_midi(pitches)
print(midi_from_pitch)

# %%
# And we can short-cut directly between pitch and frequency:
freq_from_pitch = librosa.note_to_hz(pitches)
print(freq_from_pitch)

# %%
#

# %%
# Keys and degrees
# ----------------
# By default, note spelling follows the conventions of `C:major`.
# So MIDI note 61 is written as *C♯4* rather than *D♭4*.
# If you know the musical key, you can often choose a more appropriate spelling.
# To do this, we can supply a `key` argument to any conversion function which outputs notes:
pitches_f = librosa.midi_to_note(midi, key="F:major")
print(pitches_f)

# %%
# For any key, including modes like *D:dorian*, or *G♯:mixolydian*, the spelling for each pitch class can
# be obtained by the `librosa.key_to_notes` function:

print("F:major       ⇒ ", librosa.key_to_notes("F:major"))
print("D:dorian      ⇒ ", librosa.key_to_notes("D:dorian"))
print("G♯:mixolydian ⇒ ", librosa.key_to_notes("G♯:mixolydian"))

# %%
# Note that these will always be ordered according to the chromatic scale, so the first note corresponds
# to pitch class 0 (*C*), the second to pitch class 1 (*C♯* or *D♭*), and so on, regardless of the key.

# %%
# It can also be helpful sometimes to compute a list of pitch classes belonging to a particular key or
# mode.  This can be done with the `librosa.key_to_degrees` function, which outputs a list of pitch class
# numbers corresponding to the scale degrees of the specified key or mode.
#
# Continuing the examples above:

print("F:major       ⇒ ", librosa.key_to_degrees("F:major"))
print("D:dorian      ⇒ ", librosa.key_to_degrees("D:dorian"))
print("G♯:mixolydian ⇒ ", librosa.key_to_degrees("G♯:mixolydian"))

# %%
# In this case, the degrees are ordered according to the scale of the key, so the first degree
# corresponds to the tonic (*F*, *D*, or *G♯* in the above examples), the second to the second degree (2),
# and so on, regardless of the chromatic ordering of pitch classes.

# %%
# Unicode
# -------
# In the above examples, we have used Unicode characters for the accidental symbols (♯, ♭, etc.).
# This produces unambiguous human-readable notation, but it's not always the most convenient when working
# with user-generated (i.e. keyboard) inputs or ASCII-encoded textual data.
# For this reason, librosa also supports an ASCII-compatible notation for pitch spelling, where the
# accidentals are represented by the characters '#' (sharp), 'b' or '!' (flat), and 'n' (natural).
#
# For example, the pitch *C♯4* can also be represented as *C#4*, and *D♭4* can be represented as *Db4*.
# Double-accidentals (𝄪, 𝄫) can be represented by repeating the accidental character (e.g., *C##4* for
# *C𝄪4*).
# To produce ASCII-compatible outputs when converting to pitch notation, set the ``unicode`` flag to
# ``False``:

print("unicode=True:  ", librosa.midi_to_note(midi, key="F:major", unicode=True))
print("unicode=False: ", librosa.midi_to_note(midi, key="F:major", unicode=False))

# %%
# Hindustani and Carnatic notation
# --------------------------------
# The above examples have focused on Western music notation, but librosa also supports pitch notation
# for Hindustani and Carnatic music.  These systems differ slightly from the Western system, in that they
# rely on relative pitch rather than absolute pitch, and they use a different set of symbols to represent
# the pitch classes.  When converting to these pitch notations, it is therefore necessary to specify the
# position of 'Sa' (analogous to the tonic in Western notation) encoded in whichever input
# representation is being used (e.g. MIDI or frequency).
#
# Hindustani and Carnatic notations are related, but have some meaningful differences that require
# slightly different conventions in their conversion functions.
#
# For Hindustani notation, the pitch classes are represented by the syllables 'Sa', 'Re', 'Ga', 'Ma',
# 'Pa', 'Dha', and 'Ni', and abbreviated as 'S', 'R', 'G', 'M', 'P', 'D', and 'N'.
# For example, to convert our MIDI note numbers to Hindustani notation with 'Sa' at MIDI 60 (*C4* in
# SPN), we can do:

print(librosa.midi_to_svara_h(midi, Sa=60))

# %%
# (If converting from frequency rather than MIDI, we would have to specify 'Sa' in Hz rather than MIDI
# note number.)
#
# By default, svara are abbreviated to a single letter, but the full syllables can be obtained by setting
# the ``abbr`` flag to ``False``.
# Upper and lower case letters designate natural and altered pitch classes, respectively.
# The same MIDI numbers interpreted with Sa at MIDI 67 (*G4* in SPN) would be:

print(librosa.midi_to_svara_h(midi, Sa=67, abbr=False))

# %%
# The underlying dots indicate that the MIDI numbers lie in the octave below the specified Sa position.
# Over-dots would indicate tones in the octave above the specified Sa position:

print(librosa.midi_to_svara_h([60, 61, 67, 68, 72, 73, 79, 80], Sa=67, abbr=False))

# %%
# As in Western keys and modes, the selection of scale degrees belonging to a particular *thaat* can be
# identified by the `librosa.thaat_to_degrees` function.  For example, the *kalyan* thaat, equivalent to
# the Lydian mode in Western music:
print("Kalyan   ⇒ ", librosa.thaat_to_degrees("kalyan"))
print("C:lydian ⇒ ", librosa.key_to_degrees("C:lydian"))

# %%
# .. note::
#   Thaat degrees here are encoded as ascending with Sa corresponding to value 0.
#   This differs from the convention used by `librosa.key_to_degrees`, where the degrees are ordered
#   according to the scale of the key, so the first degree corresponds to the tonic, the second to the
#   second degree, and so on, regardless of the chromatic ordering of pitch classes.
#   The values in `librosa.key_to_degrees` are always absolute pitch classes compatible with MIDI
#   numbering, while the values in `librosa.thaat_to_degrees` are relative to the specified Sa position.

# %%
# Carnatic notation also uses a relative pitch system with 'Sa' as the reference point, but like Western
# notation (and unlike Hindustani notation), the spelling of pitch classes depends on the *raga* (here
# playing a role like the key in Western music).
# Unlike Western notation conversions, librosa does not assume a default raga for Carnatic notation, so
# it must be provided explicitly when converting.
# Librosa supports conversions using the *melakarta* system of 72 parent ragas, which may be specified
# either by canonical number or by name.
# For example, the *melakarta* raga number 1 (*Kanakangi*) has the following scale degrees:
print("Kanakangi ⇒ ", librosa.mela_to_degrees("kanakangi"))

# %%
# We can now convert the MIDI numbers above to Carnatic notation with 'Sa' at MIDI 60 (*C4* in SPN) and
# the raga set to *Kanakangi*:

print(librosa.midi_to_svara_c(midi, Sa=60, mela="kanakangi", abbr=False))

# %%
# The same MIDI numbers using melakarta number 36 (*chalanatta*), which has the same scale degrees as
# *Kanakangi* but different pitch spellings, would be:
print(librosa.midi_to_svara_c(midi, Sa=60, mela=36, abbr=False))

# %%
# Various functions exist to support interacting with both Hindustani and Carnatic notations, including
# listing the supported thaats (`librosa.list_thaat`) and melas (`librosa.list_mela`).

# %%
# Non-equal temperament
# ---------------------
# All of the above examples assume 12-tone equal temperament (12TET) tuning, which is the most common
# tuning system in Western music.
#
# Librosa does have some limited support for non-12TET tuning systems which are derived from rational
# intervals rather than equal divisions of the octave.  Examples of supported systems include Pythagorean
# tuning and p-limit just intonation (for p ≤ 17).
# Support for these systems is somewhat limited due to the equal temperament assumptions of MIDI and SPN,
# though certain analysis and conversions are possible if working directly with a reference unison
# frequency and a list of rational intervals.
#
# Moreover, the notation systems used to represent pitches in these tuning systems are not as widely adopted or standardized
# as SPN.  However, librosa does support the Functional Just System (FJS) for notating just intonation intervals.
# For example with a unison note of 'A', the interval 5:4 (major third in 5-limit tuning) can be represented as follows:

print(librosa.interval_to_fjs(5/4, unison="A"))

# %%
# where the super-scripted 5 indicates that the numerator of the interval relies on prime factor 5.  (Factors of 3 and 2 are suppressed in FJS.)

# %%
# Summary
# -------
# This section introduced various tools for converting between numerical and symbolic representations of frequency and pitch.
# Much of the unit conversion functionality described above is integrated with the display module to allow flexible annotation
# of pitch and frequency axes in visualizations.  This is covered in more detail in the following section.
