#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2020-07-16 16:49:03 by Brian McFee <brian.mcfee@nyu.edu>
"""Unit tests for music notations"""
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import warnings
import numpy as np
import pytest
import librosa

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_key_to_notes_badkey():
    librosa.key_to_notes('not a key')


@pytest.mark.parametrize('key,ref_notes', [
                                        # Test for implicit accidentals, ties
                                        ('C:maj', ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']),
                                        ('A:min', ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']),
                                        # Test for implicit accidentals, unambiguous
                                        ('D:maj', ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']),
                                        ('F:min', ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']),
                                        # Test for proper enharmonics with ties
                                        ('Eb:min', ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']),
                                        ('D#:min', ['C', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#', 'B']),
                                        ('Gb:maj', ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']),
                                        ('F#:maj', ['C', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#', 'B']),
                                        # Test for theoretical keys
                                        ('G#:maj', ['B#', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'F##', 'G#', 'A', 'A#', 'B']),
                                        ('Cb:min', ['C', 'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'Bbb', 'Bb', 'Cb']),
                                        # Test the edge case of theoretical sharps
                                        ('B#:maj', ['B#', 'C#', 'C##', 'D#', 'D##', 'E#', 'F#', 'F##', 'G#', 'G##', 'A#', 'A##']),
                                    ])
def test_key_to_notes(key, ref_notes):
    notes = librosa.key_to_notes(key, unicode=False)
    assert len(notes) == len(ref_notes)
    for (n, rn) in zip(notes, ref_notes):
        assert n == rn


@pytest.mark.parametrize('key,ref_notes', [
                                        ('G#:maj', ['B♯', 'C♯', 'D', 'D♯', 'E', 'E♯', 'F♯', 'F𝄪', 'G♯', 'A', 'A♯', 'B']),
                                        ('Cb:min', ['C', 'D♭', 'E𝄫', 'E♭', 'F♭', 'F', 'G♭', 'A𝄫', 'A♭', 'B𝄫', 'B♭', 'C♭'])
                                    ])
def test_key_to_notes_unicode(key, ref_notes):
    notes = librosa.key_to_notes(key, unicode=True)
    assert len(notes) == len(ref_notes)
    for (n, rn) in zip(notes, ref_notes):
        assert n == rn


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_key_to_degrees_badkey():
    librosa.key_to_degrees('not a key')


@pytest.mark.parametrize('key,ref_degrees', [('C:maj', [0, 2, 4, 5, 7, 9, 11]),
                                             ('C:min', [0, 2, 3, 5, 7, 8, 10]),
                                             ('A:min', [ 9, 11,  0,  2,  4,  5,  7]),
                                             ('Gb:maj', [ 6,  8, 10, 11,  1,  3,  5])])
def test_key_to_degrees(key, ref_degrees):
    degrees = librosa.key_to_degrees(key)
    assert len(degrees) == len(ref_degrees)
    for (d, rd) in zip(degrees, ref_degrees):
        assert d == rd

