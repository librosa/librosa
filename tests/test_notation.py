#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2020-07-16 16:49:03 by Brian McFee <brian.mcfee@nyu.edu>
"""Unit tests for music notations"""
import os
import sys

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
    librosa.key_to_notes("not a key")


@pytest.mark.parametrize(
    "key,ref_notes",
    [
        # Test for implicit accidentals, ties
        ("C:maj", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
        ("A:min", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
        # Test for implicit accidentals, unambiguous
        ("D:maj", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
        ("F:min", ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]),
        # Test for proper enharmonics with ties
        ("Eb:min", ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"]),
        ("D#:min", ["C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"]),
        ("Gb:maj", ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"]),
        ("F#:maj", ["C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"]),
        # Test for theoretical keys
        (
            "G#:maj",
            ["B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        ),
        (
            "Cb:min",
            ["C", "Db", "Ebb", "Eb", "Fb", "F", "Gb", "Abb", "Ab", "Bbb", "Bb", "Cb"],
        ),
        # Test the edge case of theoretical sharps
        (
            "B#:maj",
            [
                "B#",
                "C#",
                "C##",
                "D#",
                "D##",
                "E#",
                "F#",
                "F##",
                "G#",
                "G##",
                "A#",
                "A##",
            ],
        ),
    ],
)
def test_key_to_notes(key, ref_notes):
    notes = librosa.key_to_notes(key, unicode=False)
    assert len(notes) == len(ref_notes)
    for (n, rn) in zip(notes, ref_notes):
        assert n == rn


@pytest.mark.parametrize(
    "key,ref_notes",
    [
        (
            "G#:maj",
            ["B♯", "C♯", "D", "D♯", "E", "E♯", "F♯", "F𝄪", "G♯", "A", "A♯", "B"],
        ),
        (
            "Cb:min",
            ["C", "D♭", "E𝄫", "E♭", "F♭", "F", "G♭", "A𝄫", "A♭", "B𝄫", "B♭", "C♭"],
        ),
    ],
)
def test_key_to_notes_unicode(key, ref_notes):
    notes = librosa.key_to_notes(key, unicode=True)
    assert len(notes) == len(ref_notes)
    for (n, rn) in zip(notes, ref_notes):
        assert n == rn


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_key_to_degrees_badkey():
    librosa.key_to_degrees("not a key")


@pytest.mark.parametrize(
    "key,ref_degrees",
    [
        ("C:maj", [0, 2, 4, 5, 7, 9, 11]),
        ("C:min", [0, 2, 3, 5, 7, 8, 10]),
        ("A:min", [9, 11, 0, 2, 4, 5, 7]),
        ("Gb:maj", [6, 8, 10, 11, 1, 3, 5]),
    ],
)
def test_key_to_degrees(key, ref_degrees):
    degrees = librosa.key_to_degrees(key)
    assert len(degrees) == len(ref_degrees)
    for (d, rd) in zip(degrees, ref_degrees):
        assert d == rd


def test_list_thaat():
    thaat = librosa.list_thaat()
    assert len(thaat) == 10


def test_list_mela():
    melas = librosa.list_mela()
    assert len(melas) == 72
    for k in melas:
        assert 1 <= melas[k] <= 72


@pytest.mark.parametrize("thaat", librosa.list_thaat())
def test_thaat_to_degrees(thaat):
    degrees = librosa.thaat_to_degrees(thaat)
    assert len(degrees) == 7
    assert np.all(degrees >= 0) and np.all(degrees < 12)


@pytest.mark.parametrize("mela, idx", librosa.list_mela().items())
def test_mela_to_degrees(mela, idx):
    degrees = librosa.mela_to_degrees(mela)
    assert np.allclose(degrees, librosa.mela_to_degrees(idx))
    assert len(degrees) == 7
    assert np.all(degrees >= 0) and np.all(degrees < 12)

    if idx < 37:
        # check shuddha
        assert np.isin(5, degrees)
    else:
        # check prati
        assert np.isin(6, degrees)

    # Other checks??


@pytest.mark.parametrize(
    "mela, svara",
    # This list doesn't cover all 72, but it does cover the edge cases
    [
        (1, ["S", "R₁", "G₁", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "N₁", "N₂", "N₃"]),
        (8, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
        (15, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
        (22, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
        (29, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
        (36, ["S", "R₁", "R₂", "R₃", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "D₃", "N₃"]),
        (43, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "N₁", "N₂", "N₃"]),
        (50, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
        (57, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
        (64, ["S", "R₁", "R₂", "G₂", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
        (71, ["S", "R₁", "R₂", "R₃", "G₃", "M₁", "M₂", "P", "D₁", "D₂", "N₂", "N₃"]),
    ],
)
@pytest.mark.parametrize("abbr", [False, True])
@pytest.mark.parametrize("unicode", [False, True])
def test_mela_to_svara(mela, svara, abbr, unicode):
    svara_est = librosa.mela_to_svara(mela, abbr=abbr, unicode=unicode)

    for (s1, s2) in zip(svara_est, svara):
        assert s1[0] == s2[0]

    if abbr:
        for s in svara_est:
            assert len(s) in (1, 2)
    else:
        for s in svara_est:
            assert 0 < len(s) < 5

    if sys.version >= "3.7":
        if not unicode:
            # If we're in non-unicode mode, this shouldn't raise an exception
            for s in svara_est:
                assert s.isascii()


@pytest.mark.xfail(raises=KeyError)
def test_mela_to_degrees_badmela():
    librosa.mela_to_degrees("some garbage")


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mela_to_degrees_badidx():
    librosa.mela_to_degrees(0)


@pytest.mark.xfail(raises=KeyError)
def test_mela_to_svara_badmela():
    librosa.mela_to_svara("some garbage")


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mela_to_svara_badidx():
    librosa.mela_to_svara(0)
