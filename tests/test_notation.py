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

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_simplify_note_badnote():
    librosa.core.notation.__simplify_note("not a note")

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_note_to_degree_badnote():
    librosa.core.notation.__note_to_degree("not a note")


@pytest.mark.parametrize(
    "key,ref_notes",
    [
        # Test for implicit accidentals, ties
        ("C:maj", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
        ("A:min", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
        # Test for implicit accidentals, unambiguous
        ("D:maj", ["Cn", "C#", "D", "D#", "E", "Fn", "F#", "G", "G#", "A", "A#", "B"]),
        ("F:min", ["C", "Db", "Dn", "Eb", "En", "F", "Gb", "G", "Ab", "An", "Bb", "Bn"]),
        # Test for proper enharmonics with ties
        ("Eb:min", ["Cn", "Db", "Dn", "Eb", "En", "F", "Gb", "Gn", "Ab", "An", "Bb", "Cb"]),
        ("D#:min", ["Cn", "C#", "Dn", "D#", "En", "E#", "F#", "Gn", "G#", "An", "A#", "B"]),
        ("Gb:maj", ["Cn", "Db", "Dn", "Eb", "En", "F", "Gb", "Gn", "Ab", "An", "Bb", "Cb"]),
        ("F#:maj", ["Cn", "C#", "Dn", "D#", "En", "E#", "F#", "Gn", "G#", "An", "A#", "B"]),
        # Test for theoretical keys
        (
            "G#:maj",
            ["B#", "C#", "Dn", "D#", "En", "E#", "F#", "F##", "G#", "An", "A#", "Bn"],
        ),
        (
            "Cb:min",
            ["Cn", "Db", "Ebb", "Eb", "Fb", "Fn", "Gb", "Abb", "Ab", "Bbb", "Bb", "Cb"],
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

        # Test for multiple accidentals in tonic name.
        ("F##:maj", ["B#", "C#", "C##", "D#", "D##", "E#", "E##", "F##", "G#", "G##", "A#", "A##"]),
        ("Fbb:maj", ["Dbb", "Db", "Ebb", "Fbb", "Fb", "Gbb", "Gb", "Abb", "Bbbb", "Bbb", "Cbb", "Cb"]),
        ("A###:min", ["A###", "B##", "B###", "C###", "D##", "D###", "E##", "E###", "F###", "G##", "G###", "A##"]),

        #Testing that the modes work. These were copied from the output generated in the discussion at https://github.com/librosa/librosa/pull/1739#issuecomment-1711949365.
        ("E:ion",['Cn', 'C#', 'Dn', 'D#', 'E', 'Fn', 'F#', 'Gn', 'G#', 'A', 'A#', 'B']),
        ("E#:mix",['B#', 'C#', 'C##', 'D#', 'En', 'E#', 'F#', 'F##', 'G#', 'G##', 'A#', 'Bn']),
        ("E#:lyd",['B#', 'C#', 'C##', 'D#', 'D##', 'E#', 'F#', 'F##', 'G#', 'G##', 'A#', 'A##']),
        ("Gb:dor",['Cn', 'Db', 'Dn', 'Eb', 'Fb', 'Fn', 'Gb', 'Gn', 'Ab', 'Bbb', 'Bb', 'Cb']),
        ("Gb:phr",['Dbb', 'Db', 'Ebb', 'Eb', 'Fb', 'Gbb', 'Gb', 'Abb', 'Ab', 'Bbb', 'Bb', 'Cb']),
        ("B#:aeol",['B#', 'C#', 'C##', 'D#', 'En', 'E#', 'F#', 'F##', 'G#', 'An', 'A#', 'Bn']),
        ("B#:loc",['B#', 'C#', 'Dn', 'D#', 'En', 'E#', 'F#', 'Gn', 'G#', 'An', 'A#', 'Bn'])
    ],
)
def test_key_to_notes(key, ref_notes):
    notes = librosa.key_to_notes(key, unicode=False, natural =True)
    assert len(notes) == len(ref_notes)
    for (n, rn) in zip(notes, ref_notes):
        assert n == rn

@pytest.mark.parametrize(
    "key,ref_notes,natural",
    [
        (
            "G#:maj",
            ["B♯", "C♯", "D", "D♯", "E", "E♯", "F♯", "F𝄪", "G♯", "A", "A♯", "B"],
            False
        ),
        (
            "Cb:min",
            ["C", "D♭", "E𝄫", "E♭", "F♭", "F", "G♭", "A𝄫", "A♭", "B𝄫", "B♭", "C♭"],
            False
        ),
        (
            "G#:maj",
            ["B♯", "C♯", "D♮", "D♯", "E♮", "E♯", "F♯", "F𝄪", "G♯", "A♮", "A♯", "B♮"],
            True
        ),
        (
            "G#:ion",
            ["B♯", "C♯", "D♮", "D♯", "E♮", "E♯", "F♯", "F𝄪", "G♯", "A♮", "A♯", "B♮"],
            True
        ),
    ],
)
def test_key_to_notes_unicode(key, ref_notes, natural):
    notes = librosa.key_to_notes(key, unicode=True, natural = natural)
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
def test_key_to_notes_no_natural(key, ref_notes):
    notes = librosa.key_to_notes(key, unicode=True, natural=False)
    assert len(notes) == len(ref_notes)
    for (n, rn) in zip(notes, ref_notes):
        assert n == rn

@pytest.mark.parametrize(
    "note, ref_simplified_ascii",
    [
        (
            "G####bb", "G##"
        ),

        (
            "F#n", "F#"
        ),
    ],
)
def test_simplify_note_ascii(note, ref_simplified_ascii):
    simplified_note = librosa.core.notation.__simplify_note(note, unicode=False)
    for (n, rn) in zip(simplified_note, ref_simplified_ascii):
        assert n == rn

@pytest.mark.parametrize(
    "notes, ref_simplified_array",
    [
        (
            ['C♭♯', 'C♭♭♭'], ['C', 'C♭𝄫']
        )
    ],
)
def test_simplify_note_array(notes, ref_simplified_array):
    simplified_note = librosa.core.notation.__simplify_note(notes)
    for (n, rn) in zip(simplified_note, ref_simplified_array):
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
        ("A###:maj", [0, 2, 4, 5, 7, 9, 11]),
        ("C:ion", [ 0,  2,  4,  5,  7,  9, 11]),
        ("C:dor", [ 0,  2,  3,  5,  7,  9, 10]),
        ("C:phr", [ 0,  1,  3,  5,  7,  8, 10]),
        ("D#:lyd", [ 3,  5,  7,  9, 10,  0,  2]),
        ("D#:mix", [ 3,  5,  7,  8, 10,  0,  1]),
        ("Ebb:aeol" , [2, 4, 5, 7, 9, 10, 0]),
        ("Ebb:loc", [2, 3, 5, 7, 8, 10, 0])

    ],
)
def test_key_to_degrees(key, ref_degrees):
    degrees = librosa.key_to_degrees(key)
    assert len(degrees) == len(ref_degrees)
    for (d, rd) in zip(degrees, ref_degrees):
        assert d == rd

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mode_to_key_badkey():
    librosa.core.notation.__mode_to_key("not a key")

@pytest.mark.parametrize(
    "mode, ref_mode",
    [
        (
            'C:maj', 'C:maj'
        )
    ],
)
def test_mode_to_key_no_change(mode, ref_mode):
    simplified_mode = librosa.core.notation.__mode_to_key(mode)
    for (n, rn) in zip(mode, ref_mode):
        assert n == rn

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

@pytest.mark.parametrize('unison, fifths, unicode, result',
        [
            ('C', 0, True, 'C'),
            ('C', 1, True, 'G'),
            ('C', -2, True, 'B♭'),
            ('C', -2, False, 'Bb'),
            ('F', 1, True, 'C'),
            ('F', -1, True, 'B♭'),
            ('B', -7, True, 'B♭'),
            ('Bb', 7, True, 'B'),
            ('Bb', 14, True, 'B♯'),
            ('B', 1, True, 'F♯'),
            ('B', 14, True, 'B𝄪'),
            ('B', -14, True, 'B𝄫'),
            ('B', 21, True, 'B𝄪♯'),
            ('B', -21, True, 'B𝄫♭'),
            ('B', 21, False, 'B###'),
            ('B', -21, False, 'Bbbb'),
        ]
)
def test_fifths_to_note(unison, fifths, unicode, result):
    note = librosa.core.notation.fifths_to_note(unison=unison, fifths=fifths, unicode=unicode)
    assert note == result


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_fifths_to_note_badunison():
    librosa.fifths_to_note(unison='X', fifths=1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_interval_to_fjs_irrational():
    # Test FJS conversion with a non-just interval
    librosa.interval_to_fjs(np.sqrt(2))

@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize('r', [0, -1, -1/2])
def test_interval_to_fjs_nonpos(r):
    librosa.interval_to_fjs(r)


@pytest.mark.parametrize('interval, unison, unicode, result',
        [
            (1, 'C', True, 'C'),
            (2, 'G', True, 'G'),
            (1/2, 'F#', True, 'F♯'),
            (1/2, 'F#', False, 'F#'),
            (3/2, 'C', True, 'G'),
            (5/4, 'C', True, 'E⁵'),
            (5/4, 'C', False, 'E^5'),
            (8/5, 'E', True, 'C₅'),
            (8/5, 'E', False, 'C_5'),
            (7/5, 'F', True, 'B⁷₅'),
            (7/5, 'F', False, 'B^7_5'),
            (49, 'C', True, 'G⁴⁹'),
            (1/49, 'C', True, 'F₄₉'),
        ]
)
def test_interval_to_fjs(interval, unison, unicode, result):
    note = librosa.interval_to_fjs(interval, unison=unison, unicode=unicode)

    assert note == result


@pytest.mark.parametrize('unison', ['C', 'F#', 'Gbb'])
@pytest.mark.parametrize('unicode', [False, True])
@pytest.mark.parametrize('intervals', [librosa.plimit_intervals(primes=[3,5,7], bins_per_octave=24)])
def test_interval_to_fjs_set(unison, unicode, intervals):
    fjs = librosa.interval_to_fjs(intervals, unison=unison, unicode=unicode)

    for (interval, note) in zip(intervals, fjs):
        fjs_single = librosa.interval_to_fjs(interval, unison=unison, unicode=unicode)
        assert fjs_single == note


@pytest.mark.parametrize('hz, fmin, unison, unicode, results',
        [
            ([55, 66, 77], None, None, True, ['A', 'C₅', 'D♯⁷₅']),
            ([55, 66, 77], 33, None, True, ['A⁵', 'C', 'E♭⁷']),
            ([55, 66, 77], 33, 'Cb', True, ['A♭⁵', 'C♭', 'E𝄫⁷']),
            ([55, 66, 77], 33, 'Cb', False, ['Ab^5', 'Cb', 'Ebb^7']),
        ]
)
def test_hz_to_fjs(hz, fmin, unison, unicode, results):
    fjs = librosa.hz_to_fjs(hz, fmin=fmin, unison=unison, unicode=unicode)
    assert list(fjs) == results


def test_hz_to_fjs_scalar():
    fjs = librosa.hz_to_fjs(110, fmin=55, unicode=False)

    assert fjs == 'A'
