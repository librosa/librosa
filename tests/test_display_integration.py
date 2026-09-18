#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integration and State Validation Tests for librosa.display"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh, PatchCollection
from matplotlib.ticker import ScalarFormatter

import librosa
from librosa.display.formatting import TimeFormatter, ChromaFormatter, NoteFormatter, AdaptiveWaveplot

@pytest.fixture(scope="module")
def audio_data():
    """Load a short example audio file."""
    y, sr = librosa.load(librosa.ex("trumpet"), duration=2)
    return y, sr

# -------------------------------------------------------------------------
# 1. Pipeline Integration & Matplotlib State Validation
# -------------------------------------------------------------------------

def test_pipeline_melspectrogram_to_specshow(audio_data):
    """Test feature.melspectrogram -> display.specshow with state validation."""
    y, sr = audio_data
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots()
    mesh = librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sr, ax=ax)
    
    # Object type assertions
    assert isinstance(mesh, QuadMesh), "specshow should return a QuadMesh"
    
    # State validation: Labels
    assert ax.get_xlabel() == 'Time', "x-axis should be labeled 'Time'"
    assert ax.get_ylabel() == 'Hz', "y-axis should be labeled 'Hz' for mel scale"
    
    # State validation: Scales
    # The mel scale uses a symlog transformation internally
    assert ax.get_yscale() == 'symlog', "y-axis should use 'symlog' scale for mel"
    
    # State validation: Formatters
    assert isinstance(ax.xaxis.get_major_formatter(), TimeFormatter)
    assert isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter)
    
    plt.close(fig)

@pytest.mark.filterwarnings("ignore:n_fft=1024 is too large")
def test_pipeline_chroma_to_specshow(audio_data):
    """Test feature.chroma_cqt -> display.specshow with Chroma formatting."""
    y, sr = audio_data
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    fig, ax = plt.subplots()
    mesh = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", ax=ax)
    
    # Object type assertions
    assert isinstance(mesh, QuadMesh)
    
    # State validation: Labels and formatters
    assert ax.get_ylabel() == 'Pitch class'
    assert isinstance(ax.yaxis.get_major_formatter(), ChromaFormatter)
    assert isinstance(ax.xaxis.get_major_formatter(), TimeFormatter)
    
    plt.close(fig)

def test_pipeline_waveshow_state(audio_data):
    """Test display.waveshow state and adaptive plotting."""
    y, sr = audio_data
    
    fig, ax = plt.subplots()
    out = librosa.display.waveshow(y, sr=sr, ax=ax, axis='time')
    
    # Object type assertions
    assert isinstance(out, AdaptiveWaveplot), "waveshow should return an AdaptiveWaveplot"
    
    # State validation
    assert ax.get_xlabel() == 'Time'
    assert isinstance(ax.xaxis.get_major_formatter(), TimeFormatter)
    
    plt.close(fig)

def test_pipeline_wavebars(audio_data):
    """Test display.wavebars state validation."""
    y, sr = audio_data
    
    fig, ax = plt.subplots()
    out = librosa.display.wavebars(y, sr=sr, ax=ax, axis='time')
    
    # Object type assertions
    assert isinstance(out, PatchCollection), "wavebars should return a PatchCollection"
    
    # State validation
    assert ax.get_xlabel() == 'Time'
    assert isinstance(ax.xaxis.get_major_formatter(), TimeFormatter)
    
    plt.close(fig)

def test_pipeline_pyin_to_wavef0(audio_data):
    """Test feature.pyin -> display.wavef0 for pitch tracking overlays."""
    y, sr = audio_data
    # Extract f0 using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    
    fig, ax = plt.subplots()
    out = librosa.display.wavef0(y, sr=sr, f0=f0, ax=ax, time_axis='time', freq_axis='cqt_note')
    
    # Object type assertions
    assert isinstance(out, AdaptiveWaveplot)
    
    # State validation: Y-axis should be labeled according to 'cqt_note'
    assert ax.get_ylabel() == 'Note'
    assert isinstance(ax.yaxis.get_major_formatter(), NoteFormatter)
    
    plt.close(fig)

# -------------------------------------------------------------------------
# 2. Namespace & API Fidelity Tests
# -------------------------------------------------------------------------

def test_display_namespace_fidelity():
    """
    Ensure the refactored display module correctly exposes 
    the legacy public symbols from the new submodules.
    """
    expected_public_symbols = [
        'specshow',
        'waveshow',
        'wavebars',
        'wavef0',
        'TimeFormatter',
        'NoteFormatter',
        'LogHzFormatter',
        'ChromaFormatter',
        'TonnetzFormatter',
        'FJSFormatter',
        'AdaptiveWaveplot',
        'cmap',
        'multiplot'
    ]
    
    for symbol in expected_public_symbols:
        assert hasattr(librosa.display, symbol), f"Missing '{symbol}' in librosa.display namespace."

def test_display_internal_aliases():
    """
    Ensure the internal aliases expected by existing tests are preserved.
    """
    expected_internal_aliases = [
        '_same_axes',
        '_parse_vscale',
        '_mp_setup_axes',
        '_WAVESHOW_ADAPTORS',
        '_squeeze_shape',
    ]
    
    for symbol in expected_internal_aliases:
        assert hasattr(librosa.display, symbol), f"Missing internal alias '{symbol}' in librosa.display."
