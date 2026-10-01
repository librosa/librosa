import pytest
import librosa.display

def test_public_api_surface():
    expected_symbols = [
        "specshow",
        "waveshow",
        "wavebars",
        "wavef0",
        "multiplot",
        "highlight",
        "infer_cmap",
        "colorbar_db",
        "colorbar_phase",
        "legend_for_axes",
        "TimeFormatter",
        "NoteFormatter",
        "FJSFormatter",
        "LogHzFormatter",
        "ChromaFormatter",
        "ChromaSvaraFormatter",
        "ChromaFJSFormatter",
        "TonnetzFormatter",
        "AdaptiveWaveplot",
        "Transformf0",
    ]
    for symbol in expected_symbols:
        assert hasattr(librosa.display, symbol), f"librosa.display missing {symbol}"

def test_internal_aliases_present():
    aliases = [
        "_same_axes",
        "_parse_vscale",
        "_squeeze_shape",
        "_resolve_multiplot",
        "_mp_get_layout",
        "_mp_setup_axes",
        "_mp_setup_labels",
        "_mp_setup_prop_group",
        "_mp_setup_properties",
        "_WAVESHOW_ADAPTORS"
    ]
    for a in aliases:
        assert hasattr(librosa.display, a), f"librosa.display missing internal alias {a}"
