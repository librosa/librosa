import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import librosa


def test_tone_str():
    t_str = librosa.tone("A4", duration=0.1, sr=22050)
    t_num = librosa.tone(librosa.note_to_hz("A4"), duration=0.1, sr=22050)
    assert np.allclose(t_str, t_num)


def test_chirp_str():
    c_str = librosa.chirp(fmin="A2", fmax="A4", duration=0.1, sr=22050)
    c_num = librosa.chirp(
        fmin=librosa.note_to_hz("A2"),
        fmax=librosa.note_to_hz("A4"),
        duration=0.1,
        sr=22050,
    )
    assert np.allclose(c_str, c_num)


def test_clicks_str():
    clk_str = librosa.clicks(click_freq="A5", times=[0.05], length=2205, sr=22050)
    clk_num = librosa.clicks(
        click_freq=librosa.note_to_hz("A5"), times=[0.05], length=2205, sr=22050
    )
    assert np.allclose(clk_str, clk_num)


def test_cqt_fmin_str():
    y = np.random.default_rng(0).standard_normal(4096)
    sr = 22050
    C_str = librosa.cqt(y, sr=sr, fmin="C2", n_bins=24)
    C_ref = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz("C2"), n_bins=24)
    assert np.allclose(C_str, C_ref)


def test_vqt_fmin_str():
    y = np.random.default_rng(0).standard_normal(4096)
    sr = 22050
    V_str = librosa.vqt(y, sr=sr, fmin="C2", n_bins=24, intervals="equal")
    V_ref = librosa.vqt(
        y, sr=sr, fmin=librosa.note_to_hz("C2"), n_bins=24, intervals="equal"
    )
    assert np.allclose(V_str, V_ref)


def test_hybrid_cqt_fmin_str():
    y = np.random.default_rng(0).standard_normal(4096)
    sr = 22050
    H_str = librosa.hybrid_cqt(y, sr=sr, fmin="C2", n_bins=24)
    H_ref = librosa.hybrid_cqt(y, sr=sr, fmin=librosa.note_to_hz("C2"), n_bins=24)
    assert np.allclose(H_str, H_ref)


def test_pseudo_cqt_fmin_str():
    y = np.random.default_rng(0).standard_normal(4096)
    sr = 22050
    P_str = librosa.pseudo_cqt(y, sr=sr, fmin="C2", n_bins=24)
    P_ref = librosa.pseudo_cqt(y, sr=sr, fmin=librosa.note_to_hz("C2"), n_bins=24)
    assert np.allclose(P_str, P_ref)


def test_icqt_fmin_str():
    y = np.random.default_rng(0).standard_normal(4096)
    sr = 22050
    C = librosa.cqt(y, sr=sr, fmin="C2", n_bins=24)
    y_inv_str = librosa.icqt(C, sr=sr, fmin="C2")
    y_inv_num = librosa.icqt(C, sr=sr, fmin=librosa.note_to_hz("C2"))
    assert np.allclose(y_inv_str, y_inv_num)


def test_yin_fmin_fmax_str():
    y = librosa.chirp(fmin=440, fmax=880, duration=0.5, sr=22050)
    f0_str = librosa.yin(y, fmin="A4", fmax="A5", sr=22050)
    f0_ref = librosa.yin(
        y, fmin=librosa.note_to_hz("A4"), fmax=librosa.note_to_hz("A5"), sr=22050
    )
    assert np.allclose(f0_str, f0_ref)


def test_pyin_fmin_fmax_str():
    y = librosa.chirp(fmin=440, fmax=880, duration=0.5, sr=22050)
    f0_str, v_str, p_str = librosa.pyin(y, fmin="A4", fmax="A5", sr=22050)
    f0_ref, v_ref, p_ref = librosa.pyin(
        y, fmin=librosa.note_to_hz("A4"), fmax=librosa.note_to_hz("A5"), sr=22050
    )
    assert np.allclose(f0_str, f0_ref, equal_nan=True)
    assert np.allclose(v_str, v_ref)
    assert np.allclose(p_str, p_ref)


def test_piptrack_fmin_fmax_str():
    y = librosa.chirp(fmin=440, fmax=880, duration=0.5, sr=22050)
    pitches_str, mags_str = librosa.piptrack(y=y, sr=22050, fmin="A4", fmax="A5")
    pitches_ref, mags_ref = librosa.piptrack(
        y=y,
        sr=22050,
        fmin=librosa.note_to_hz("A4"),
        fmax=librosa.note_to_hz("A5"),
    )
    assert np.allclose(pitches_str, pitches_ref)
    assert np.allclose(mags_str, mags_ref)


def test_cqt_frequencies_str():
    freqs_str = librosa.cqt_frequencies(12, fmin="C2")
    freqs_num = librosa.cqt_frequencies(12, fmin=librosa.note_to_hz("C2"))
    assert np.allclose(freqs_str, freqs_num)


def test_mel_frequencies_str():
    freqs_str = librosa.mel_frequencies(128, fmin="C2", fmax="C7")
    freqs_num = librosa.mel_frequencies(
        128, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    assert np.allclose(freqs_str, freqs_num)


def test_interval_frequencies_str():
    freqs_str = librosa.interval_frequencies(12, fmin="A1", intervals="equal")
    freqs_num = librosa.interval_frequencies(
        12, fmin=librosa.note_to_hz("A1"), intervals="equal"
    )
    assert np.allclose(freqs_str, freqs_num)


def test_hz_to_fjs_str():
    fjs_str = librosa.hz_to_fjs([440, 880], fmin="A4")
    fjs_num = librosa.hz_to_fjs([440, 880], fmin=440.0)
    assert list(fjs_str) == list(fjs_num)


def test_filters_mel_str():
    m_str = librosa.filters.mel(sr=22050, n_fft=2048, fmin="C2", fmax="C7")
    m_num = librosa.filters.mel(
        sr=22050,
        n_fft=2048,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    assert np.allclose(m_str, m_num)


def test_filters_cq_to_chroma_str():
    f_str = librosa.filters.cq_to_chroma(n_input=84, bins_per_octave=12, fmin="C2")
    f_num = librosa.filters.cq_to_chroma(
        n_input=84, bins_per_octave=12, fmin=librosa.note_to_hz("C2")
    )
    assert np.allclose(f_str, f_num)


def test_spectral_features_str():
    y = np.random.default_rng(0).standard_normal(4096)
    sr = 22050
    # chroma_cqt
    chr_str = librosa.feature.chroma_cqt(y=y, sr=sr, fmin="C2")
    chr_num = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz("C2"))
    assert np.allclose(chr_str, chr_num)

    # chroma_cens
    cens_str = librosa.feature.chroma_cens(y=y, sr=sr, fmin="C2")
    cens_num = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz("C2"))
    assert np.allclose(cens_str, cens_num)

    # chroma_vqt
    vqt_str = librosa.feature.chroma_vqt(y=y, sr=sr, fmin="C2", intervals="equal")
    vqt_num = librosa.feature.chroma_vqt(
        y=y, sr=sr, fmin=librosa.note_to_hz("C2"), intervals="equal"
    )
    assert np.allclose(vqt_str, vqt_num)

    # spectral_contrast
    con_str = librosa.feature.spectral_contrast(y=y, sr=sr, fmin="C3")
    con_num = librosa.feature.spectral_contrast(
        y=y, sr=sr, fmin=librosa.note_to_hz("C3")
    )
    assert np.allclose(con_str, con_num)


def test_display_specshow_str():
    fig, ax = plt.subplots(nrows=2)
    D = np.abs(librosa.stft(np.zeros(2048)))
    # specshow with fmin, fmax
    librosa.display.specshow(D, sr=22050, y_axis="mel", fmin="C2", fmax="C7", ax=ax[0])
    librosa.display.specshow(D, sr=22050, y_axis="cqt_hz", fmin="C2", ax=ax[1])
    plt.close(fig)


def test_invalid_str_raises():
    with pytest.raises(librosa.ParameterError):
        librosa.tone("not_a_note", duration=0.1)

    with pytest.raises(librosa.ParameterError):
        librosa.cqt_frequencies(12, fmin="not_a_note")
