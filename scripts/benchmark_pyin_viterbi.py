#!/usr/bin/env python
import argparse
import importlib
import json
import pathlib
import sys
import time

import numpy as np
import scipy

# Ensure the local checkout is imported rather than any site-packages install.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa

pitch_mod = importlib.import_module("librosa.core.pitch")
PYIN_VITERBI = getattr(pitch_mod, "__pyin_viterbi")
PYIN_HELPER = getattr(pitch_mod, "__pyin_helper")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--repeat-audio", type=int, default=1)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def synth_audio(duration: float, sr: int, repeat_audio: int) -> np.ndarray:
    y = librosa.chirp(fmin=220, fmax=640, sr=sr, duration=duration, linear=False)
    y = np.pad(y, (sr,))
    if repeat_audio > 1:
        y = np.tile(y, repeat_audio)
    return y.astype(np.float64, copy=False)


def time_call(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    durations_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    durations_ms.sort()
    return durations_ms[len(durations_ms) // 2]


def prepare_pyin_state(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: float = 110.0,
    fmax: float = 880.0,
    n_thresholds: int = 100,
    beta_parameters=(2, 18),
    boltzmann_parameter: float = 2.0,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
):
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)
    y_frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    yin_frames = pitch_mod._cumulative_mean_normalized_difference(
        y_frames, min_period, max_period
    )
    parabolic_shifts = pitch_mod._parabolic_interpolation(yin_frames)

    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    helper = np.vectorize(
        lambda a, b: PYIN_HELPER(
            a,
            b,
            sr,
            thresholds,
            boltzmann_parameter,
            beta_probs,
            no_trough_prob,
            min_period,
            fmin,
            n_pitch_bins,
            n_bins_per_semitone,
        ),
        signature="(f,t),(k,t)->(1,d,t),(j,t)",
    )
    observation_probs, voiced_prob = helper(yin_frames, parabolic_shifts)

    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    transition_local = librosa.sequence.transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )
    p_init = np.ones(2 * n_pitch_bins) / (2 * n_pitch_bins)

    return {
        "observation_probs": observation_probs,
        "voiced_prob": voiced_prob,
        "transition_local": transition_local,
        "switch_prob": switch_prob,
        "p_init": p_init,
        "frames": int(observation_probs.shape[-1]),
    }


def benchmark(args):
    y = synth_audio(args.duration, args.sr, args.repeat_audio)
    state = prepare_pyin_state(y, args.sr)

    legacy_states = PYIN_VITERBI(
        state["observation_probs"],
        state["transition_local"],
        p_init=state["p_init"],
        viterbi_impl="legacy",
        switch_prob=state["switch_prob"],
    )
    fast_states = PYIN_VITERBI(
        state["observation_probs"],
        state["transition_local"],
        p_init=state["p_init"],
        viterbi_impl="fast",
        switch_prob=state["switch_prob"],
    )

    f0_legacy, voiced_legacy, prob_legacy = librosa.pyin(
        y,
        sr=args.sr,
        fmin=110,
        fmax=880,
        center=False,
        fill_na=-1,
        viterbi_impl="legacy",
    )
    f0_fast, voiced_fast, prob_fast = librosa.pyin(
        y,
        sr=args.sr,
        fmin=110,
        fmax=880,
        center=False,
        fill_na=-1,
        viterbi_impl="fast",
    )

    return {
        "duration": args.duration,
        "repeat_audio": args.repeat_audio,
        "frames": state["frames"],
        "decoder_parity_ok": bool(np.array_equal(legacy_states, fast_states)),
        "f0_parity_ok": bool(np.array_equal(f0_legacy, f0_fast)),
        "voiced_parity_ok": bool(np.array_equal(voiced_legacy, voiced_fast)),
        "prob_parity_ok": bool(np.array_equal(prob_legacy, prob_fast)),
        "pyin_viterbi_legacy_ms": time_call(
            lambda: PYIN_VITERBI(
                state["observation_probs"],
                state["transition_local"],
                p_init=state["p_init"],
                viterbi_impl="legacy",
                switch_prob=state["switch_prob"],
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "pyin_viterbi_fast_ms": time_call(
            lambda: PYIN_VITERBI(
                state["observation_probs"],
                state["transition_local"],
                p_init=state["p_init"],
                viterbi_impl="fast",
                switch_prob=state["switch_prob"],
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "pyin_legacy_ms": time_call(
            lambda: librosa.pyin(
                y,
                sr=args.sr,
                fmin=110,
                fmax=880,
                center=False,
                fill_na=-1,
                viterbi_impl="legacy",
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "pyin_fast_ms": time_call(
            lambda: librosa.pyin(
                y,
                sr=args.sr,
                fmin=110,
                fmax=880,
                center=False,
                fill_na=-1,
                viterbi_impl="fast",
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }


def render_markdown(result):
    core_speedup = result["pyin_viterbi_legacy_ms"] / result["pyin_viterbi_fast_ms"]
    full_speedup = result["pyin_legacy_ms"] / result["pyin_fast_ms"]
    lines = [
        "## librosa pYIN Benchmark",
        "",
        f"**Duration:** `{result['duration']}` seconds",
        f"**Repeated:** `{result['repeat_audio']}`",
        f"**Frames:** `{result['frames']}`",
        "",
        "---",
        "",
        "## pYIN Decoder Core",
        "",
        "| Impl | Time |",
        "|:--|--:|",
        f"| `legacy` | **{result['pyin_viterbi_legacy_ms']:.3f} ms** |",
        f"| `fast` | **{result['pyin_viterbi_fast_ms']:.3f} ms** |",
        "",
        f"> ✅ **Core speedup:** `{core_speedup:.2f}x`",
        "",
        "---",
        "",
        "## Full pYIN",
        "",
        "| Impl | Time |",
        "|:--|--:|",
        f"| `legacy` | **{result['pyin_legacy_ms']:.3f} ms** |",
        f"| `fast` | **{result['pyin_fast_ms']:.3f} ms** |",
        "",
        f"> ✅ **End-to-end speedup:** `{full_speedup:.2f}x`",
        "",
        "## Parity",
        "",
        f"- decoder states: `{'ok' if result['decoder_parity_ok'] else 'mismatch'}`",
        f"- f0: `{'ok' if result['f0_parity_ok'] else 'mismatch'}`",
        f"- voiced flags: `{'ok' if result['voiced_parity_ok'] else 'mismatch'}`",
        f"- voiced probability: `{'ok' if result['prob_parity_ok'] else 'mismatch'}`",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    result = benchmark(args)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(render_markdown(result))


if __name__ == "__main__":
    main()
