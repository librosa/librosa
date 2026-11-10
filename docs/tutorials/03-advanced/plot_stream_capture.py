# coding: utf-8
# Code source: Brian McFee
# License: ISC
"""
======================
Live stream processing
======================

This tutorial illustrates how to perform real-time stream processing and visualization
on signals captured from an audio device.

Live audio processing often follows a **producer-consumer architecture**:

1. **Producer (Audio Device Layer)**: An audio capture stream (such as `sounddevice.InputStream`)
   invokes a high-priority system callback whenever a new block of audio samples arrives from
   the hardware. To prevent buffer overruns and audio dropouts, this callback must never perform
   heavy computations, blocking I/O, or GUI drawing; instead, it immediately places incoming
   blocks into a thread-safe FIFO (first-in, first-out) queue.
2. **I/O Bridge (`librosa.util.StreamBridge`)**: Rather than managing manual queues, callbacks,
   and channel transposition, Librosa provides `librosa.util.StreamBridge` (or
   `librosa.util.streaming.StreamBridge`). It serves directly as the audio callback,
   normalizes incoming audio arrays, manages queue capacity and overflow policies, and yields
   normalized NumPy blocks for consumption.
3. **Sliding Analysis Buffer (`librosa.StreamBuffer`)**: For frequency-domain features (like STFT,
   chroma, or spectral centroid), analysis requires overlapping observation windows and often
   sample-rate conversion. `librosa.StreamBuffer` maintains a sliding analysis window and manages
   online streaming resampling via `soxr`.
4. **Display Layer**: The extracted features are rendered in real time (for example, as an
   animated rolling sparkline using `matplotlib.animation.FuncAnimation` as below).

Aside: virtual devices for this documentation
---------------------------------------------
This tutorial document exists as part of the librosa documentation site, which is compiled
automatically in a virtual environment and without access to physical audio devices.

To illustrate the necessary concepts for this section, we construct a virtual audio device
(`VirtualInputStream`) using `soundfile` to stream fixed-size blocks from one of librosa's packaged example recordings.

.. note::
    The `VirtualInputStream` code below is only needed for this document to run offline.
    If you are adapting this tutorial for your own use with a real audio device, the `VirtualInputStream`
    code is not needed.


In this notebook, we present three examples:

- **Example 1**: Basic stream processing with `librosa.util.StreamBridge`, computing instantaneous Root-Mean-Square (RMS) energy.
- **Example 2**: Sliding analysis buffers and online resampling with `librosa.StreamBuffer` to track spectral centroid.
- **Example 3**: Asynchronous streaming using `librosa.util.async_capture` and `librosa.astream`.

At the end, we describe how to swap the virtual device for a live microphone with `sounddevice.InputStream`.
"""

# %%
# Imports and Audio Source
# ------------------------
# We'll need standard plotting modules, `soundfile` for simulated audio I/O,
# and `librosa` for streaming utilities and signal processing.

from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import librosa

# We'll use the packaged 'trumpet' recording as our audio source
filename = librosa.ex("trumpet")
HTML(librosa.util.example_info("trumpet", html=True))


# %%
# Part 0: The Audio Device and Bridge Setup
# ----------------------------------------
# In real-time audio systems, the hardware driver calls an audio callback
# function at regular intervals in a separate, high-priority OS thread.
#
# `librosa.util.StreamBridge` acts directly as this callback function. It takes
# incoming chunks, converts them to Librosa-canonical floating point arrays,
# and stores them in an internal queue without blocking the audio driver thread.
#
# For consumer loops and animation frame callbacks, `StreamBridge.drain()` retrieves
# all newly arrived blocks without blocking.


# %%
# Simulating the Audio Device
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In an offline build or automated test environment, no physical input device is available.
# We create a `VirtualInputStream` class that mirrors the context-manager interface of
# `sounddevice.InputStream`. It reads fixed-size chunks from an audio file using
# `soundfile.SoundFile.blocks()` and feeds them to the exact same audio callback.
#
# Notice that this virtual stream knows nothing about `librosa`—it strictly mimics hardware I/O.


class VirtualInputStream:
    """Simulate sounddevice.InputStream from an audio file.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    samplerate : int, optional
        Target sample rate. If None, uses the file's native sample rate.
    blocksize : int
        Number of frames per block.
    channels : int
        Number of channels.
    dtype : str
        Data type of the audio samples.
    callback : callable
        Audio callback function matching PortAudio callback signature.
    """

    def __init__(
        self,
        filepath,
        samplerate=None,
        blocksize=1024,
        channels=1,
        dtype="float32",
        callback=None,
    ):
        self.filepath = filepath
        self.blocksize = blocksize
        self.channels = channels
        self.dtype = dtype
        self.callback = callback
        with sf.SoundFile(self.filepath) as af:
            self.samplerate = samplerate or af.samplerate
            self.total_frames = af.frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        """Generator yielding blocks from the file into the callback."""
        with sf.SoundFile(self.filepath) as af:
            for block in af.blocks(
                blocksize=self.blocksize, dtype=self.dtype, always_2d=True
            ):
                if self.callback:
                    self.callback(block, len(block), None, None)
                yield block


# %%
# Part 1: Example 1 — Basic Stream Processing (RMS Energy)
# --------------------------------------------------------
# In our first example, we process incoming blocks independently to compute
# instantaneous Root-Mean-Square (RMS) energy.
#
# The processing function receives a raw in-memory numpy array from `StreamBridge`,
# with zero knowledge of whether it originated from a real microphone or our simulated device.


def process_rms(buffer):
    """Compute RMS energy from an incoming audio buffer using librosa.

    Parameters
    ----------
    buffer : np.ndarray
        Audio buffer of shape (frames, channels) or (frames,).

    Returns
    -------
    rms_value : float
        Scalar RMS energy of the current block.
    """
    y = buffer.squeeze()
    # Compute RMS energy for the current frame
    rms = librosa.feature.rms(
        y=y, frame_length=len(y), hop_length=len(y), center=False
    )
    return float(rms[0, 0])


# %%
# Visualizing the Live RMS Sparkline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We create a `StreamBridge` and pass it directly as the stream callback.
# On each animation tick, `bridge_rms.drain()` yields whatever chunks have arrived.
# Time 0 on the plot corresponds directly to time 0 in the audio stream.

blocksize = 1024

bridge_rms = librosa.util.StreamBridge(orig_sr=22050, channels=1)

with VirtualInputStream(
    filename, blocksize=blocksize, callback=bridge_rms
) as stream:
    sr = stream.samplerate
    duration = stream.total_frames / sr
    n_blocks = int(np.ceil(stream.total_frames / blocksize))
    times = np.arange(n_blocks) * (blocksize / sr)
    rms_values = np.full(n_blocks, np.nan)

    fig_rms, ax_rms = plt.subplots(figsize=(8, 3))
    (line_rms,) = ax_rms.plot(times, rms_values, color="#3f90da", lw=2)
    ax_rms.set_ylim(0, 0.4)
    ax_rms.set_xlim(0, duration)
    ax_rms.set_title("Live Stream Processing: Instantaneous RMS Energy")
    ax_rms.set_xlabel("Time (seconds)")
    ax_rms.set_ylabel("RMS Amplitude")
    ax_rms.grid(True, linestyle="--", alpha=0.5)

    block_idx_rms = 0

    def init_rms():
        """Reset plot data and state before starting an animation pass."""
        global block_idx_rms
        block_idx_rms = 0
        rms_values[:] = np.nan
        bridge_rms.reset()
        line_rms.set_ydata(rms_values)
        return (line_rms,)

    def update_rms(_frame):
        """Update function invoked by FuncAnimation on each frame."""
        global block_idx_rms
        updated = False
        for block in bridge_rms.drain():
            if block_idx_rms < n_blocks:
                rms_values[block_idx_rms] = process_rms(block)
                block_idx_rms += 1
                updated = True

        if updated:
            line_rms.set_ydata(rms_values)
        return (line_rms,)

    ani_rms = animation.FuncAnimation(
        fig_rms,
        update_rms,
        init_func=init_rms,
        frames=stream,
        interval=1000 * blocksize / stream.samplerate,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

# Sphinx-Gallery automatically captures the animation and renders it as an HTML5 video


# %%
# Part 2: Example 2 — Sliding Analysis Window and Resampling (`StreamBuffer`)
# ---------------------------------------------------------------------------
# While computing features block-by-block works for simple metrics like RMS energy,
# real-world music information retrieval (MIR) often introduces two additional challenges:
#
# 1. **Latency vs. Frequency Resolution**: Low-latency interactive applications require
#    small audio blocks (e.g. 512 samples / ~23 ms). However, frequency-domain features
#    such as the Short-Time Fourier Transform (STFT), spectral centroid, or chroma require
#    a longer observation window (e.g. 1024 samples / ~93 ms) to obtain good frequency resolution.
#    To resolve this without introducing latency, we maintain a **sliding analysis window**
#    where each incoming block advances the buffer by a smaller hop size.
# 2. **Sample-Rate Mismatch**: Audio capture devices often operate at hardware sample rates
#    like 44.1 kHz or 48 kHz, while downstream models or acoustic features frequently
#    expect standard rates like 22.05 kHz or 11.025 kHz.
#
# Instead of writing custom ring buffers and resamplers, Librosa provides `librosa.StreamBuffer`
# (or `librosa.util.streaming.StreamBuffer`). It manages the sliding window, downmixes multi-channel audio,
# and automatically streams chunks through `soxr` resampling on-the-fly. Setting ``warm_up=True``
# ensures the first analysis window is produced immediately without startup latency.


def process_spectral_centroid(analysis_window, sr):
    """Compute spectral centroid from the sliding analysis window using librosa.

    Parameters
    ----------
    analysis_window : np.ndarray
        Window of audio samples of length `n_fft`.
    sr : float
        Sample rate of the audio buffer.

    Returns
    -------
    centroid : float
        Spectral centroid frequency in Hz.
    """
    sc = librosa.feature.spectral_centroid(
        y=analysis_window,
        sr=sr,
        n_fft=len(analysis_window),
        hop_length=len(analysis_window),
        center=False,
    )
    return float(sc[0, 0])


# %%
# Animating Spectral Centroid Tracking
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this example, we downsample incoming 22,050 Hz audio to 11,025 Hz, maintain an
# analysis window of 1,024 samples (at 11,025 Hz), advance by 256 samples per hop,
# and animate `librosa.feature.spectral_centroid` in real time:

target_sr = 11025
frame_length = 1024
hop_length = 512  # At native 22050 Hz, 512 samples corresponds to 256 samples at 11025 Hz

bridge_sc = librosa.util.StreamBridge(orig_sr=22050, channels=1)

with VirtualInputStream(
    filename, blocksize=hop_length, callback=bridge_sc
) as stream:
    sr = stream.samplerate
    duration = stream.total_frames / sr
    target_hop = int(hop_length * target_sr / sr)
    n_blocks_sc = int(np.ceil(stream.total_frames / hop_length))
    times_sc = np.arange(n_blocks_sc) * (hop_length / sr)
    centroid_values = np.full(n_blocks_sc, np.nan)

    # StreamBuffer automatically handles sliding window overlap and soxr resampling
    stream_buf = librosa.StreamBuffer(
        frame_length=frame_length,
        hop_length=target_hop,
        block_length=1,
        orig_sr=sr,
        sr=target_sr,
        warm_up=True,
    )

    fig_sc, ax_sc = plt.subplots(figsize=(8, 3))
    (line_sc,) = ax_sc.plot(times_sc, centroid_values, color="#bd1f01", lw=2)
    ax_sc.set_ylim(500, 4000)
    ax_sc.set_xlim(0, duration)
    ax_sc.set_title("Live Stream Processing: Spectral Centroid (StreamBuffer)")
    ax_sc.set_xlabel("Time (seconds)")
    ax_sc.set_ylabel("Frequency (Hz)")
    ax_sc.grid(True, linestyle="--", alpha=0.5)

    block_idx_sc = 0

    def init_sc():
        """Reset plot data and buffer state before starting an animation pass."""
        global block_idx_sc
        block_idx_sc = 0
        centroid_values[:] = np.nan
        stream_buf.reset()
        bridge_sc.reset()
        line_sc.set_ydata(centroid_values)
        return (line_sc,)

    def update_centroid(_frame):
        """Update function for spectral centroid tracking."""
        global block_idx_sc
        updated = False
        for chunk in bridge_sc.drain():
            for block in stream_buf.push(chunk):
                if block_idx_sc < n_blocks_sc:
                    centroid_values[block_idx_sc] = process_spectral_centroid(
                        block, sr=target_sr
                    )
                    block_idx_sc += 1
                    updated = True

        if updated:
            line_sc.set_ydata(centroid_values)
        return (line_sc,)

    ani_sc = animation.FuncAnimation(
        fig_sc,
        update_centroid,
        init_func=init_sc,
        frames=stream,
        interval=1000 * hop_length / stream.samplerate,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )


# %%
# Part 3: Asynchronous Audio Streaming (`async_capture` and `astream`)
# --------------------------------------------------------------------
# In addition to real-time UI animation loops, modern Python applications
# (such as WebSockets, asyncio servers, and background workers) often rely on
# `asyncio` for non-blocking I/O.
#
# Librosa provides first-class asynchronous streaming support:
#
# 1. `librosa.util.async_capture`: An asynchronous context manager wrapping
#    `sounddevice.InputStream` and yielding a `StreamBridge`.
# 2. `librosa.astream`: An asynchronous generator that consumes chunks from any
#    asynchronous iterable (or `StreamBridge`) and yields framed, resampled analysis blocks:
#
# .. code:: python
#
#    import asyncio
#    import librosa
#
#    async def process_live_audio():
#        # Capture audio asynchronously from the default microphone
#        async with librosa.util.async_capture(samplerate=44100, blocksize=512) as stream:
#            async for block in librosa.astream(
#                stream,
#                frame_length=2048,
#                hop_length=512,
#                sr=22050,
#            ):
#                # Non-blocking analysis on 22.05 kHz resampled blocks
#                centroid = librosa.feature.spectral_centroid(y=block, sr=22050)
#                print(f"Spectral Centroid: {centroid[0, 0]:.1f} Hz")
#
# Similarly, for synchronous pipelines (such as background threads or command-line scripts),
# `librosa.stream` can iterate directly over a `StreamBridge`:
#
# .. code:: python
#
#    import sounddevice as sd
#
#    bridge = librosa.util.StreamBridge(orig_sr=44100)
#    with sd.InputStream(samplerate=44100, blocksize=512, callback=bridge):
#        for block in librosa.stream(bridge, frame_length=2048, hop_length=512, sr=22050):
#            centroid = librosa.feature.spectral_centroid(y=block, sr=22050)


# %%
# Conclusion: Connecting to a Live Audio Device
# ---------------------------------------------
# The major strength of separating the audio I/O layer (`StreamBridge`) from the
# analysis buffer (`StreamBuffer`) is that switching from simulation to a live microphone
# requires changing **only the stream source**.
#
# To connect to a real microphone or audio interface on your computer, replace
# `VirtualInputStream` with `sounddevice.InputStream`:
#
# .. code:: python
#
#    import sounddevice as sd
#
#    # 1. Query available capture hardware
#    print(sd.query_devices())
#
#    # 2. Create the Librosa stream bridge
#    bridge = librosa.util.StreamBridge(orig_sr=22050, channels=1)
#
#    # 3. Open an InputStream connected to your microphone
#    with sd.InputStream(
#        samplerate=22050,
#        blocksize=1024,
#        channels=1,
#        dtype="float32",
#        callback=bridge,
#    ):
#        # 4. In an interactive Python environment or script:
#        ani = animation.FuncAnimation(fig_rms, update_rms, interval=30, blit=True)
#        plt.show()
#
# Practical considerations for live streaming
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# When running on physical hardware, keep the following best practices in mind:
#
# - **Bounded vs. indefinite streams**: In this demonstration, we streamed an audio
#   file with a fixed duration, allowing the plot to map directly onto ``[0, duration]``
#   in sync with the player. For continuous, indefinite streams (like an open microphone),
#   you can instead plot a rolling window of recent time (e.g. the last 5 seconds)
#   by scrolling the line array with ``np.roll``.
# - **Block size and latency**: Smaller block sizes (e.g. 256-512 samples) reduce capture
#   latency but increase callback frequency and CPU overhead. If you experience dropouts
#   or buffer overruns (indicated by `status.input_overflow`), increase `blocksize`.
# - **Computation budget**: All processing within each consumer iteration must complete
#   faster than the real-time block duration (:math:`\\text{blocksize} / \\text{samplerate}`).
# - **Feature coupling**: When computing features like MFCCs incrementally, remember that
#   parameters like ``top_db`` or callable ``ref`` can introduce inter-frame dependencies.
#   Refer to the :ref:`sphx_glr_auto_tutorials_03-advanced_plot_mfcc_stream.py` tutorial
#   for details on frame-local configurations.
