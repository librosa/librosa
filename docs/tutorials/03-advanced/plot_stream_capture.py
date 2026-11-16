# coding: utf-8
# Code source: Brian McFee
# License: ISC
"""
======================
Live stream processing
======================

This tutorial illustrates how to perform real-time stream processing and visualization
on signals captured from an audio device.

In real-world applications, streaming audio processing requires coordinating three
components:

1. **Audio I/O & Callbacks (Producer)**: High-priority system callbacks from the audio hardware
   (such as :class:`sounddevice.InputStream` or PortAudio) execute on a dedicated OS thread
   under hard real-time deadlines and must never block. :class:`librosa.StreamBridge` acts directly
   as the audio callback, receiving raw audio chunks, normalizing channel geometries and data types,
   and buffering them in a thread-safe queue.
2. **Buffering, Overlap, & Resampling (Analysis)**: Feature extractors often require observation windows that
   differ in size or sample rate from the capture device. Calling :meth:`librosa.StreamBridge.stream`
   accumulates incoming chunks into overlapping analysis windows,
   downmixes multi-channel audio, and handles on-the-fly streaming sample-rate conversion via ``soxr``.
3. **Display or Processing (Consumer)**: Downstream consumers (such as GUI animation loops,
   feature extraction pipelines, or worker threads) iterate over the yielded analysis blocks
   without stalling the audio callback.
"""

# %%
# Imports and Audio Source
# ------------------------
# We import Matplotlib for animation, ``soundfile`` for simulated hardware I/O,
# and ``librosa`` for signal processing and streaming utilities.

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
# Real-Time Audio Architecture
# ----------------------------
# Before diving into code, let's examine how live capture works under the hood
# and how :class:`~librosa.StreamBridge` connects the hardware to Librosa:
#
# 1. Audio Callbacks & Real-Time Deadlines
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Audio capture libraries like :mod:`sounddevice` use an event-driven **push model**:
# the hardware Analog-to-Digital Converter (ADC) fills small audio buffers on a dedicated
# OS real-time thread, invoking an **audio callback function** every ``blocksize`` frames.
#
# Callbacks operate under hard deadlines: if a callback blocks (e.g. performing heavy FFTs,
# file I/O, or memory allocations), the hardware buffer overflows (**xrun**), causing
# permanent data loss and audible dropouts.
#
# 2. The Role of StreamBridge
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Python data analysis code is **pull-based** (iterating over a generator in a ``for`` loop).
# :class:`librosa.StreamBridge` bridges this gap:
#
# - **Producer (Audio Callback)**: Passed directly as ``callback=bridge`` to :class:`sounddevice.InputStream`.
#   On the real-time thread, it normalizes incoming data (converting integer PCM like 16-bit to standard float32
#   and transposing channels to canonical Librosa layout) and pushes chunks into an internal queue without blocking.
# - **Consumer (Python Generator)**: In downstream threads, calling :meth:`bridge.stream(...) <librosa.StreamBridge.stream>`
#   yields complete analysis blocks safely.
# - **Latency & Jitter**: The internal queue absorbs transient CPU spikes. Configurable overflow policies
#   (such as ``overflow_policy="drop_oldest"``) prevent unbounded latency if the consumer falls behind.
#
# 3. Hardware Rate vs. Analysis Rate
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In live streaming, the **hardware capture rate** and the **downstream analysis rate** are decoupled:
#
# - **Hardware capture rate**: Sound capture devices operate natively at
#   fixed rates (typically 44,100 Hz or 48,000 Hz). Both :class:`sounddevice.InputStream` and
#   :class:`librosa.StreamBridge` **must match the device's native rate**.
# - **Analysis rate**: Downstream algorithms or models often expect a lower rate (e.g. 22,050 Hz
#   or 16,000 Hz). Resampling is *not* done in the callback; it is performed on the fly when calling
#   :meth:`bridge.stream(..., sr=target_sr) <librosa.StreamBridge.stream>`.
#
# 4. Sliding Windows & Overlapping Frames
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Capture devices push small buffers (e.g. 512 samples) to minimize latency, but spectral transforms
# require longer observation windows (``frame_length``) stepped forward by ``hop_length``.
# :meth:`librosa.StreamBridge.stream` buffers incoming audio and yields complete overlapping frames.
# Setting ``warm_up=True`` pre-fills the initial overlap with silence so that the very first hop produces
# an analysis frame without startup lag.
#
#
# Simulating an Input Device
# --------------------------
# To make the examples below self-contained and reproducible without requiring a
# physical microphone, we define a small helper class (``VirtualInputStream``) that
# reads fixed-size blocks from an audio file and invokes ``callback`` with the same
# signature as :class:`sounddevice.InputStream`.
#
# In ``__enter__``, it slices the audio file into chunks of size ``blocksize`` and passes them to
# ``callback(block, len(block), None, None)``. In ``__exit__``, it closes the callback to signal
# stream completion.


class VirtualInputStream:
    """Simulate :class:`sounddevice.InputStream` from an audio file.

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
            if samplerate is not None and af.samplerate != samplerate:
                raise ValueError(f"Sample rate {samplerate} does not match input file's sample rate {af.samplerate}")
            self.samplerate = af.samplerate
            self.total_frames = af.frames

    def __enter__(self):
        with sf.SoundFile(self.filepath) as af:
            for block in af.blocks(
                blocksize=self.blocksize, dtype=self.dtype, always_2d=True
            ):
                if self.callback is not None:
                    self.callback(block, len(block), None, None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.callback, "close"):
            self.callback.close()


# %%
# Encapsulating Live Visualization
# --------------------------------
# To prevent animation setup from obscuring signal processing logic,
# we encapsulate real-time sparkline plotting into a helper class:


class LiveSparkline:
    """Encapsulates Matplotlib animation setup and real-time line plotting."""

    def __init__(self, times, ylim, title, ylabel):
        self.times = times
        self.data = np.full(len(times), np.nan)
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        (self.line,) = self.ax.plot(self.times, self.data, lw=2, label="Live")
        self.ax.set_ylim(*ylim)
        self.ax.set_xlim(times[0], times[-1])
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle="-.", alpha=0.5)

    def reset(self):
        """Reset plot data and pointer for a clean animation replay pass."""
        self.idx = 0
        self.data[:] = np.nan
        self.line.set_ydata(self.data)
        return (self.line,)

    def append(self, value):
        """Append an extracted feature scalar to the visual trace."""
        if self.idx < len(self.data):
            self.data[self.idx] = value
            self.idx += 1
            self.line.set_ydata(self.data)


# %%
# Example 1: Instantaneous Block Processing (RMS Energy)
# ------------------------------------------------------
# For instantaneous energy metrics like Root-Mean-Square (RMS), we can process
# incoming audio block-by-block with non-overlapping frames (``frame_length=blocksize, hop_length=blocksize``).
#
# Because our audio source is recorded at 22,050 Hz and no sample-rate conversion is needed,
# both :class:`VirtualInputStream` and :class:`~librosa.StreamBridge` use ``sr=22050``,
# and ``bridge_rms.stream(...)`` runs directly at the native rate without resampling.


def process_rms(y):
    """Compute instantaneous RMS energy for a single audio block."""
    rms = librosa.feature.rms(
        y=y, frame_length=len(y), hop_length=len(y), center=False
    )
    return float(rms[0, 0])


# %%
# Visualizing the Live RMS Sparkline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We pass ``bridge_rms.stream(...)`` directly as the ``frames`` generator to
# :class:`matplotlib.animation.FuncAnimation`:

blocksize = 1024

with (
    librosa.StreamBridge(sr=22050, channels=1) as bridge_rms,
    VirtualInputStream(filename, blocksize=blocksize, callback=bridge_rms) as stream,
):
    sr = stream.samplerate
    n_blocks = int(np.ceil(stream.total_frames / blocksize))
    times = np.arange(n_blocks) * (blocksize / sr)

    sparkline_rms = LiveSparkline(
        times,
        ylim=(0, 0.4),
        title="Live Stream: Instantaneous RMS Energy",
        ylabel="RMS Amplitude",
    )

    def update_rms(block):
        sparkline_rms.append(process_rms(block))
        return (sparkline_rms.line,)

    ani_rms = animation.FuncAnimation(
        sparkline_rms.fig,
        update_rms,
        init_func=sparkline_rms.reset,
        frames=bridge_rms.stream(
            block_length=1,
            frame_length=blocksize,
            hop_length=blocksize,
            sr=None,
        ),
        interval=1000 * blocksize / sr,
        blit=True,
        cache_frame_data=False,
    )

    # And for comparison, visualize the same analysis computed from a
    # the full file
    y, sr = librosa.load(filename, sr=22050)
    full_rms = librosa.feature.rms(y=y,
                                   center=False,
                                   frame_length=blocksize,
                                   hop_length=blocksize
                                   )
    times = librosa.times_like(full_rms, sr=sr, hop_length=blocksize)
    sparkline_rms.ax.plot(times, full_rms[0], linestyle="--", label="Full")
    sparkline_rms.ax.legend(loc="upper right")

# %%
# Example 2: Overlapping Windows & Online Resampling
# --------------------------------------------------
# Spectral features (such as :func:`librosa.feature.spectral_centroid` or STFT)
# introduce two additional streaming requirements:
#
# 1. **Sliding Window Overlap**: Low-latency capture delivers small chunks (e.g. 512 samples / ~23 ms),
#    but spectral resolution requires wider observation windows (e.g. 1024 samples).
#    :meth:`~librosa.StreamBridge.stream` accumulates incoming chunks into overlapping windows.
# 2. **Stream resampling**: Hardware often captures at a higher rate (here 22,050 Hz) than
#    analysis requires (here 11,025 Hz). Passing ``sr=11025`` to :meth:`~librosa.StreamBridge.stream`
#    resamples chunks on the fly via ``soxr`` before windowing.
#
# Setting ``warm_up=True`` pre-fills the initial overlap with silence (zeros) so that
# the first analysis frame is produced immediately upon receiving the first hop.


def process_spectral_centroid(analysis_window, sr):
    """Compute spectral centroid from the sliding analysis window."""
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
# Here we downsample 22,050 Hz capture audio to 11,025 Hz, maintain an analysis window
# of 1,024 samples (at 11,025 Hz), and step forward by 256 samples per hop:

target_sr = 11025
frame_length = 1024
hop_length = 512  # 512 samples at 22050 Hz corresponds to 256 samples at 11025 Hz

with (
    librosa.StreamBridge(sr=22050, channels=1) as bridge_sc,
    VirtualInputStream(filename, blocksize=hop_length, callback=bridge_sc) as stream,
):
    sr = stream.samplerate
    target_hop = int(hop_length * target_sr / sr)
    n_blocks_sc = int(np.ceil(stream.total_frames / hop_length))
    times_sc = np.arange(n_blocks_sc) * (hop_length / sr)

    sparkline_sc = LiveSparkline(
        times_sc,
        ylim=(500, 4000),
        title="Live Stream: Spectral Centroid",
        ylabel="Frequency (Hz)",
    )

    def update_centroid(block):
        centroid_val = process_spectral_centroid(block, sr=target_sr)
        sparkline_sc.append(centroid_val)
        return (sparkline_sc.line,)

    ani_sc = animation.FuncAnimation(
        sparkline_sc.fig,
        update_centroid,
        init_func=sparkline_sc.reset,
        frames=bridge_sc.stream(
            block_length=1,
            frame_length=frame_length,
            hop_length=target_hop,
            sr=target_sr,
            warm_up=True,
        ),
        interval=1000 * hop_length / sr,
        blit=True,
        cache_frame_data=False,
    )

# %%
# In the example above, we retain `block_length=1` to minimize latency: each
# frame is processed as soon as possible.  However, it is also possible to
# process multiple overlapping frames by increasing `block_length`, exactly as
# in :func:`librosa.stream`.  This is particularly useful when working with
# algorithms that require multiple frames' worth of input to process.

# %%
# Connecting to a Physical Microphone
# -----------------------------------
# To run this pipeline on a physical microphone with :mod:`sounddevice`,
# query the input device's default sample rate, configure both :class:`sounddevice.InputStream`
# and :class:`librosa.StreamBridge` at that rate, and specify your desired analysis rate in
# :meth:`~librosa.StreamBridge.stream`:
#
# .. code:: python
#
#     import sounddevice as sd
#     import librosa
#
#     # 1. Query the device's native sampling rate (typically 44100 or 48000 Hz)
#     device_info = sd.query_devices(kind="input")
#     device_sr = int(device_info["default_samplerate"])
#
#     # 2. Select target analysis rate with an integer ratio (e.g. 2:1 or 3:1)
#     target_sr = 22050 if device_sr % 22050 == 0 else (device_sr // 2)
#
#     # 3. Open the bridge and input stream at the device's native rate
#     with (
#         librosa.StreamBridge(sr=device_sr, channels=1) as bridge,
#         sd.InputStream(
#             samplerate=device_sr, blocksize=512, channels=1, callback=bridge
#         ),
#     ):
#         # 4. Stream analysis blocks, converting from device_sr to target_sr on the fly
#         for block in bridge.stream(
#             block_length=1, frame_length=1024, hop_length=512, sr=target_sr
#         ):
#             rms = librosa.feature.rms(
#                 y=block, frame_length=1024, hop_length=512, center=False
#             )
#
# Practical Tips for Live Streams
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# - **Integer resampling ratios**: When converting sample rates in :meth:`~librosa.StreamBridge.stream`,
#   ensure ``(block_length * hop_length * orig_sr) / sr`` evaluates to an exact integer (such as 44.1 kHz -> 22.05 kHz
#   or 48 kHz -> 24 kHz / 16 kHz).
# - **Latency vs. CPU load**: Smaller hardware block sizes (e.g. 256 or 512 frames) minimize latency,
#   while larger sizes reduce callback frequency and CPU overhead.
# - **Queue overflow handling**: If downstream processing experiences intermittent stalls, configure
#   :class:`~librosa.StreamBridge` with ``overflow_policy="drop_oldest"`` to operate as a fixed-latency
#   ring buffer and discard stale audio rather than accumulating latency.
# - **Continuous rolling displays**: For indefinite microphone streaming, maintain a rolling display
#   buffer (e.g. the last 5 seconds) by shifting visual traces with :func:`numpy.roll`.
# - **Frame-local feature extraction**: Keep streaming features consistent with whole-file analysis by
#   ensuring computations are context-independent (e.g., using ``center=False`` and avoiding dynamic global
#   normalizations like ``top_db``; see :ref:`sphx_glr_auto_tutorials_03-advanced_plot_mfcc_stream.py`).

