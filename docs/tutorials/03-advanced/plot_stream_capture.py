# coding: utf-8
# Code source: Brian McFee
# License: ISC
"""
======================
Live stream processing
======================

This tutorial illustrates how to perform real-time stream processing and visualization
on signals captured from an audio device.

In real-world applications, streaming audio processing decouples into three independent
responsibilities:

1. **Audio I/O (Producer)**: High-priority system callbacks from the audio hardware
   (such as :class:`sounddevice.InputStream` or PortAudio) execute on a separate OS thread
   and must never block. :class:`librosa.util.StreamBridge` acts directly as the
   audio callback, receiving raw audio blocks, normalizing channel geometries and data types,
   and buffering them in a thread-safe queue.
2. **Buffering & DSP (Analysis)**: Feature extractors often require observation windows that
   differ in size or sample rate from the capture device. Calling :meth:`librosa.util.StreamBridge.stream`
   (or :func:`librosa.stream`, or :meth:`~librosa.util.StreamBridge.astream` for ``asyncio``)
   accumulates incoming chunks into overlapping analysis windows, downmixes multi-channel audio,
   and handles on-the-fly streaming sample-rate conversion via ``soxr``.
3. **Display or Processing (Consumer)**: Downstream consumers (such as GUI animation threads,
   event loops, or async workers) iterate over the yielded analysis blocks without stalling
   the audio callback.

Live microphone capture pattern
-------------------------------
When connecting to a real microphone with :mod:`sounddevice`, the complete capture and analysis
pattern requires only a few lines:

.. code:: python

    import sounddevice as sd
    import librosa

    # 1. Create a stream bridge and open the hardware input stream
    with (
        librosa.util.StreamBridge(sr=22050, channels=1) as bridge,
        sd.InputStream(samplerate=22050, blocksize=512, channels=1, callback=bridge),
    ):
        # 2. Stream blocks (or equivalently, librosa.stream(bridge, ...))
        for block in bridge.stream(
            block_length=1, frame_length=1024, hop_length=512
        ):
            # Compute features in real time
            rms = librosa.feature.rms(y=block, center=False)

Similarly, asynchronous applications can capture and process audio concurrently using
:meth:`librosa.util.StreamBridge.astream` (or :func:`librosa.astream`):

.. code:: python

    async with librosa.util.StreamBridge(sr=22050, channels=1) as bridge:
        with sd.InputStream(samplerate=22050, blocksize=512, channels=1, callback=bridge):
            async for block in bridge.astream(
                block_length=1, frame_length=1024, hop_length=512
            ):
                rms = librosa.feature.rms(y=block, center=False)
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
# Simulating an Input Device
# --------------------------
# To make the examples below self-contained and reproducible without requiring a
# physical microphone, we define a small helper class (``VirtualInputStream``) that
# reads fixed-size blocks from an audio file and invokes ``callback`` with the same
# signature as :class:`sounddevice.InputStream`.
#
# .. note::
#     To run any of the examples below with a physical microphone, replace
#     ``VirtualInputStream(filename, ...)`` with :class:`sounddevice.InputStream`.


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
            self.samplerate = samplerate or af.samplerate
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
# To prevent Matplotlib animation setup and array manipulation from obscuring
# the signal processing logic, we encapsulate real-time sparkline plotting into
# a helper class:


class LiveSparkline:
    """Encapsulates Matplotlib animation setup and real-time line plotting."""

    def __init__(self, times, ylim, title, ylabel):
        self.times = times
        self.data = np.full(len(times), np.nan)
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        (self.line,) = self.ax.plot(self.times, self.data, lw=2)
        self.ax.set_ylim(*ylim)
        self.ax.set_xlim(times[0], times[-1])
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle="--", alpha=0.5)

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
# In this first example, incoming blocks are processed with non-overlapping frames
# (``frame_length=blocksize, hop_length=blocksize``) to compute instantaneous
# Root-Mean-Square (RMS) energy using :func:`librosa.feature.rms`.
#
# The architecture cleanly separates each component:
#
# - **I/O**: :class:`librosa.util.StreamBridge` receives incoming blocks from the device.
# - **Streaming & Analysis**: :meth:`librosa.util.StreamBridge.stream` yields each block to ``process_rms``.
# - **Display**: ``LiveSparkline.append`` plots each value forward in time.


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
    librosa.util.StreamBridge(sr=22050, channels=1) as bridge_rms,
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
        ),
        interval=1000 * blocksize / sr,
        blit=True,
        cache_frame_data=False,
    )


# %%
# Example 2: Overlapping Windows & Online Resampling
# --------------------------------------------------
# While instantaneous metrics like RMS can operate on non-overlapping blocks, frequency-domain
# features (such as :func:`librosa.feature.spectral_centroid`, STFT, or chroma)
# introduce two additional requirements:
#
# 1. **Sliding Window Overlap**: Low-latency capture delivers small chunks (e.g. 512 samples / ~23 ms),
#    but frequency transforms require longer observation windows (e.g. 1024 samples / ~93 ms)
#    for spectral resolution. We need a sliding observation window where each incoming chunk shifts
#    the buffer forward by a smaller hop size.
# 2. **Online Resampling**: Hardware inputs frequently capture at 44.1 kHz or 48 kHz, while downstream
#    models expect 22.05 kHz or 11.025 kHz.
#
# :meth:`librosa.util.StreamBridge.stream` (and :func:`librosa.stream`) handles both requirements:
#
# - It accumulates incoming chunks and yields complete analysis blocks matching ``frame_length`` and ``hop_length``.
# - It streams chunks through ``soxr`` resampling on-the-fly when ``sr`` differs from ``bridge.sr``.
# - Setting ``warm_up=True`` pre-fills the initial observation window with silence so the first analysis
#   frame is produced immediately without startup latency.


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
    librosa.util.StreamBridge(sr=22050, channels=1) as bridge_sc,
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
# Practical Considerations for Live Hardware Streams
# ---------------------------------------------------
# When deploying live stream processing on physical hardware, keep the following
# best practices in mind:
#
# - **Continuous streams with rolling history**: In this demonstration, we processed a
#   finite recording with a known duration. For open-ended live microphone streams,
#   maintain a rolling window of recent history (such as the last 5 seconds) by shifting
#   the display array with ``numpy.roll``.
# - **Block size and latency**: Smaller block sizes reduce latency at the expense of higher
#   CPU callback frequency. If buffer overruns occur (indicated by status flags in the audio
#   callback), increase ``blocksize``.
# - **Frame-local feature extraction**: To keep streaming features consistent with whole-file
#   analysis, ensure that frame-level computations are context-independent (e.g. using
#   ``center=False`` and avoiding dynamic global parameters like ``top_db`` or callable ``ref``).
#   For further discussion, refer to :ref:`sphx_glr_auto_tutorials_03-advanced_plot_mfcc_stream.py`.
