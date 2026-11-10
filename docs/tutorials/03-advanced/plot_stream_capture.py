# coding: utf-8
# Code source: Brian McFee
# License: ISC
"""
======================
Live stream processing
======================

This tutorial illustrates how to perform real-time stream processing and visualization
on signals captured from an audio device.

Real-time audio processing typically decouples into three independent responsibilities:

1. **Audio I/O (Producer)**: High-priority system callbacks from the audio hardware
   (such as `sounddevice.InputStream` or PortAudio) must never block or perform heavy
   computation. [`librosa.util.StreamBridge`](file:///home/bmcfee/git/librosa/librosa/util/streaming.py#L354) acts directly as the
   audio callback, receives raw audio blocks, normalizes channel geometries and data types,
   and queues them safely.
2. **Buffering & DSP (Analysis)**: Feature extractors often require observation windows that
   differ in size or sample rate from the capture device. [`librosa.util.StreamBuffer`](file:///home/bmcfee/git/librosa/librosa/util/streaming.py#L52)
   accumulates incoming chunks into overlapping analysis windows, downmixes multi-channel audio,
   and handles on-the-fly streaming sample-rate conversion via `soxr`.
3. **Display (Consumer)**: Downstream consumers (such as GUI threads or Matplotlib animations)
   pull ready blocks and update the user interface without holding up the audio pipeline.

Aside: virtual devices for this documentation
---------------------------------------------
This tutorial document is compiled automatically in a headless CI/CD environment without
access to physical audio capture hardware.

To illustrate these concepts, we construct a virtual audio device (`VirtualInputStream`)
that streams fixed-size chunks from Librosa's packaged example recordings through the exact
same callback interface used by `sounddevice.InputStream`.

.. note::
    `VirtualInputStream` is only needed for offline rendering. When running live on your
    computer, simply swap it for `sounddevice.InputStream`.


This tutorial covers three progression steps:

- **Example 1: Instantaneous Block Processing**: Ingesting blocks with `StreamBridge` and computing RMS energy.
- **Example 2: Overlapping Buffers & Streaming Resampling**: Using `StreamBuffer` to track spectral centroid across sliding analysis windows.
- **Example 3: Modern Asynchronous Streaming**: Using `librosa.util.async_capture` and `librosa.astream`.
"""

# %%
# Imports and Audio Source
# ------------------------
# We import Matplotlib for animation, `soundfile` for simulated hardware I/O,
# and `librosa` for signal processing and streaming utilities.

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
# Simulating the Audio Device
# ---------------------------
# In an automated documentation build, no physical microphone is available.
# `VirtualInputStream` mimics the context-manager interface of `sounddevice.InputStream`,
# reading chunks from an audio file and invoking the stream callback with PortAudio-compatible
# arguments:


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
# Encapsulating Live Visualization
# --------------------------------
# To prevent visualization boilerplate from cluttering our signal processing code,
# we encapsulate the plotting canvas, sparkline trace, and animation state reset
# into a self-contained helper class.
#
# This completely decouples plotting concerns from audio I/O and feature analysis:


class LiveSparkline:
    """Encapsulates Matplotlib animation setup and real-time line plotting."""

    def __init__(self, times, ylim, title, ylabel, color):
        self.times = times
        self.data = np.full(len(times), np.nan)
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        (self.line,) = self.ax.plot(self.times, self.data, color=color, lw=2)
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
# Part 1: Example 1 — Basic Stream Processing (RMS Energy)
# --------------------------------------------------------
# In our first example, we process incoming blocks independently to compute
# instantaneous Root-Mean-Square (RMS) energy.
#
# Notice the clear separation of roles:
#
# 1. **I/O**: `librosa.util.StreamBridge` serves directly as the hardware callback
#    and buffers incoming blocks.
# 2. **Analysis**: `process_rms` computes the energy on each raw audio block.
# 3. **Display**: `sparkline.append` draws the computed point on the timeline.


def process_rms(buffer):
    """Compute instantaneous RMS energy for a single audio block."""
    y = buffer.squeeze()
    rms = librosa.feature.rms(
        y=y, frame_length=len(y), hop_length=len(y), center=False
    )
    return float(rms[0, 0])


# %%
# Running the RMS Animation
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# On each tick of `FuncAnimation`, `bridge.drain()` retrieves whatever blocks
# arrived since the last frame without blocking the audio callback:

blocksize = 1024

# 1. Set up audio I/O bridge
bridge_rms = librosa.util.StreamBridge(orig_sr=22050, channels=1)

with VirtualInputStream(
    filename, blocksize=blocksize, callback=bridge_rms
) as stream:
    sr = stream.samplerate
    duration = stream.total_frames / sr
    n_blocks = int(np.ceil(stream.total_frames / blocksize))
    times = np.arange(n_blocks) * (blocksize / sr)

    # 2. Set up visualization
    sparkline_rms = LiveSparkline(
        times,
        ylim=(0, 0.4),
        title="Live Stream: Instantaneous RMS Energy",
        ylabel="RMS Amplitude",
        color="#3f90da",
    )

    # 3. Consumer animation callbacks
    def init_rms():
        bridge_rms.reset()
        return sparkline_rms.reset()

    def update_rms(_frame):
        for chunk in bridge_rms.drain():
            sparkline_rms.append(process_rms(chunk))
        return (sparkline_rms.line,)

    ani_rms = animation.FuncAnimation(
        sparkline_rms.fig,
        update_rms,
        init_func=init_rms,
        frames=stream,
        interval=1000 * blocksize / stream.samplerate,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

# Sphinx-Gallery captures the animation and renders it as an HTML5 video


# %%
# Part 2: Example 2 — Overlapping Buffers & Resampling (`StreamBuffer`)
# ---------------------------------------------------------------------
# While instantaneous metrics like RMS work block-by-block, spectral and music
# information retrieval (MIR) features introduce two common requirements:
#
# 1. **Sliding Window Overlap**: Low-latency capture delivers small chunks (e.g. 512 samples / ~23 ms),
#    but frequency transforms (like STFT or spectral centroid) require longer analysis windows
#    (e.g. 1024 samples / ~93 ms) for frequency resolution. We need a sliding observation window
#    where each incoming chunk shifts the buffer forward by a smaller hop size.
# 2. **Online Resampling**: Hardware inputs frequently capture at 44.1 kHz or 48 kHz, while downstream
#    models expect 22.05 kHz or 11.025 kHz.
#
# `librosa.util.StreamBuffer` encapsulates both requirements:
#
# - It accumulates incoming chunks and yields complete analysis blocks matching `frame_length` and `hop_length`.
# - It streams chunks through `soxr` resampling on-the-fly.
# - Setting ``warm_up=True`` pre-fills the initial observation window so the first frame produces output immediately without startup delay.


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

# 1. Set up audio I/O bridge
bridge_sc = librosa.util.StreamBridge(orig_sr=22050, channels=1)

with VirtualInputStream(
    filename, blocksize=hop_length, callback=bridge_sc
) as stream:
    sr = stream.samplerate
    duration = stream.total_frames / sr
    target_hop = int(hop_length * target_sr / sr)
    n_blocks_sc = int(np.ceil(stream.total_frames / hop_length))
    times_sc = np.arange(n_blocks_sc) * (hop_length / sr)

    # 2. Set up sliding window DSP buffer with online resampling
    stream_buf = librosa.util.StreamBuffer(
        frame_length=frame_length,
        hop_length=target_hop,
        block_length=1,
        orig_sr=sr,
        sr=target_sr,
        warm_up=True,
    )

    # 3. Set up visualization
    sparkline_sc = LiveSparkline(
        times_sc,
        ylim=(500, 4000),
        title="Live Stream: Spectral Centroid (StreamBuffer)",
        ylabel="Frequency (Hz)",
        color="#bd1f01",
    )

    # 4. Consumer animation callbacks
    def init_sc():
        stream_buf.reset()
        bridge_sc.reset()
        return sparkline_sc.reset()

    def update_centroid(_frame):
        for chunk in bridge_sc.drain():
            for block in stream_buf.push(chunk):
                centroid_val = process_spectral_centroid(block, sr=target_sr)
                sparkline_sc.append(centroid_val)
        return (sparkline_sc.line,)

    ani_sc = animation.FuncAnimation(
        sparkline_sc.fig,
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
# Beyond UI animation loops, modern Python architectures (such as WebSocket servers,
# FastAPI, or background worker threads) often process streaming audio asynchronously.
#
# Librosa provides first-class asynchronous streaming primitives:
#
# 1. `librosa.util.async_capture`: An asynchronous context manager wrapping
#    `sounddevice.InputStream` to yield a `StreamBridge`.
# 2. `librosa.astream`: An asynchronous generator that consumes any async stream
#    and yields framed, resampled analysis blocks:
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
# In synchronous batch scripts or worker threads, `librosa.stream` can also
# consume directly from a `StreamBridge`:
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
# Conclusion: Connecting to Physical Hardware
# -------------------------------------------
# Because the audio I/O layer (`StreamBridge`) is completely decoupled from the analysis
# buffer (`StreamBuffer`) and display (`LiveSparkline`), switching from offline simulation
# to a physical microphone requires changing **only the input device**:
#
# .. code:: python
#
#    import sounddevice as sd
#
#    # 1. Check available hardware devices
#    print(sd.query_devices())
#
#    # 2. Create the stream bridge
#    bridge = librosa.util.StreamBridge(orig_sr=22050, channels=1)
#
#    # 3. Connect to live microphone input
#    with sd.InputStream(
#        samplerate=22050,
#        blocksize=1024,
#        channels=1,
#        dtype="float32",
#        callback=bridge,
#    ):
#        # 4. Run interactive animation
#        ani = animation.FuncAnimation(sparkline_rms.fig, update_rms, interval=30, blit=True)
#        plt.show()
#
# Practical considerations for live streaming
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - **Rolling history for open streams**: For continuous microphone capture with no set end time,
#   maintain a rolling buffer of recent history (e.g. the last 5 seconds) by shifting the display
#   array with `np.roll`.
# - **Block size and latency**: Smaller block sizes reduce latency at the expense of higher CPU callback
#   frequency. If dropouts or overflows occur (reported in `status`), increase `blocksize`.
# - **Frame-local feature extraction**: Incremental feature extractors should avoid global normalization
#   dependencies (e.g. use ``center=False``, fixed ``ref``, or frame-local parameters).
