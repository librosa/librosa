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
independent responsibilities:

1. **Audio I/O & Callbacks (Producer)**: High-priority system callbacks from the audio hardware
   (such as :class:`sounddevice.InputStream` or PortAudio) execute on a dedicated OS thread
   under hard real-time deadlines and must never block. :class:`librosa.StreamBridge` acts directly
   as the audio callback, receiving raw audio chunks, normalizing channel geometries and data types,
   and buffering them in a thread-safe queue.
2. **Buffering, Overlap, & Resampling (Analysis)**: Feature extractors often require observation windows that
   differ in size or sample rate from the capture device. Calling :meth:`librosa.StreamBridge.stream`
   (or :func:`librosa.stream`) accumulates incoming chunks into overlapping analysis windows,
   downmixes multi-channel audio, and handles on-the-fly streaming sample-rate conversion via ``soxr``.
3. **Display or Processing (Consumer)**: Downstream consumers (such as GUI animation loops,
   feature extraction pipelines, or worker threads) iterate over the yielded analysis blocks
   without stalling the audio callback.

.. note::
    **Understanding Sampling Rates: Hardware vs. Analysis**

    A frequent source of errors in live capture is confusing the **hardware capture rate**
    with the **downstream analysis rate**:

    - **Device rate (``device_sr``)**: Physical audio hardware (built-in microphones, USB interfaces)
      operates at fixed native rates—almost always 44,100 Hz or 48,000 Hz. If you request an
      unsupported rate like 22,050 Hz from :class:`sounddevice.InputStream`, the driver will
      typically fail with an invalid sample rate error. Both :class:`sounddevice.InputStream`
      and :class:`librosa.StreamBridge` **must match the device's native sampling rate**.
    - **Analysis rate (``target_sr``)**: Downstream analysis algorithms or machine learning models
      often expect a different rate (such as 22,050 Hz or 16,000 Hz). Resampling is *not* done
      by the callback or the bridge; it is performed on the fly when calling
      :meth:`librosa.StreamBridge.stream(..., sr=target_sr) <librosa.StreamBridge.stream>`.

Live microphone capture pattern
-------------------------------
When connecting to a real microphone with :mod:`sounddevice`, query the input device's
native sample rate, configure both :class:`sounddevice.InputStream` and :class:`librosa.StreamBridge`
at that rate, and specify your desired analysis rate in :meth:`~librosa.StreamBridge.stream`:

.. code:: python

    import sounddevice as sd
    import librosa

    # 1. Query the device's native sampling rate (typically 44100 or 48000 Hz)
    device_info = sd.query_devices(kind="input")
    device_sr = int(device_info["default_samplerate"])

    # 2. Select target analysis rate (e.g. 22050 Hz for 44.1 kHz, or 24000 Hz for 48 kHz).
    #    An integer downsampling ratio (2:1 or 3:1) ensures seamless sample alignment.
    target_sr = 22050 if device_sr % 22050 == 0 else (device_sr // 2)

    # 3. Create the bridge and open the input stream at the device's native rate
    with (
        librosa.StreamBridge(sr=device_sr, channels=1) as bridge,
        sd.InputStream(
            samplerate=device_sr, blocksize=512, channels=1, callback=bridge
        ),
    ):
        # 4. Stream analysis blocks, converting from device_sr to target_sr on the fly
        for block in bridge.stream(
            block_length=1, frame_length=1024, hop_length=512, sr=target_sr
        ):
            # Compute features in real time on the resampled audio block
            rms = librosa.feature.rms(
                y=block, frame_length=1024, hop_length=512, center=False
            )
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
# Real-Time Audio Architecture: Callbacks, Bridging, and Windows
# --------------------------------------------------------------
# Before looking at the code, let's examine how streaming audio capture works
# under the hood and why a bridge is required between the audio hardware and Librosa.
#
# 1. Audio Callbacks & Real-Time Constraints
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Audio capture libraries like :mod:`sounddevice` (and PortAudio) use an event-driven
# **push model**. When an input stream starts, the operating system's audio subsystem
# runs a dedicated, high-priority background audio thread. As the hardware's Analog-to-Digital
# Converter (ADC) samples sound, it fills a small hardware buffer. Once filled (every ``blocksize``
# frames), PortAudio interrupts the OS to invoke an **audio callback function**, passing the
# newly captured samples.
#
# Audio callbacks operate under hard real-time constraints:
#
# - **Strict time deadlines**: The callback must return before the hardware fills the next buffer.
# - **Buffer overruns (xruns)**: If the callback runs too slow, the hardware buffer overflows.
#   Incoming samples are dropped, causing permanent data loss and audible glitches.
# - **Forbidden operations**: Audio callbacks must **never** perform blocking operations, heavy
#   computations (such as FFTs, STFT, or deep learning inference), memory allocations, file/network I/O,
#   or GUI updates.
#
# 2. The Role of StreamBridge
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Python data analysis code is typically **pull-based** (e.g. iterating over a generator in a
# ``for`` loop). This creates an architectural clash with the **push-based** audio callback.
#
# :class:`librosa.StreamBridge` bridges this gap:
#
# - **Producer side (Audio Callback)**: Passed directly as ``callback=bridge`` to :class:`sounddevice.InputStream`.
#   When invoked on the real-time audio thread, it rapidly normalizes incoming data (converting integer
#   PCM like 16-bit or 24-bit to standard float32, transposing channel dimensions to Librosa's canonical
#   shape), pushes the chunk into a thread-safe internal queue, and immediately returns.
# - **Consumer side (Python Iterator)**: In your main or worker thread, ``bridge`` acts as an iterable
#   stream of audio chunks, yielding data safely without risking audio thread starvation.
# - **Jitter absorption & Overflow protection**: If feature extraction or plotting experiences a brief
#   CPU spike, the internal queue absorbs the latency. Configurable overflow policies (such as ``"warn"``
#   or ``"drop_oldest"``) prevent uncontrolled memory growth if the consumer falls behind.
#
# 3. Hardware Sampling Rate vs. Analysis Sampling Rate
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A key concept in live streaming is that **the capture sampling rate and the analysis sampling rate
# are completely decoupled**:
#
# - **Hardware capture rate (``device_sr``)**: Physical sound cards and microphones operate natively at
#   fixed rates—most commonly 44,100 Hz or 48,000 Hz. Requesting an arbitrary rate like 22,050 Hz from
#   :class:`sounddevice.InputStream` will fail on devices that do not support hardware resampling.
#   Therefore, **both** :class:`sounddevice.InputStream` **and** :class:`librosa.StreamBridge`
#   **must match the device's native rate**.
# - **Analysis rate (``target_sr``)**: Librosa feature extraction pipelines often expect a specific
#   rate (such as 22,050 Hz or 16,000 Hz). The ``StreamBridge`` does *not* resample chunks inside the callback,
#   as resampling is too computationally expensive for the real-time thread. Instead, sample-rate conversion
#   is handled downstream by calling :meth:`librosa.StreamBridge.stream(..., sr=target_sr) <librosa.StreamBridge.stream>`.
#   The stream generator resamples incoming chunks on the fly via ``soxr`` before yielding analysis blocks.
#
# 4. Hardware Blocks, Sliding Windows, and Overlapping Frames
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Live audio involves three distinct sizing parameters:
#
# - **Capture chunk size (``blocksize``)**: The number of raw samples pushed per callback invocation
#   (e.g., 512 samples at ``device_sr``). Lower values reduce latency; higher values reduce CPU overhead.
# - **Analysis frame length (``frame_length``)**: The observation window required by spectral algorithms
#   (e.g., 1024 or 2048 samples at ``target_sr``) to achieve sufficient frequency resolution.
# - **Hop length (``hop_length``)**: The step size between consecutive analysis windows (e.g., 512 samples).
#   Because ``frame_length > hop_length``, consecutive analysis blocks overlap in time.
#
# :meth:`librosa.StreamBridge.stream` accumulates incoming chunks into an internal sliding buffer,
# advances by ``hop_length`` samples on each step, and yields full analysis blocks of length ``frame_length``.
# Setting ``warm_up=True`` pre-fills the initial buffer history with silence so that the first analysis block
# is yielded immediately as soon as the first hop arrives, eliminating startup delay.
#
#
# %%
# Simulating an Input Device
# --------------------------
# To make the examples below self-contained and reproducible without requiring a
# physical microphone, we define a small helper class (``VirtualInputStream``) that
# reads fixed-size blocks from an audio file and invokes ``callback`` with the same
# signature as :class:`sounddevice.InputStream`.
#
# Notice how ``VirtualInputStream`` mimics PortAudio:
#
# 1. In ``__enter__``, it slices the audio file into chunks of size ``blocksize`` and passes them to
#    ``callback(block, len(block), None, None)``, exactly as the PortAudio C library does.
# 2. In ``__exit__``, it invokes ``callback.close()``, which signals to the :class:`~librosa.StreamBridge`
#    that capture has ended. Downstream consumer loops will cleanly complete once the buffer drains.
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
# Notice how the sampling rates are configured:
#
# - The audio source (the trumpet example) is recorded at 22,050 Hz.
# - Both :class:`VirtualInputStream` and :class:`~librosa.StreamBridge` are configured
#   with ``sr=22050`` to match this input rate.
# - In ``bridge_rms.stream(...)``, we do not specify ``sr`` (leaving it as ``None``), so
#   analysis runs directly at the native 22,050 Hz rate without resampling.
#
# The architecture cleanly separates each component:
#
# - **I/O**: :class:`librosa.StreamBridge` receives incoming blocks from the device callback.
# - **Streaming & Analysis**: :meth:`librosa.StreamBridge.stream` yields each block to ``process_rms``.
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
# 1. **Sliding Window Overlap**: Low-latency capture delivers small chunks (e.g. 512 samples / ~23 ms
#    at 22,050 Hz), but frequency transforms require longer observation windows (e.g. 1024 samples / ~93 ms
#    at 11,025 Hz) for spectral resolution. We need a sliding observation window where each incoming
#    chunk shifts the buffer forward by a smaller hop size, producing overlapping windows.
# 2. **Online Resampling**: Hardware inputs frequently capture at 44.1 kHz or 48 kHz, while downstream
#    models expect 22.05 kHz or 11.025 kHz.
#
# Here, we clearly distinguish between the two sampling rates:
#
# - **Input / Bridge Rate (22,050 Hz)**: :class:`~librosa.StreamBridge` is initialized with ``sr=22050``,
#   matching the simulated device (the trumpet file).
# - **Analysis / Resampled Rate (11,025 Hz)**: The call to :meth:`bridge_sc.stream(..., sr=target_sr) <librosa.StreamBridge.stream>`
#   specifies ``sr=11025``. The stream generator resamples incoming chunks on the fly using ``soxr``
#   before assembling them into sliding analysis windows.
#
# In addition:
#
# - Setting ``warm_up=True`` pre-fills the initial observation window with silence (zeros) so that
#   the first analysis frame is produced immediately upon receiving the first hop, eliminating
#   startup latency.


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
# Practical Considerations for Live Hardware Streams
# ---------------------------------------------------
# When deploying live stream processing on physical hardware, keep the following
# best practices in mind:
#
# - **Matching the hardware device rate**: Audio hardware devices typically operate natively
#   at 44.1 kHz or 48 kHz. Always query the default rate of your input device using
#   ``device_sr = int(sd.query_devices(kind='input')['default_samplerate'])`` and supply that
#   exact rate to both :class:`sounddevice.InputStream` and :class:`librosa.StreamBridge`.
#   Never request an arbitrary unsupported rate in :class:`sounddevice.InputStream`.
# - **Choosing resampling ratios**: When converting sample rates in :meth:`~librosa.StreamBridge.stream`,
#   Librosa requires that ``(block_length * hop_length * orig_sr) / sr`` evaluates to an exact integer.
#   Integer downsampling ratios (such as 44,100 Hz -> 22,050 Hz [2:1] or 48,000 Hz -> 24,000 Hz [2:1]
#   or 16,000 Hz [3:1]) satisfy this cleanly for any hop size.
# - **Block size and latency**: Smaller hardware block sizes (e.g. 256 or 512 frames) reduce latency
#   at the expense of higher CPU callback frequency. If buffer overruns occur (indicated by status flags
#   in the audio callback), increase ``blocksize`` (e.g. 1024 or 2048 frames).
# - **Queue overflow policies**: In production environments where feature extraction or rendering
#   may experience intermittent stalls, configure :class:`~librosa.StreamBridge` with
#   ``overflow_policy="drop_oldest"`` to operate as a fixed-latency ring buffer, preventing unbounded
#   memory accumulation.
# - **Continuous streams with rolling history**: In this demonstration, we processed a
#   finite recording with a known duration. For open-ended live microphone streams,
#   maintain a rolling window of recent history (such as the last 5 seconds) by shifting
#   the display array with ``numpy.roll``.
# - **Frame-local feature extraction**: To keep streaming features consistent with whole-file
#   analysis, ensure that frame-level computations are context-independent (e.g. using
#   ``center=False`` and avoiding dynamic global parameters like ``top_db`` or callable ``ref``).
#   For further discussion, refer to :ref:`sphx_glr_auto_tutorials_03-advanced_plot_mfcc_stream.py`.
