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
2. **Consumer (Buffer Processing Layer)**: A consumer loop (or animation frame update) pulls
   blocks from the queue, maintains analysis buffers as needed, and invokes `librosa` to extract
   features on each buffer.
3. **Display Layer**: The extracted features are rendered in real time (for example, as an
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


In this notebook, we present two examples:

- **Example 1**: Basic stream processing, computing instantaneous Root-Mean-Square (RMS) energy per block.
- **Example 2**: Advanced buffer handling, demonstrating stream resampling (via `soxr.ResampleStream`)
  and overlapping sliding analysis windows (via a rolling FIFO buffer) to compute spectral centroid.

At the end, we describe how to swap the virtual device for a live microphone with `sounddevice.InputStream`.
"""

# %%
# Imports and Audio Source
# ------------------------
# We'll need standard queue and plotting modules, `soundfile` for simulated audio I/O,
# `soxr` for streaming sample-rate conversion, and `librosa` for signal processing.

import queue
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import soxr

import librosa

# We'll use the packaged 'trumpet' recording as our audio source
filename = librosa.ex("trumpet")
HTML(librosa.util.example_info("trumpet", html=True))


# %%
# Part 0: The Audio Device and Queue Setup
# ----------------------------------------
# In real-time audio systems, the audio hardware driver calls an audio callback
# function at regular intervals in a separate, high-priority OS thread.
#
# To avoid blocking the audio callback, we push incoming blocks into a thread-safe
# FIFO queue (`queue.Queue`) and let the consumer process them asynchronously.

audio_queue = queue.Queue()


def audio_callback(indata, _frames, _time, status):
    """Audio callback invoked by the stream for each new audio block.

    Parameters
    ----------
    indata : np.ndarray
        Array of input audio samples, shape (frames, channels).
    _frames : int
        Number of frames in the block (unused in this basic callback).
    _time : CData
        Timestamps from PortAudio (unused in this basic callback).
    status : CallbackFlags
        Status flags indicating underflow/overflow conditions.
    """
    if status:
        print(f"Stream status warning: {status}")
    # Always push a copy of the incoming block to avoid data races
    audio_queue.put(indata.copy())


# %%
# Simulating the Audio Device
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In an offline build or automated test environment, no physical input device is available.
# We create a `VirtualInputStream` class that mirrors the context-manager interface of
# `sounddevice.InputStream`. It reads fixed-size chunks from an audio file using
# `soundfile.SoundFile.blocks()` and feeds them to the exact same `audio_callback`.
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
# The processing function receives a raw in-memory numpy array, with zero knowledge
# of whether it originated from a real microphone or our simulated device.


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
# We set up a Matplotlib figure with a rolling history buffer.
# The animation update function drains `audio_queue`, computes the RMS energy for each
# chunk using `process_rms`, rolls the history array, and updates the plot line.

blocksize = 1024
history_len = 60
rms_history = np.zeros(history_len)

# Represent time relative to the current moment (0 seconds is now)
sr = librosa.get_samplerate(filename)
time_window = (history_len - 1) * blocksize / sr
times = np.linspace(-time_window, 0, history_len)

fig_rms, ax_rms = plt.subplots(figsize=(8, 3))
(line_rms,) = ax_rms.plot(times, rms_history, color="#3f90da", lw=2)
ax_rms.set_ylim(0, 0.4)
ax_rms.set_xlim(-time_window, 0)
ax_rms.set_title("Live Stream Processing: Instantaneous RMS Energy")
ax_rms.set_xlabel("Time (seconds relative to current)")
ax_rms.set_ylabel("RMS Amplitude")
ax_rms.grid(True, linestyle="--", alpha=0.5)


def update_rms(_frame):
    """Update function invoked by FuncAnimation on each frame."""
    global rms_history
    updated = False
    while not audio_queue.empty():
        block = audio_queue.get_nowait()
        rms_val = process_rms(block)
        rms_history = np.roll(rms_history, -1)
        rms_history[-1] = rms_val
        updated = True

    if updated:
        line_rms.set_ydata(rms_history)
    return (line_rms,)


# Open our virtual stream as a context manager, mirroring sounddevice.InputStream
with VirtualInputStream(
    filename, blocksize=blocksize, callback=audio_callback
) as stream:
    ani_rms = animation.FuncAnimation(
        fig_rms,
        update_rms,
        frames=stream,
        interval=1000 * blocksize / stream.samplerate,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

# Sphinx-Gallery automatically captures the animation and renders it as an HTML5 video


# %%
# Part 2: Example 2 — Advanced Buffer Handling (Overlap & Resampling)
# -------------------------------------------------------------------
# While computing features block-by-block works for simple metrics like RMS energy,
# real-world music information retrieval (MIR) often introduces two additional challenges:
#
# 1. **Latency vs. Frequency Resolution**: Low-latency interactive applications require
#    small audio blocks (e.g. 512 samples / ~23 ms). However, frequency-domain features
#    such as the Short-Time Fourier Transform (STFT), spectral centroid, or chroma require
#    a longer observation window (e.g. 2048 samples / ~93 ms) to obtain good frequency resolution.
#    To resolve this without introducing latency, we maintain a **sliding analysis window**
#    where each incoming 512-sample block shifts the buffer, producing 75% overlap.
# 2. **Sample-Rate Mismatch**: Audio capture devices often operate at hardware sample rates
#    like 44.1 kHz or 48 kHz, while downstream models (or pretrained acoustic features)
#    frequently expect standard rates like 22.05 kHz or 16 kHz.
#
# We can solve both requirements cleanly using tools already bundled with librosa:
# `soxr.ResampleStream` for on-the-fly streaming resampling, and `numpy.roll` for sliding buffers.


class StreamingBuffer:
    """Sliding analysis buffer with on-the-fly stream resampling.

    Parameters
    ----------
    frame_length : int
        Size of the active analysis window passed to librosa (e.g. 2048 samples).
    in_sr : float
        Input sample rate from the capture device.
    target_sr : float
        Target sample rate expected by the downstream feature extractor.
    quality : str
        Quality mode for soxr streaming resampler ('HQ', 'VHQ', etc.).
    """

    def __init__(self, frame_length=2048, in_sr=22050, target_sr=22050, quality="HQ"):
        self.frame_length = frame_length
        self.target_sr = target_sr
        self.resampler = (
            soxr.ResampleStream(in_sr, target_sr, num_channels=1, quality=quality)
            if in_sr != target_sr
            else None
        )
        self.buffer = np.zeros(frame_length, dtype=np.float32)

    def update(self, chunk):
        """Append a new raw audio block and return the updated analysis window.

        Parameters
        ----------
        chunk : np.ndarray
            Incoming audio block of shape (samples, channels) or (samples,).

        Returns
        -------
        window : np.ndarray
            The updated analysis buffer of length `frame_length`.
        """
        y = chunk.squeeze()
        if self.resampler is not None:
            y = self.resampler.resample_chunk(y)

        n = len(y)
        if n > 0:
            if n >= self.frame_length:
                self.buffer[:] = y[-self.frame_length :]
            else:
                self.buffer = np.roll(self.buffer, -n)
                self.buffer[-n:] = y
        return self.buffer


# %%
# Extracting Features on Overlapping Windows
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this example, we'll downsample our incoming 22,050 Hz audio stream to 11,025 Hz
# using `soxr.ResampleStream`, maintain an overlapping analysis window of 1,024 samples,
# and compute `librosa.feature.spectral_centroid` on each sliding window.


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
# We set up an animation to track the live spectral brightness of the trumpet:

target_sr = 11025
frame_length = 1024
hop_length = 512

stream_buf = StreamingBuffer(
    frame_length=frame_length,
    in_sr=sr,
    target_sr=target_sr,
)

history_len_sc = 60
centroid_history = np.zeros(history_len_sc)

# Represent time relative to the current moment (0 seconds is now)
time_window_sc = (history_len_sc - 1) * hop_length / sr
times_sc = np.linspace(-time_window_sc, 0, history_len_sc)

fig_sc, ax_sc = plt.subplots(figsize=(8, 3))
(line_sc,) = ax_sc.plot(times_sc, centroid_history, color="#bd1f01", lw=2)
ax_sc.set_ylim(500, 4000)
ax_sc.set_xlim(-time_window_sc, 0)
ax_sc.set_title("Live Stream Processing: Spectral Centroid (Overlapping Buffer)")
ax_sc.set_xlabel("Time (seconds relative to current)")
ax_sc.set_ylabel("Frequency (Hz)")
ax_sc.grid(True, linestyle="--", alpha=0.5)


def update_centroid(_frame):
    """Update function for spectral centroid tracking."""
    global centroid_history
    updated = False
    while not audio_queue.empty():
        block = audio_queue.get_nowait()
        analysis_window = stream_buf.update(block)
        sc_val = process_spectral_centroid(analysis_window, sr=target_sr)
        centroid_history = np.roll(centroid_history, -1)
        centroid_history[-1] = sc_val
        updated = True

    if updated:
        line_sc.set_ydata(centroid_history)
    return (line_sc,)


# Open our virtual stream with shorter blocks (512 samples)
with VirtualInputStream(
    filename, blocksize=hop_length, callback=audio_callback
) as stream:
    ani_sc = animation.FuncAnimation(
        fig_sc,
        update_centroid,
        frames=stream,
        interval=1000 * hop_length / stream.samplerate,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )


# %%
# Conclusion: Connecting to a Live Audio Device
# ---------------------------------------------
# The major strength of separating the audio I/O layer from the buffer processing
# layer is that switching from simulation to a live microphone requires changing
# **only the stream source**.
#
# Everything else—the queue, `audio_callback`, `process_rms`, `StreamingBuffer`, and
# plotting routines—remains 100% identical.
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
#    # 2. Open an InputStream connected to your microphone
#    #    (leaving device=None selects the system default input device)
#    with sd.InputStream(
#        samplerate=22050,
#        blocksize=1024,
#        channels=1,
#        dtype="float32",
#        callback=audio_callback,
#    ):
#        # 3. In an interactive Python environment or script:
#        ani = animation.FuncAnimation(fig_rms, update_rms, interval=30, blit=True)
#        plt.show()
#
# Notice how the code structure matches: both `VirtualInputStream` and `sd.InputStream`
# are context managers taking the same callback, block size, and channel settings.
# The only operational difference is that `VirtualInputStream` acts as an iterable
# (`frames=stream`) so `FuncAnimation` can render the animation frames deterministically
# during offline documentation generation, whereas `sd.InputStream` runs on a live
# background thread while `FuncAnimation` polls on an interactive timer (`interval=30`).
#
# Practical considerations for live streaming
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# When running on physical hardware, keep the following best practices in mind:
#
# - **Block size and latency**: Smaller block sizes (e.g. 256-512 samples) reduce capture
#   latency but increase callback frequency and CPU overhead. If you experience dropouts
#   or buffer overruns (indicated by `status.input_overflow`), increase `blocksize`.
# - **Computation budget**: All processing within each consumer iteration must complete
#   faster than the real-time block duration (:math:`\\text{blocksize} / \\text{samplerate}`).
# - **Feature coupling**: When computing features like MFCCs incrementally, remember that
#   parameters like ``top_db`` or callable ``ref`` can introduce inter-frame dependencies.
#   Refer to the :ref:`sphx_glr_auto_tutorials_03-advanced_plot_mfcc_stream.py` tutorial
#   for details on frame-local configurations.
