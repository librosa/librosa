#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming utilities for audio I/O bridging and sliding window buffering."""

from __future__ import annotations

import os
import threading
import warnings
from collections import deque
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import soxr

from ..exceptions import (
    ParameterShapeError,
    ParameterTypeError,
    ParameterValueError,
    StreamBufferError,
    StreamClosedError,
    StreamTimeoutError,
)
from ..util.utils import align_samples, fix_length, is_positive_int
from .audio import to_mono

if TYPE_CHECKING:
    from typing import BinaryIO, Generator, Iterable, Self

    from numpy.typing import DTypeLike


__all__ = [
    "StreamBridge",
    "stream",
]


class _StreamBuffer:
    """Internal sliding analysis window accumulator and streaming DSP processor.

    `_StreamBuffer` maintains a sliding observation buffer for streaming audio
    processing. It handles on-the-fly sample rate conversion via
    `soxr.ResampleStream`, multi-channel downmixing to mono, offset skipping,
    and duration bounds.
    """

    def __init__(
        self,
        *,
        frame_length: int,
        hop_length: int,
        block_length: int = 1,
        orig_sr: float,
        sr: float | None = 22050,
        channels: int = 1,
        mono: bool = True,
        offset: float = 0.0,
        duration: float | None = None,
        warm_up: bool = False,
        fill_value: float | None = None,
        res_type: str = "soxr_hq",
        dtype: DTypeLike = np.float32,
    ):
        if not is_positive_int(block_length):
            raise ParameterValueError(f"block_length={block_length} must be a positive integer")
        if not is_positive_int(frame_length):
            raise ParameterValueError(f"frame_length={frame_length} must be a positive integer")
        if not is_positive_int(hop_length):
            raise ParameterValueError(f"hop_length={hop_length} must be a positive integer")
        if not (np.isfinite(orig_sr) and orig_sr > 0):
            raise ParameterValueError(f"orig_sr={orig_sr} must be a positive number")
        if sr is not None and not (np.isfinite(sr) and sr > 0):
            raise ParameterValueError(f"sr={sr} must be a positive number")
        if not is_positive_int(channels):
            raise ParameterValueError(f"channels={channels} must be a positive integer")
        if offset < 0:
            raise ParameterValueError(f"offset={offset} must be non-negative")
        if duration is not None and duration < 0:
            raise ParameterValueError(f"duration={duration} must be non-negative")
        if res_type not in {"soxr_vhq", "soxr_hq", "soxr_mq", "soxr_lq", "soxr_qq"}:
            raise ParameterValueError(f"res_type={res_type} is not a valid soxr resampling mode for streaming")

        self.block_length = int(block_length)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.orig_sr = float(orig_sr)
        self.sr = float(sr) if sr is not None else self.orig_sr
        self.channels = int(channels)
        self.mono = bool(mono)
        self.offset = float(offset)
        self.duration = float(duration) if duration is not None else None
        self.fill_value = fill_value
        self.res_type = res_type
        self.dtype = np.dtype(dtype)

        self.target_yield_size = (self.block_length - 1) * self.hop_length + self.frame_length
        self.block_hop_length = self.block_length * self.hop_length
        self.process_channels = 1 if self.mono else self.channels

        self.orig_read_size = align_samples(
            self.block_hop_length,
            orig_sr=self.sr,
            target_sr=self.orig_sr,
            error=True,
        )
        self.needs_resampling = (sr is not None) and (self.orig_sr != self.sr)
        if self.needs_resampling:
            self.resampler = soxr.ResampleStream(
                in_rate=self.orig_sr,
                out_rate=self.sr,
                num_channels=self.process_channels,
                dtype=self.dtype,
                quality=res_type,
            )
        else:
            self.resampler = None

        self.capacity = self.target_yield_size + (self.block_hop_length * 2)
        self.buffer = np.zeros(self._buffer_shape(self.capacity), dtype=self.dtype)
        self.warm_up = bool(warm_up)
        self.prefill = max(0, self.frame_length - self.hop_length) if self.warm_up else 0
        self.write_idx = self.prefill
        self.read_idx = 0

        self.samples_to_skip = int(self.offset * self.orig_sr) if self.offset > 0 else 0
        self.samples_remaining = int(self.duration * self.orig_sr) if self.duration is not None else None
        self.has_input = False

    def _buffer_shape(self, capacity: int) -> tuple[int, ...]:
        """Return the buffer shape for a given sample capacity."""
        return (capacity,) if self.process_channels == 1 else (capacity, self.process_channels)

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Ingest a canonical audio chunk and return any ready analysis blocks.

        Parameters
        ----------
        chunk : np.ndarray
            Incoming audio chunk of shape ``(samples,)`` or ``(channels, samples)``.

        Returns
        -------
        blocks : list of np.ndarray
            List of complete analysis blocks of size `target_yield_size`.
        """
        if self.is_complete() or chunk.size == 0:
            return []

        if chunk.dtype != self.dtype:
            chunk = chunk.astype(self.dtype)

        # Downmix or transpose from canonical (channels, samples) to (samples, channels)
        if self.mono:
            y = to_mono(chunk)
        elif chunk.ndim > 1:
            y = chunk.T
        else:
            y = chunk

        # Skip initial offset samples
        if self.samples_to_skip > 0:
            n_samples = y.shape[0]
            if n_samples <= self.samples_to_skip:
                self.samples_to_skip -= n_samples
                return []
            y = y[self.samples_to_skip :]
            self.samples_to_skip = 0

        # Truncate to duration if specified
        if self.samples_remaining is not None:
            if y.shape[0] >= self.samples_remaining:
                y = y[: self.samples_remaining]
                self.samples_remaining = 0
            else:
                self.samples_remaining -= y.shape[0]
            if y.shape[0] == 0:
                return []

        self.has_input = True

        # Resample on-the-fly
        if self.resampler is not None:
            target_chunk = self.resampler.resample_chunk(y)
        else:
            target_chunk = y

        n_incoming = target_chunk.shape[0]
        if n_incoming == 0:
            return []

        # Shift buffer if near capacity
        if self.write_idx + n_incoming > self.capacity:
            available = self.write_idx - self.read_idx
            self.buffer[:available] = self.buffer[self.read_idx : self.write_idx]
            self.read_idx = 0
            self.write_idx = available

            if self.write_idx + n_incoming > self.capacity:
                self.capacity = self.write_idx + n_incoming + self.target_yield_size
                new_buffer = np.zeros(self._buffer_shape(self.capacity), dtype=self.dtype)
                new_buffer[: self.write_idx] = self.buffer[: self.write_idx]
                self.buffer = new_buffer

        self.buffer[self.write_idx : self.write_idx + n_incoming] = target_chunk
        self.write_idx += n_incoming

        # Slice all complete blocks
        blocks = []
        while self.write_idx - self.read_idx >= self.target_yield_size:
            block = self.buffer[self.read_idx : self.read_idx + self.target_yield_size]
            blocks.append(block.T.copy())
            self.read_idx += self.block_hop_length

        return blocks

    def flush(self) -> list[np.ndarray]:
        """Flush resampler delay buffers and yield final (optionally padded) blocks.

        Returns
        -------
        blocks : list of np.ndarray
            Remaining complete or padded analysis blocks.
        """
        if not self.has_input:
            return []

        remainder = self.buffer[self.read_idx : self.write_idx]
        if self.resampler is not None:
            empty = np.empty((0, *remainder.shape[1:]), dtype=self.dtype)
            tail = self.resampler.resample_chunk(empty, last=True)
            if tail.shape[0] > 0:
                remainder = np.concatenate([remainder, tail], axis=0)

        blocks = []
        for rem_idx in range(0, remainder.shape[0], self.block_hop_length):
            block = remainder[rem_idx : rem_idx + self.target_yield_size].T.copy()
            if block.shape[-1] < self.target_yield_size and self.fill_value is not None:
                block = fix_length(
                    block,
                    size=self.target_yield_size,
                    constant_values=self.fill_value,
                )
            blocks.append(block)

        self.read_idx = self.write_idx
        return blocks

    def is_complete(self) -> bool:
        """Check if the maximum input duration has been reached."""
        return self.samples_remaining is not None and self.samples_remaining <= 0

    def process_stream(self, chunks: Iterable[np.ndarray]) -> Generator[np.ndarray, None, None]:
        """Ingest chunks from an iterable, yielding ready analysis blocks followed by flush."""
        if not self.is_complete():
            for chunk in chunks:
                yield from self.push(chunk)
                if self.is_complete():
                    break
        yield from self.flush()


class StreamBridge:
    """Universal I/O bridge connecting audio callbacks into `librosa.stream`.

    `StreamBridge` acts as an audio callback (compatible with PortAudio,
    :mod:`sounddevice`, and ``PyAudio``) that enqueues incoming raw audio chunks,
    and simultaneously acts as an iterator of Librosa-canonical NumPy arrays.

    Parameters
    ----------
    orig_sr : float > 0
        Sampling rate of the incoming audio stream (required keyword-only argument).
    channels : int > 0, default=1
        Number of audio channels in the incoming stream.
    channels_first : bool, default=False
        If True, incoming multi-channel buffers are shaped ``(channels, samples)``.
        If False, incoming multi-channel buffers are shaped ``(samples, channels)``
        (standard for :mod:`sounddevice` and PortAudio) and will be transposed to
        canonical shape ``(channels, samples)``.
    dtype : DTypeLike, default=np.float32
        Data type of incoming audio chunks or byte buffers. If integer PCM,
        samples are normalized to floating-point ``[-1.0, 1.0]`` according to
        their integer bit depth.
    maxsize : int >= 0, default=0
        Maximum capacity of the internal chunk buffer. If 0, capacity is unbounded.
        For live real-time hardware streams where downstream processing might
        intermittently stall, setting a bounded capacity (e.g. ``maxsize=100``)
        combined with ``overflow_policy="drop_oldest"`` prevents uncontrolled
        memory growth and latency accumulation.
    overflow_policy : {'warn', 'drop_oldest', 'drop_newest', 'error'}, default='warn'
        Action taken when `maxsize` is reached:
            - ``"warn"``: Discard the oldest queued chunk and emit a warning when chunks are retrieved.
            - ``"drop_oldest"``: Discard the oldest queued chunk (fixed-latency ring buffer).
            - ``"drop_newest"``: Discard the incoming chunk.
            - ``"error"``: Raise `BufferError`.
    timeout : float | None, default=None
        Maximum time in seconds to wait for a chunk during iteration
        before raising `queue.Empty`. If ``None``, blocks indefinitely.

    Examples
    --------
    Bridge real-time input from :mod:`sounddevice` and stream analysis blocks.
    Note that streaming analysis requires left-aligned frames, so we pass
    ``center=False`` to avoid boundary padding artifacts between blocks.
    For a complete walkthrough, see :ref:`sphx_glr_auto_tutorials_03-advanced_plot_stream_capture.py`.

    >>> import sounddevice as sd  # doctest: +SKIP
    >>> with librosa.StreamBridge(
    ...     orig_sr=48000, channels=1
    ... ) as bridge:  # doctest: +SKIP
    ...     with sd.InputStream(samplerate=48000, channels=1, callback=bridge):
    ...         for i, block in enumerate(
    ...             bridge.stream(
    ...                 block_length=256, frame_length=2048, hop_length=512, sr=None
    ...             )
    ...         ):
    ...             centroid = librosa.feature.spectral_centroid(
    ...                 y=block, sr=48000, n_fft=2048, hop_length=512, center=False
    ...             )
    ...             print(f"Block {i}: mean centroid = {centroid.mean():.1f} Hz")
    ...             if i >= 10:
    ...                 break
    """

    def __init__(
        self,
        *,
        orig_sr: float,
        channels: int | np.integer[Any] = 1,
        channels_first: bool = False,
        dtype: DTypeLike = np.float32,
        maxsize: int | np.integer[Any] = 0,
        overflow_policy: str = "warn",
        timeout: float | None = None,
    ):
        if not (np.isfinite(orig_sr) and orig_sr > 0):
            raise ParameterValueError(f"orig_sr={orig_sr} must be a positive number")
        if not is_positive_int(channels):
            raise ParameterValueError(f"channels={channels} must be a positive integer")
        if not isinstance(maxsize, (int, np.integer)):
            raise ParameterTypeError(f"maxsize={maxsize} must be an integer")
        if maxsize < 0:
            raise ParameterValueError(f"maxsize={maxsize} must be a non-negative integer")
        if overflow_policy not in {"warn", "drop_oldest", "drop_newest", "error"}:
            raise ParameterValueError(f"Invalid overflow_policy='{overflow_policy}'")

        self.orig_sr = float(orig_sr)
        self.channels = int(channels)
        self.channels_first = bool(channels_first)
        self.dtype = np.dtype(dtype)
        self.maxsize = int(maxsize)
        self.overflow_policy = overflow_policy
        self.timeout = timeout

        self._buffer: deque[np.ndarray] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._dropped_chunks = 0

    @property
    def sr(self) -> float:
        """Sampling rate of the incoming audio stream.

        Alias for `orig_sr`.

        Returns
        -------
        sr : float
            Sampling rate of the incoming audio stream.
        """
        return self.orig_sr

    def __call__(self, data: np.ndarray | bytes, *_args: Any, **_kwargs: Any) -> tuple[None, int]:
        """Universal audio callback.

        Compatible with :mod:`sounddevice` (which ignores the return value) and
        ``PyAudio`` (which expects a tuple ``(out_data, flag)`` where flag 0
        corresponds to ``paContinue`` and flag 1 corresponds to ``paComplete``).
        """
        if not self._enqueue(data, raise_on_closed=False):
            return (None, 1)
        return (None, 0)

    def push(self, data: np.ndarray | bytes | memoryview) -> None:
        """Push a raw audio chunk into the bridge.

        Parameters
        ----------
        data : np.ndarray, bytes, or memoryview
            Incoming audio buffer. Integer arrays or byte buffers are
            converted to float32 according to `dtype`. Single-channel audio is
            normalized to 1D shape ``(samples,)``, and multi-channel audio is
            normalized to shape ``(channels, samples)``.
        """
        self._enqueue(data, raise_on_closed=True)

    def _enqueue(self, data: np.ndarray | bytes | memoryview, *, raise_on_closed: bool) -> bool:
        """Normalize and atomically enqueue an audio chunk if the bridge is open."""
        if not raise_on_closed and self._closed:
            return False

        chunk = self._normalize(data)

        with self._cond:
            if self._closed:
                if raise_on_closed:
                    raise StreamClosedError("Cannot push data to a closed StreamBridge.")
                return False

            if self.maxsize > 0 and len(self._buffer) >= self.maxsize:
                if self.overflow_policy == "drop_oldest":
                    self._buffer.popleft()
                elif self.overflow_policy == "drop_newest":
                    return True
                elif self.overflow_policy == "error":
                    raise StreamBufferError("StreamBridge queue overflow: consumer is slower than capture.")
                elif self.overflow_policy == "warn":
                    self._dropped_chunks += 1
                    self._buffer.popleft()

            self._buffer.append(chunk)
            self._cond.notify()
            return True

    def _normalize(self, data: np.ndarray | bytes | memoryview) -> np.ndarray:
        """Normalize any incoming audio representation into Librosa canonical float32."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            raw = np.frombuffer(data, dtype=self.dtype)
        elif isinstance(data, np.ndarray):
            raw = data
        else:
            raise ParameterTypeError(f"Unsupported audio data type: {type(data)}")

        if np.issubdtype(raw.dtype, np.signedinteger):
            scale = float(1 << (8 * raw.dtype.itemsize - 1))
            arr = raw.astype(np.float32) / scale
        elif np.issubdtype(raw.dtype, np.unsignedinteger):
            midpoint = float(1 << (8 * raw.dtype.itemsize - 1))
            arr = (raw.astype(np.float32) - midpoint) / midpoint
        else:
            arr = raw.astype(np.float32, copy=True)

        if arr.ndim == 1:
            if self.channels > 1:
                if arr.size % self.channels != 0:
                    raise ParameterShapeError(
                        f"1D input of size {arr.size} is not divisible by channels={self.channels}"
                    )
                arr = arr.reshape(self.channels, -1) if self.channels_first else arr.reshape(-1, self.channels).T
        elif arr.ndim == 2:
            if not self.channels_first:
                arr = arr.T
            if arr.shape[0] != self.channels:
                raise ParameterShapeError(f"Expected {self.channels} channels, but got {arr.shape[0]}")
            if self.channels == 1:
                arr = arr.reshape(-1)
        else:
            raise ParameterShapeError(f"Audio data must be 1D or 2D; got {arr.ndim}D")

        return arr

    def __iter__(self) -> Self:
        """Return iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next audio chunk from the buffer."""
        dropped = 0
        with self._cond:
            if not self._cond.wait_for(
                lambda: bool(self._buffer) or self._closed,
                timeout=self.timeout,
            ):
                raise StreamTimeoutError("Timeout waiting for audio data from StreamBridge.")

            if self._dropped_chunks > 0:
                dropped = self._dropped_chunks
                self._dropped_chunks = 0

            if self._buffer:
                chunk = self._buffer.popleft()
            else:
                chunk = None

        if dropped > 0:
            warnings.warn(
                f"StreamBridge queue is full; dropped {dropped} chunk(s).",
                RuntimeWarning,
                stacklevel=2,
            )

        if chunk is not None:
            return chunk
        raise StopIteration

    def stream(
        self,
        *,
        block_length: int,
        frame_length: int,
        hop_length: int,
        sr: float | None = 22050,
        mono: bool = True,
        offset: float = 0.0,
        duration: float | None = None,
        warm_up: bool = False,
        fill_value: float | None = None,
        res_type: str = "soxr_hq",
        dtype: DTypeLike = np.float32,
    ) -> Generator[np.ndarray, None, None]:
        """Stream audio from this bridge in fixed-length analysis buffers.

        This method is equivalent to calling ``librosa.stream(self, ...)``.

        Parameters
        ----------
        block_length : int > 0
            The number of frames to include in each block.
        frame_length : int > 0
            The number of samples per frame.
        hop_length : int > 0
            The number of samples to advance between frames.
        sr : float > 0, default=22050
            Target sampling rate.
            If None, uses `self.orig_sr` without resampling.
        mono : bool, default=True
            Convert multi-channel signals to mono during streaming.
        offset : float >= 0, default=0.0
            Start reading after this time (in seconds).
        duration : float > 0, optional
            Only yield up to this much audio (in seconds).
        warm_up : bool, default=False
            If True, pre-fill the buffer with silence (zeros) so that the first
            incoming chunk of size ``block_length * hop_length`` immediately yields
            a full analysis block without startup delay.
        fill_value : float, optional
            Value used to pad short blocks at the end of the stream.
        res_type : str, default='soxr_hq'
            Resample type for `soxr`.
        dtype : numeric type, default=np.float32
            Data type of the yielded blocks.

        Yields
        ------
        block : np.ndarray
            An audio buffer of (at most)
            ``(block_length-1) * hop_length + frame_length`` samples.

        See Also
        --------
        librosa.stream
        """
        buf = _StreamBuffer(
            frame_length=frame_length,
            hop_length=hop_length,
            block_length=block_length,
            orig_sr=self.orig_sr,
            sr=sr,
            channels=self.channels,
            mono=mono,
            offset=offset,
            duration=duration,
            warm_up=warm_up,
            fill_value=fill_value,
            res_type=res_type,
            dtype=dtype,
        )
        yield from buf.process_stream(self)

    def close(self) -> None:
        """Close the bridge and signal termination to waiting consumers."""
        with self._cond:
            if not self._closed:
                self._closed = True
                self._cond.notify_all()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, closing the bridge."""
        self.close()


def stream(
    path: str | int | os.PathLike[Any] | sf.SoundFile | BinaryIO | StreamBridge,
    *,
    block_length: int,
    frame_length: int,
    hop_length: int,
    sr: float | None = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    warm_up: bool = False,
    fill_value: float | None = None,
    res_type: str = "soxr_hq",
    dtype: DTypeLike = np.float32,
) -> Generator[np.ndarray, None, None]:
    """Stream audio in fixed-length buffers.

    This is primarily useful for processing large files that won't
    fit entirely in memory at once, or for processing live audio streams
    captured via `librosa.StreamBridge`.

    Instead of loading the entire audio signal into memory (as
    in `load`), this function produces *blocks* of audio spanning
    a fixed number of frames at a specified frame length and hop
    length.

    While this function strives for similar behavior to `load`,
    there are a few caveats that users should be aware of:

        1. This function does not return audio buffers directly.
           It returns a generator, which you can iterate over
           to produce blocks of audio.  A *block*, in this context,
           refers to a buffer of audio which spans a given number of
           (potentially overlapping) frames.
        2. Automatic sample-rate conversion is supported,
           but not all combinations of ``block_length``, ``frame_length``, and
           ``hop_length`` will be compatible with all sampling rates.
           Specifically, ``(block_length * hop_length * native_sr) / sr``
           must evaluate to an exact integer, where ``native_sr`` is the original
           sampling rate of the stream prior to resampling.
        3. Many analyses require access to the entire signal
           to behave correctly, such as `resample`, `cqt`, or
           `beat_track`, so these methods will not be appropriate
           for streamed data.
        4. The ``block_length`` parameter specifies how many frames
           of audio will be produced per block.  Larger values will
           consume more memory, but will be more efficient to process
           down-stream.  The best value will ultimately depend on your
           application and other system constraints.
        5. By default, most librosa analyses (e.g., short-time Fourier
           transform) assume centered frames, which requires padding the
           signal at the beginning and end.  This will not work correctly
           when the signal is carved into blocks, because it would introduce
           padding in the middle of the signal.  To disable this feature,
           use ``center=False`` in all frame-based analyses.
        6. If you break out of the generator loop early, the underlying audio
           file handle and resampling buffers will remain open until the
           generator object is garbage-collected. To explicitly release resources,
           call the generator's ``.close()`` method.

    See the examples below for proper usage of this function.

    Parameters
    ----------
    path : str, int, os.PathLike, sf.SoundFile, file-like object, or librosa.StreamBridge
        Path to the input file or a `librosa.StreamBridge` instance to stream.

        Any codec supported by `soundfile` is permitted here.
        An existing `soundfile.SoundFile` object may also be provided.

        Alternatively, a `librosa.StreamBridge` receiving live audio chunks
        may be provided.

    block_length : int > 0
        The number of frames to include in each block.

        Note that at the end of the file, there may not be enough
        data to fill an entire block, resulting in a shorter block
        by default.  To pad the signal out so that blocks are always
        full length, set ``fill_value`` (see below).

    frame_length : int > 0
        The number of samples per frame.

    hop_length : int > 0
        The number of samples to advance between frames.

        Note that when ``hop_length < frame_length``, neighboring frames
        will overlap.  Similarly, the last frame of one *block* will overlap
        with the first frame of the next *block*.

    sr : number > 0 [scalar], default=22050
        target sampling rate.
        'None' uses the native sampling rate of the file or `StreamBridge`.

    mono : bool
        Convert the signal to mono during streaming

    offset : float
        Start reading after this time (in seconds)

        If negative, it will be interpreted relative to the end of the file
        (files only).

    duration : float
        Only load up to this much audio (in seconds)

    warm_up : bool, default=False
        If True, pre-fill the buffer with silence (zeros) so that the first
        incoming chunk of size ``block_length * hop_length`` immediately yields
        a full analysis block without startup delay.

    fill_value : float [optional]
        If padding the signal to produce constant-length blocks,
        this value will be used at the end of the signal.

        In most cases, ``fill_value=0`` (silence) is expected, but
        you may specify any value here.

    res_type : str
        Resample type, must be one of the following:

        'soxr_vhq', 'soxr_hq', 'soxr_mq' or 'soxr_lq'
            `soxr` Very high-, High-, Medium-, Low-quality FFT-based bandlimited interpolation.
            ``'soxr_hq'`` is the default setting of `soxr`.
        'soxr_qq'
            `soxr` Quick cubic interpolation (very fast, but not bandlimited)
    dtype : numeric type
        data type of audio buffers to be produced

    Yields
    ------
    y : np.ndarray
        An audio buffer of (at most)
        ``(block_length-1) * hop_length + frame_length`` samples.

    See Also
    --------
    StreamBridge
    load
    get_samplerate
    soundfile.blocks

    Examples
    --------
    Apply a short-term Fourier transform to blocks of 256 frames
    at a time.  Note that streaming operation requires left-aligned
    frames, so we must set ``center=False`` to avoid padding artifacts.

    >>> filename = librosa.ex("brahms")
    >>> stream = librosa.stream(
    ...     filename, block_length=256, frame_length=4096, hop_length=1024
    ... )
    >>> for y_block in stream:
    ...     D_block = librosa.stft(y_block, center=False)

    Or compute a mel spectrogram over a stream in its native sampling rate,
    using a shorter frame and non-overlapping windows

    >>> filename = librosa.ex("brahms")
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(
    ...     filename, block_length=256, frame_length=2048, hop_length=2048, sr=None
    ... )
    >>> for y_block in stream:
    ...     m_block = librosa.feature.melspectrogram(
    ...         y=y_block, sr=sr, n_fft=2048, hop_length=2048, center=False
    ...     )
    """
    if isinstance(path, StreamBridge):
        yield from path.stream(
            block_length=block_length,
            frame_length=frame_length,
            hop_length=hop_length,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration,
            warm_up=warm_up,
            fill_value=fill_value,
            res_type=res_type,
            dtype=dtype,
        )
        return

    is_file = isinstance(path, (str, os.PathLike, int, sf.SoundFile)) or hasattr(path, "read")
    if not is_file:
        raise ParameterTypeError(f"Unsupported source type for stream: {type(path)}")

    context = nullcontext(path) if isinstance(path, sf.SoundFile) else sf.SoundFile(path)

    with context as sfo:
        native_sr = sfo.samplerate
        num_channels = sfo.channels
        buf = _StreamBuffer(
            frame_length=frame_length,
            hop_length=hop_length,
            block_length=block_length,
            orig_sr=native_sr,
            sr=sr,
            channels=num_channels,
            mono=mono,
            offset=0.0,
            duration=duration,
            warm_up=warm_up,
            fill_value=fill_value,
            res_type=res_type,
            dtype=dtype,
        )
        read_frames = int(duration * native_sr) if duration is not None else -1

        if offset >= 0:
            sfo.seek(int(offset * native_sr))
        else:
            sfo.seek(-int(abs(offset) * native_sr), whence=sf.SEEK_END)

        chunk_iter = (
            chunk.T
            for chunk in sfo.blocks(
                blocksize=buf.orig_read_size,
                overlap=0,
                dtype=dtype,
                always_2d=False,
                frames=read_frames,
            )
        )
        yield from buf.process_stream(chunk_iter)
