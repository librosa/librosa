#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming utilities for audio I/O bridging, sliding window buffering, and asynchronous capture."""
from __future__ import annotations

import asyncio
import importlib
import queue
import warnings
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, AsyncIterator

import numpy as np
import soxr

from .exceptions import ParameterError
from .utils import buf_to_float, is_positive_int

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from typing_extensions import Self


__all__ = [
    "StreamBuffer",
    "StreamBridge",
    "async_capture",
]


def _align_step_size(
    target_step: int, target_sr: float | None, orig_sr: float
) -> int:
    """Ensure that the target step size remains integer-valued in the original sampling rate."""
    if target_sr is None or target_sr == orig_sr:
        return target_step

    exact_native_step = (target_step * orig_sr) / target_sr
    aligned_native_step = np.round(exact_native_step)

    # Tolerances safely absorb precision errors from float arithmetic
    if not np.isclose(exact_native_step, aligned_native_step, rtol=1e-7, atol=1e-5):
        raise ParameterError(
            f"Target block step of {target_step} samples at {target_sr} results in a "
            f"fractional sample advance at original sampling rate {orig_sr}."
        )

    return int(aligned_native_step)


class StreamBuffer:
    """Sliding analysis window accumulator and streaming DSP processor.

    `StreamBuffer` maintains a sliding observation buffer for real-time and
    streaming audio processing. It handles on-the-fly sample rate conversion
    via `soxr.ResampleStream`, multi-channel downmixing to mono, offset skipping,
    and duration bounds.

    Parameters
    ----------
    frame_length : int > 0
        The number of samples per frame.
    hop_length : int > 0
        The number of samples to advance between successive frames.
    block_length : int > 0, default=1
        The number of frames to include in each yielded block.
    orig_sr : float > 0, default=22050
        Native sample rate of incoming audio chunks.
    sr : float > 0, optional
        Target sample rate for analysis. If None, defaults to `orig_sr`.
    channels : int > 0, default=1
        Number of audio channels in incoming chunks.
    mono : bool, default=True
        If True, convert multi-channel signals to mono.
    offset : float >= 0, default=0.0
        Time (in seconds) to skip before yielding frames.
    duration : float > 0, optional
        Maximum time (in seconds) of audio to yield.
    warm_up : bool, default=False
        If True, pre-fill the buffer with silence (zeros) so that the first
        incoming chunk immediately yields an analysis block without delay.
    fill_value : float, optional
        If provided, pad the final incomplete block to full length on `flush`.
    res_type : str, default='soxr_hq'
        Soxr resampling mode ('soxr_vhq', 'soxr_hq', 'soxr_mq', 'soxr_lq', 'soxr_qq').
    dtype : numeric type, default=np.float32
        Data type of the output blocks.

    Examples
    --------
    Process incoming blocks of audio with a sliding window:

    >>> buf = librosa.util.streaming.StreamBuffer(frame_length=1024, hop_length=512, orig_sr=22050)
    >>> for chunk in [np.ones(512, dtype=np.float32), np.ones(512, dtype=np.float32)]:
    ...     blocks = buf.push(chunk)
    ...     for block in blocks:
    ...         print(block.shape)
    (1024,)
    """

    def __init__(
        self,
        *,
        frame_length: int,
        hop_length: int,
        block_length: int = 1,
        orig_sr: float = 22050,
        sr: float | None = None,
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
            raise ParameterError(
                f"block_length={block_length} must be a positive integer"
            )
        if not is_positive_int(frame_length):
            raise ParameterError(
                f"frame_length={frame_length} must be a positive integer"
            )
        if not is_positive_int(hop_length):
            raise ParameterError(f"hop_length={hop_length} must be a positive integer")
        if not (np.isfinite(orig_sr) and orig_sr > 0):
            raise ParameterError(f"orig_sr={orig_sr} must be a positive number")
        if sr is not None and not (np.isfinite(sr) and sr > 0):
            raise ParameterError(f"sr={sr} must be a positive number")
        if res_type not in {"soxr_vhq", "soxr_hq", "soxr_mq", "soxr_lq", "soxr_qq"}:
            raise ParameterError(
                f"res_type={res_type} is not a valid soxr resampling mode for streaming"
            )

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

        self.target_yield_size = (
            self.block_length - 1
        ) * self.hop_length + self.frame_length
        self.target_advance = self.block_length * self.hop_length
        self.process_channels = 1 if self.mono else self.channels

        self.needs_resampling = (sr is not None) and (self.orig_sr != self.sr)
        if self.needs_resampling:
            _align_step_size(self.target_advance, self.sr, self.orig_sr)
            self.resampler = soxr.ResampleStream(
                in_rate=self.orig_sr,
                out_rate=self.sr,
                num_channels=self.process_channels,
                dtype=self.dtype,
                quality=res_type,
            )
        else:
            self.resampler = None

        self.capacity = self.target_yield_size + (self.target_advance * 2)
        buffer_shape = (
            (self.capacity,)
            if self.process_channels == 1
            else (self.capacity, self.process_channels)
        )
        self.buffer = np.zeros(buffer_shape, dtype=self.dtype)
        self.warm_up = bool(warm_up)
        self.prefill = (
            max(0, self.target_yield_size - self.target_advance)
            if self.warm_up
            else 0
        )
        self.write_idx = self.prefill
        self.read_idx = 0

        self.samples_to_skip = (
            int(self.offset * self.orig_sr) if self.offset > 0 else 0
        )
        self.yielded_samples = 0
        self.max_samples = (
            int(self.duration * self.sr) if self.duration is not None else None
        )

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Ingest a raw audio chunk and return any ready analysis blocks.

        Parameters
        ----------
        chunk : np.ndarray
            Incoming audio chunk of shape ``(samples,)``, ``(channels, samples)``,
            or ``(samples, channels)``.

        Returns
        -------
        blocks : list of np.ndarray
            List of complete analysis blocks of size `target_yield_size`.
        """
        if self.is_complete() or chunk.size == 0:
            return []

        # Downmix / channel alignment
        if chunk.ndim > 1:
            if chunk.shape[0] == self.channels:
                y = (
                    np.mean(chunk, axis=0, dtype=chunk.dtype)
                    if self.mono
                    else chunk.T
                )
            else:
                y = (
                    np.mean(chunk, axis=-1, dtype=chunk.dtype)
                    if self.mono
                    else chunk
                )
        else:
            y = chunk

        if y.dtype != self.dtype:
            y = y.astype(self.dtype)

        # Skip initial offset samples
        if self.samples_to_skip > 0:
            n_samples = y.shape[0]
            if n_samples <= self.samples_to_skip:
                self.samples_to_skip -= n_samples
                return []
            y = y[self.samples_to_skip :]
            self.samples_to_skip = 0

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
                new_shape = (
                    (self.capacity,)
                    if self.process_channels == 1
                    else (self.capacity, self.process_channels)
                )
                new_buffer = np.zeros(new_shape, dtype=self.dtype)
                new_buffer[: self.write_idx] = self.buffer[: self.write_idx]
                self.buffer = new_buffer

        self.buffer[self.write_idx : self.write_idx + n_incoming] = target_chunk
        self.write_idx += n_incoming

        # Slice all complete blocks
        blocks = []
        while self.write_idx - self.read_idx >= self.target_yield_size:
            if self.is_complete():
                break
            block = self.buffer[self.read_idx : self.read_idx + self.target_yield_size]
            blocks.append(block.T.copy())
            self.read_idx += self.target_advance
            self.yielded_samples += self.target_advance

        return blocks

    def flush(self) -> list[np.ndarray]:
        """Flush resampler delay buffers and yield final (optionally padded) blocks.

        Returns
        -------
        blocks : list of np.ndarray
            Remaining complete or padded analysis blocks.
        """
        blocks = []
        if self.resampler is not None:
            empty = np.empty(
                (0,) if self.process_channels == 1 else (0, self.process_channels),
                dtype=self.dtype,
            )
            tail = self.resampler.resample_chunk(empty, last=True)
        else:
            tail = None

        final_data_list = [self.buffer[self.read_idx : self.write_idx]]
        if tail is not None and tail.shape[0] > 0:
            final_data_list.append(tail)
            remainder = (
                np.concatenate(final_data_list, axis=0)
                if final_data_list
                else np.array([])
            )
        else:
            remainder = self.buffer[self.read_idx : self.write_idx]

        rem_idx = 0
        while rem_idx < remainder.shape[0]:
            if self.is_complete():
                break
            current_slice = remainder[rem_idx : rem_idx + self.target_yield_size]
            if current_slice.shape[0] < self.target_yield_size:
                pad_length = self.target_yield_size - current_slice.shape[0]
                pad_width = (
                    (0, pad_length)
                    if self.process_channels == 1
                    else ((0, pad_length), (0, 0))
                )
                if self.fill_value is not None:
                    current_slice = np.pad(
                        current_slice,
                        pad_width,
                        mode="constant",
                        constant_values=self.fill_value,
                    )
            blocks.append(current_slice.T.copy())
            rem_idx += self.target_advance
            self.yielded_samples += self.target_advance

        self.read_idx = self.write_idx
        return blocks

    def reset(self) -> None:
        """Reset internal buffer state for replay or animation restarts."""
        self.buffer[:] = 0
        self.write_idx = self.prefill
        self.read_idx = 0
        self.yielded_samples = 0
        self.samples_to_skip = (
            int(self.offset * self.orig_sr) if self.offset > 0 else 0
        )
        if self.resampler is not None:
            self.resampler.clear()

    def is_complete(self) -> bool:
        """Check if the maximum duration has been reached."""
        return self.max_samples is not None and self.yielded_samples >= self.max_samples


class StreamBridge:
    """Universal I/O bridge connecting audio callbacks into NumPy chunk iterators.

    `StreamBridge` acts as an audio callback (compatible with PortAudio,
    :mod:`sounddevice`, and ``PyAudio``) that enqueues incoming raw audio chunks,
    and simultaneously acts as both a synchronous and an asynchronous iterator
    of Librosa-canonical NumPy arrays.

    Parameters
    ----------
    orig_sr : float > 0
        Native sample rate of the incoming audio stream.
    channels : int > 0, default=1
        Number of audio channels in the incoming stream.
    channels_first : bool, default=False
        If True, incoming multi-channel buffers are shaped ``(channels, samples)``.
        If False, incoming multi-channel buffers are shaped ``(samples, channels)``
        (standard for :mod:`sounddevice` and PortAudio) and will be transposed to
        canonical shape ``(channels, samples)``.
    in_dtype : np.dtype, default=np.float32
        Data type of incoming audio chunks or byte buffers. If integer PCM,
        samples are normalized to floating-point ``[-1.0, 1.0]`` using `buf_to_float`.
    maxsize : int >= 0, default=0
        Maximum capacity of the internal chunk buffer. If 0, capacity is unbounded.
    overflow_policy : {'warn', 'drop_oldest', 'drop_newest', 'error'}, default='warn'
        Action taken when `maxsize` is reached:
            - ``"warn"``: Discard the oldest queued chunk and emit a warning.
            - ``"drop_oldest"``: Discard the oldest queued chunk (fixed-latency ring buffer).
            - ``"drop_newest"``: Discard the incoming chunk.
            - ``"error"``: Raise `BufferError`.
    timeout : float | None, default=None
        Maximum time in seconds to wait for a chunk during synchronous iteration
        before raising `queue.Empty`. If ``None``, blocks indefinitely.

    Examples
    --------
    Bridge real-time input from :mod:`sounddevice` into `librosa.stream`:

    >>> import sounddevice as sd  # doctest: +SKIP
    >>> bridge = librosa.util.StreamBridge(orig_sr=48000, channels=1)  # doctest: +SKIP
    >>> with sd.InputStream(samplerate=48000, channels=1, callback=bridge):  # doctest: +SKIP
    ...     for block in librosa.stream(bridge, frame_length=2048, hop_length=512):  # doctest: +SKIP
    ...         centroid = librosa.feature.spectral_centroid(y=block)  # doctest: +SKIP
    """

    def __init__(
        self,
        orig_sr: float,
        *,
        channels: int = 1,
        channels_first: bool = False,
        in_dtype: DTypeLike = np.float32,
        maxsize: int = 0,
        overflow_policy: str = "warn",
        timeout: float | None = None,
    ):
        if not (np.isfinite(orig_sr) and orig_sr > 0):
            raise ParameterError(f"orig_sr={orig_sr} must be a positive number")
        if channels < 1:
            raise ParameterError(f"channels={channels} must be a positive integer")
        if overflow_policy not in {"warn", "drop_oldest", "drop_newest", "error"}:
            raise ParameterError(f"Invalid overflow_policy='{overflow_policy}'")

        self.orig_sr = float(orig_sr)
        self.samplerate = self.orig_sr  # Duck-typing compatibility with SoundFile
        self.channels = int(channels)
        self.channels_first = bool(channels_first)
        self.in_dtype = np.dtype(in_dtype)
        self.maxsize = int(maxsize)
        self.overflow_policy = overflow_policy
        self.timeout = timeout

        self._sync_queue: queue.Queue = queue.Queue()
        self._async_queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._sentinel = object()

    def __call__(
        self, data: np.ndarray | bytes, *_args: Any, **_kwargs: Any
    ) -> tuple[None, int]:
        """Universal audio callback.

        Compatible with :mod:`sounddevice` (which ignores the return value) and
        ``PyAudio`` (which expects a tuple ``(out_data, flag)`` where flag 0
        corresponds to ``paContinue``).
        """
        self.push(data)
        return (None, 0)

    def push(self, data: np.ndarray | bytes | memoryview) -> None:
        """Push a raw audio chunk into the bridge.

        Parameters
        ----------
        data : np.ndarray, bytes, or memoryview
            Incoming audio buffer. Integer arrays or byte buffers are
            converted to float32 according to `in_dtype`. Multi-channel audio
            is converted to Librosa canonical shape ``(channels, samples)``.
        """
        if self._closed:
            raise RuntimeError("Cannot push data to a closed StreamBridge.")

        chunk = self._normalize(data)

        # Handle sync queue overflow
        if self.maxsize > 0 and self._sync_queue.qsize() >= self.maxsize:
            if self.overflow_policy == "drop_oldest":
                with suppress(queue.Empty):
                    self._sync_queue.get_nowait()
            elif self.overflow_policy == "drop_newest":
                return
            elif self.overflow_policy == "error":
                raise BufferError(
                    "StreamBridge queue overflow: consumer is slower than capture."
                )
            elif self.overflow_policy == "warn":
                warnings.warn(
                    "StreamBridge queue is full; dropping oldest chunk.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                with suppress(queue.Empty):
                    self._sync_queue.get_nowait()

        self._sync_queue.put_nowait(chunk)

        # Forward to async queue if active
        if self._async_queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._async_put, chunk)

    def _async_put(self, chunk: np.ndarray) -> None:
        """Enqueue into async queue within the event loop thread."""
        assert self._async_queue is not None
        if self.maxsize > 0 and self._async_queue.qsize() >= self.maxsize:
            if self.overflow_policy == "drop_oldest":
                with suppress(asyncio.QueueEmpty):
                    self._async_queue.get_nowait()
            elif self.overflow_policy == "drop_newest":
                return
            elif self.overflow_policy == "error":
                raise BufferError(
                    "StreamBridge async queue overflow: consumer is slower than capture."
                )
            elif self.overflow_policy == "warn":
                warnings.warn(
                    "StreamBridge async queue is full; dropping oldest chunk.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                with suppress(asyncio.QueueEmpty):
                    self._async_queue.get_nowait()

        self._async_queue.put_nowait(chunk)

    def _normalize(self, data: np.ndarray | bytes | memoryview) -> np.ndarray:
        """Normalize any incoming audio representation into Librosa canonical float32."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            if np.issubdtype(self.in_dtype, np.integer):
                arr = buf_to_float(
                    data, n_bytes=self.in_dtype.itemsize, dtype=np.float32
                )
            else:
                arr = np.frombuffer(data, dtype=self.in_dtype).astype(np.float32)
        elif isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.integer):
                arr = buf_to_float(data, n_bytes=data.dtype.itemsize, dtype=np.float32)
            else:
                arr = data.astype(np.float32, copy=True)
        else:
            raise ParameterError(f"Unsupported audio data type: {type(data)}")

        if self.channels > 1 and arr.ndim == 1:
            if not self.channels_first:
                arr = arr.reshape(-1, self.channels)
            else:
                arr = arr.reshape(self.channels, -1)

        if arr.ndim == 2 and not self.channels_first:
            arr = arr.T

        return arr

    @property
    def empty(self) -> bool:
        """Check if the internal queue is currently empty."""
        return self._sync_queue.empty()

    def get_nowait(self) -> np.ndarray:
        """Retrieve an audio chunk immediately without blocking.

        Raises
        ------
        StopIteration
            If the bridge is closed and no more data is available.
        queue.Empty
            If no audio chunk is currently available.
        """
        if self._closed and self._sync_queue.empty():
            raise StopIteration

        try:
            item = self._sync_queue.get_nowait()
        except queue.Empty:
            if self._closed:
                raise StopIteration from None
            raise

        if item is self._sentinel:
            raise StopIteration

        return item

    def drain(self) -> list[np.ndarray]:
        """Drain all currently available audio chunks without blocking.

        Returns
        -------
        chunks : list of np.ndarray
            List of buffered audio chunks retrieved from the queue.
        """
        chunks = []
        while True:
            try:
                chunks.append(self.get_nowait())
            except (queue.Empty, StopIteration):
                break
        return chunks

    def __iter__(self) -> Self:
        """Return synchronous iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next audio chunk from the queue."""
        if self._closed and self._sync_queue.empty():
            raise StopIteration

        try:
            item = self._sync_queue.get(timeout=self.timeout)
        except queue.Empty:
            if self._closed:
                raise StopIteration from None
            raise

        if item is self._sentinel:
            raise StopIteration

        return item

    def __aiter__(self) -> Self:
        """Return asynchronous iterator, initializing the async queue if needed."""
        if self._async_queue is None:
            self._loop = asyncio.get_running_loop()
            self._async_queue = asyncio.Queue()
            # Drain any items already present in the sync queue
            item = None
            while not self._sync_queue.empty():
                try:
                    item = self._sync_queue.get_nowait()
                    if self.maxsize > 0 and self._async_queue.qsize() >= self.maxsize:
                        with suppress(asyncio.QueueEmpty):
                            self._async_queue.get_nowait()
                    self._async_queue.put_nowait(item)
                except queue.Empty:
                    break
            if self._closed and (
                self._async_queue.empty() or item is not self._sentinel
            ):
                self._async_queue.put_nowait(self._sentinel)
        return self

    async def __anext__(self) -> np.ndarray:
        """Asynchronously return the next audio chunk from the queue."""
        assert self._async_queue is not None
        if self._closed and self._async_queue.empty():
            raise StopAsyncIteration
        item = await self._async_queue.get()
        if item is self._sentinel:
            raise StopAsyncIteration
        return item

    def close(self) -> None:
        """Close the bridge and signal termination to waiting consumers."""
        if not self._closed:
            self._closed = True
            self._sync_queue.put_nowait(self._sentinel)
            if self._async_queue is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._async_queue.put_nowait, self._sentinel
                )

    def reset(self) -> None:
        """Reset internal queue state for replay or animation restarts."""
        self._closed = False
        while not self._sync_queue.empty():
            with suppress(queue.Empty):
                self._sync_queue.get_nowait()

        if self._async_queue is not None:
            while not self._async_queue.empty():
                with suppress(asyncio.QueueEmpty):
                    self._async_queue.get_nowait()

    def __enter__(self) -> Self:
        """Enter synchronous context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit synchronous context manager, closing the bridge."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter asynchronous context manager."""
        self.__aiter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit asynchronous context manager, closing the bridge."""
        self.close()


@asynccontextmanager
async def async_capture(
    samplerate: float = 22050,
    *,
    channels: int = 1,
    blocksize: int = 512,
    device: int | str | None = None,
    dtype: DTypeLike = "float32",
    maxsize: int = 0,
    overflow_policy: str = "warn",
) -> AsyncIterator[StreamBridge]:
    """Asynchronous context manager for live audio capture using :mod:`sounddevice`.

    Parameters
    ----------
    samplerate : float, default=22050
        Target sample rate of the capture stream.
    channels : int, default=1
        Number of input channels.
    blocksize : int, default=512
        Number of frames passed to each callback invocation.
    device : int, str, or None, default=None
        Capture device identifier passed to :class:`sounddevice.InputStream`.
        If None, selects the system default input device.
    dtype : str or np.dtype, default='float32'
        Data type expected from the input stream.
    maxsize : int >= 0, default=0
        Maximum capacity of the internal buffer.
    overflow_policy : str, default='warn'
        Policy for handling queue overflow.

    Yields
    ------
    stream : StreamBridge
        Asynchronously iterable audio stream bridge.

    Raises
    ------
    ImportError
        If :mod:`sounddevice` is not installed.

    Examples
    --------
    Asynchronously stream live microphone input:

    >>> async def capture_audio():  # doctest: +SKIP
    ...     async with librosa.util.async_capture(samplerate=22050) as stream:
    ...         async for block in librosa.astream(stream, frame_length=1024, hop_length=512):
    ...             centroid = librosa.feature.spectral_centroid(y=block)
    """
    try:
        sd = importlib.import_module("sounddevice")
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for async_capture. "
            "Install it via `pip install sounddevice` or `conda install -c conda-forge python-sounddevice`."
        ) from exc

    bridge = StreamBridge(
        orig_sr=samplerate,
        channels=channels,
        in_dtype=dtype,
        maxsize=maxsize,
        overflow_policy=overflow_policy,
    )
    bridge.__aiter__()

    input_stream = sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        blocksize=blocksize,
        device=device,
        dtype=dtype,
        callback=bridge,
    )

    with input_stream:
        try:
            yield bridge
        finally:
            bridge.close()
