#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming utilities for audio I/O bridging and sliding window buffering."""
from __future__ import annotations

import asyncio
import queue
import threading
import warnings
from collections import deque
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import numpy as np
import soxr

from .exceptions import ParameterError
from .utils import buf_to_float, is_positive_int

if TYPE_CHECKING:
    from typing import AsyncGenerator, Generator

    from numpy.typing import DTypeLike
    from typing_extensions import Self


__all__ = [
    "StreamBridge",
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
        if not is_positive_int(channels):
            raise ParameterError(f"channels={channels} must be a positive integer")
        if offset < 0:
            raise ParameterError(f"offset={offset} must be non-negative")
        if duration is not None and duration < 0:
            raise ParameterError(f"duration={duration} must be non-negative")
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
        self.samples_remaining = (
            int(self.duration * self.orig_sr) if self.duration is not None else None
        )

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

        # Downmix or transpose from canonical (channels, samples) to (samples, channels)
        if chunk.ndim > 1:
            y = (
                np.mean(chunk, axis=0, dtype=self.dtype)
                if self.mono
                else chunk.T
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

        # Truncate to duration if specified
        if self.samples_remaining is not None:
            if y.shape[0] >= self.samples_remaining:
                y = y[: self.samples_remaining]
                self.samples_remaining = 0
            else:
                self.samples_remaining -= y.shape[0]
            if y.shape[0] == 0:
                return []

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
            block = self.buffer[self.read_idx : self.read_idx + self.target_yield_size]
            blocks.append(block.T.copy())
            self.read_idx += self.target_advance

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
            remainder = np.concatenate(final_data_list, axis=0)
        else:
            remainder = self.buffer[self.read_idx : self.write_idx]

        rem_idx = 0
        while rem_idx < remainder.shape[0]:
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

        self.read_idx = self.write_idx
        return blocks

    def is_complete(self) -> bool:
        """Check if the maximum input duration has been reached."""
        return self.samples_remaining is not None and self.samples_remaining <= 0


class StreamBridge:
    """Universal I/O bridge connecting audio callbacks into `librosa.stream` and `librosa.astream`.

    `StreamBridge` acts as an audio callback (compatible with PortAudio,
    :mod:`sounddevice`, and ``PyAudio``) that enqueues incoming raw audio chunks,
    and simultaneously acts as both a synchronous and an asynchronous iterator
    of Librosa-canonical NumPy arrays.

    Parameters
    ----------
    sr : float > 0
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
    Bridge real-time input from :mod:`sounddevice` and stream analysis blocks:

    >>> import sounddevice as sd  # doctest: +SKIP
    >>> with librosa.util.StreamBridge(sr=48000, channels=1) as bridge:  # doctest: +SKIP
    ...     with sd.InputStream(samplerate=48000, channels=1, callback=bridge):
    ...         for i, block in enumerate(bridge.stream(
    ...             block_length=256, frame_length=2048, hop_length=512
    ...         )):
    ...             centroid = librosa.feature.spectral_centroid(y=block)
    ...             print(f"Block {i}: mean centroid = {centroid.mean():.1f} Hz")
    ...             if i >= 10:
    ...                 break
    """

    def __init__(
        self,
        *,
        sr: float,
        channels: int = 1,
        channels_first: bool = False,
        dtype: DTypeLike = np.float32,
        maxsize: int = 0,
        overflow_policy: str = "warn",
        timeout: float | None = None,
    ):
        if not (np.isfinite(sr) and sr > 0):
            raise ParameterError(f"sr={sr} must be a positive number")
        if not is_positive_int(channels):
            raise ParameterError(f"channels={channels} must be a positive integer")
        if maxsize < 0:
            raise ParameterError(f"maxsize={maxsize} must be non-negative")
        if overflow_policy not in {"warn", "drop_oldest", "drop_newest", "error"}:
            raise ParameterError(f"Invalid overflow_policy='{overflow_policy}'")

        self.sr = float(sr)
        self.channels = int(channels)
        self.channels_first = bool(channels_first)
        self.dtype = np.dtype(dtype)
        self.maxsize = int(maxsize)
        self.overflow_policy = overflow_policy
        self.timeout = timeout

        self._buffer: deque[np.ndarray] = deque()
        self._cond = threading.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_event: asyncio.Event | None = None
        self._closed = False

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
            converted to float32 according to `dtype`. Single-channel audio is
            normalized to 1D shape ``(samples,)``, and multi-channel audio is
            normalized to shape ``(channels, samples)``.
        """
        chunk = self._normalize(data)

        with self._cond:
            if self._closed:
                raise RuntimeError("Cannot push data to a closed StreamBridge.")

            if self.maxsize > 0 and len(self._buffer) >= self.maxsize:
                if self.overflow_policy == "drop_oldest":
                    self._buffer.popleft()
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
                    self._buffer.popleft()

            self._buffer.append(chunk)
            self._cond.notify()
            if self._loop is not None and self._async_event is not None:
                self._loop.call_soon_threadsafe(self._async_event.set)

    def _normalize(self, data: np.ndarray | bytes | memoryview) -> np.ndarray:
        """Normalize any incoming audio representation into Librosa canonical float32."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            raw = np.frombuffer(data, dtype=self.dtype)
        elif isinstance(data, np.ndarray):
            raw = data
        else:
            raise ParameterError(f"Unsupported audio data type: {type(data)}")

        if np.issubdtype(raw.dtype, np.integer):
            arr = buf_to_float(raw, n_bytes=raw.dtype.itemsize, dtype=np.float32)
        else:
            arr = raw.astype(np.float32, copy=True)

        if self.channels == 1:
            return arr.reshape(-1)

        if arr.ndim == 1:
            if not self.channels_first:
                arr = arr.reshape(-1, self.channels).T
            else:
                arr = arr.reshape(self.channels, -1)
        elif arr.ndim == 2 and not self.channels_first:
            arr = arr.T

        return arr

    def __iter__(self) -> Self:
        """Return synchronous iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next audio chunk from the buffer."""
        with self._cond:
            if not self._buffer and not self._closed:
                self._cond.wait_for(
                    lambda: bool(self._buffer) or self._closed,
                    timeout=self.timeout,
                )
            if self._buffer:
                return self._buffer.popleft()
            if self._closed:
                raise StopIteration
            raise queue.Empty

    def __aiter__(self) -> Self:
        """Return asynchronous iterator."""
        with self._cond:
            if self._async_event is None:
                self._loop = asyncio.get_running_loop()
                self._async_event = asyncio.Event()
                if self._buffer or self._closed:
                    self._async_event.set()
        return self

    async def __anext__(self) -> np.ndarray:
        """Asynchronously return the next audio chunk from the buffer."""
        if self._async_event is None:
            self.__aiter__()
        assert self._async_event is not None

        while True:
            with self._cond:
                if self._buffer:
                    return self._buffer.popleft()
                if self._closed:
                    raise StopAsyncIteration
                self._async_event.clear()
            await self._async_event.wait()

    def stream(
        self,
        *,
        block_length: int,
        frame_length: int,
        hop_length: int,
        sr: float | None = None,
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
        sr : float > 0, optional
            Target sampling rate. If None, defaults to `self.sr`.
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
        astream
        """
        buf = _StreamBuffer(
            frame_length=frame_length,
            hop_length=hop_length,
            block_length=block_length,
            orig_sr=self.sr,
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
        for chunk in self:
            yield from buf.push(chunk)
            if buf.is_complete():
                break
        yield from buf.flush()

    async def astream(
        self,
        *,
        block_length: int,
        frame_length: int,
        hop_length: int,
        sr: float | None = None,
        mono: bool = True,
        offset: float = 0.0,
        duration: float | None = None,
        warm_up: bool = False,
        fill_value: float | None = None,
        res_type: str = "soxr_hq",
        dtype: DTypeLike = np.float32,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Asynchronously stream audio from this bridge in fixed-length analysis buffers.

        This method is equivalent to calling ``librosa.astream(self, ...)``.

        Parameters
        ----------
        block_length : int > 0
            The number of frames to include in each block.
        frame_length : int > 0
            The number of samples per frame.
        hop_length : int > 0
            The number of samples to advance between frames.
        sr : float > 0, optional
            Target sampling rate. If None, defaults to `self.sr`.
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
        librosa.astream
        stream
        """
        buf = _StreamBuffer(
            frame_length=frame_length,
            hop_length=hop_length,
            block_length=block_length,
            orig_sr=self.sr,
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
        async for chunk in self:
            for block in buf.push(chunk):
                yield block
            if buf.is_complete():
                break
        for block in buf.flush():
            yield block

    def close(self) -> None:
        """Close the bridge and signal termination to waiting consumers."""
        with self._cond:
            if not self._closed:
                self._closed = True
                self._cond.notify_all()
                if self._loop is not None and self._async_event is not None:
                    with suppress(RuntimeError):
                        self._loop.call_soon_threadsafe(self._async_event.set)

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
