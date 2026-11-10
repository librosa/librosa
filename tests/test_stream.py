#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for streaming DSP and I/O bridge components."""
from __future__ import annotations

import asyncio
from unittest import mock

import numpy as np
import pytest

import librosa
from librosa.util.streaming import StreamBridge, StreamBuffer, async_capture


def test_streaming_imports():
    """Test submodule exports and top-level aliases."""
    import librosa.util.streaming as streaming

    assert streaming.StreamBuffer is StreamBuffer
    assert streaming.StreamBridge is StreamBridge
    assert streaming.async_capture is async_capture

    assert librosa.util.StreamBuffer is StreamBuffer
    assert librosa.util.StreamBridge is StreamBridge
    assert librosa.util.async_capture is async_capture

    assert librosa.StreamBuffer is StreamBuffer


def test_streambuffer_basic():
    """Test basic StreamBuffer push, frame framing, and flush."""
    # Frame length 1024, hop 512, block_length 1
    buf = librosa.StreamBuffer(frame_length=1024, hop_length=512, orig_sr=22050)
    chunk1 = np.ones(512, dtype=np.float32)
    chunk2 = np.ones(512, dtype=np.float32) * 2

    blocks1 = buf.push(chunk1)
    assert len(blocks1) == 0  # 512 samples < 1024 frame length

    blocks2 = buf.push(chunk2)
    assert len(blocks2) == 1
    assert blocks2[0].shape == (1024,)
    assert np.allclose(blocks2[0][:512], 1.0)
    assert np.allclose(blocks2[0][512:], 2.0)

    # Flush should yield remaining partial block if fill_value is given
    buf.fill_value = 0.0
    tail_blocks = buf.flush()
    assert len(tail_blocks) == 1
    assert tail_blocks[0].shape == (1024,)
    assert np.allclose(tail_blocks[0][:512], 2.0)
    assert np.allclose(tail_blocks[0][512:], 0.0)


def test_streambuffer_resampling():
    """Test StreamBuffer with online resampling."""
    orig_sr = 44100
    sr = 22050
    buf = librosa.StreamBuffer(
        frame_length=1024,
        hop_length=512,
        block_length=1,
        orig_sr=orig_sr,
        sr=sr,
    )

    # 10 chunks of 1024 samples at 44.1k = ~5120 samples at 22.05k
    rng = np.random.default_rng(0)
    all_blocks = []
    for _ in range(10):
        chunk = rng.standard_normal(1024).astype(np.float32)
        all_blocks.extend(buf.push(chunk))

    all_blocks.extend(buf.flush())
    assert len(all_blocks) > 0
    for b in all_blocks:
        assert b.shape[-1] <= 1024


def test_streambuffer_multichannel():
    """Test StreamBuffer with multichannel mono conversion and stereo preservation."""
    # Stereo input (2 channels, 512 samples)
    buf_mono = librosa.StreamBuffer(
        frame_length=1024, hop_length=512, channels=2, mono=True
    )
    chunk = np.ones((2, 512), dtype=np.float32)
    chunk[0] = 1.0
    chunk[1] = 3.0

    buf_mono.push(chunk)
    blocks = buf_mono.push(chunk)
    assert len(blocks) == 1
    assert blocks[0].ndim == 1
    assert np.allclose(blocks[0], 2.0)  # Average of 1.0 and 3.0

    # Stereo preserving
    buf_stereo = librosa.StreamBuffer(
        frame_length=1024, hop_length=512, channels=2, mono=False
    )
    buf_stereo.push(chunk)
    blocks_st = buf_stereo.push(chunk)
    assert len(blocks_st) == 1
    assert blocks_st[0].shape == (2, 1024)
    assert np.allclose(blocks_st[0][0], 1.0)
    assert np.allclose(blocks_st[0][1], 3.0)


def test_streambuffer_reset():
    """Test StreamBuffer reset functionality."""
    buf = librosa.StreamBuffer(frame_length=1024, hop_length=512)
    buf.push(np.ones(1024, dtype=np.float32))
    assert buf.yielded_samples > 0
    buf.reset()
    assert buf.write_idx == 0
    assert buf.read_idx == 0
    assert buf.yielded_samples == 0


def test_stream_with_iterable():
    """Test librosa.stream with an arbitrary chunk iterator."""
    def chunk_generator():
        for i in range(10):
            yield np.ones(512, dtype=np.float32) * i

    blocks = list(
        librosa.stream(
            chunk_generator(),
            block_length=1,
            frame_length=1024,
            hop_length=512,
            orig_sr=22050,
            sr=22050,
        )
    )

    assert len(blocks) > 0
    for b in blocks:
        assert b.shape[-1] <= 1024


def test_streambridge_sync():
    """Test StreamBridge synchronous push, drain, and iteration."""
    bridge = librosa.util.StreamBridge(orig_sr=22050, channels=1)
    assert bridge.empty

    # Simulate sounddevice callback
    bridge(np.ones(512, dtype=np.float32))
    assert not bridge.empty
    drained = bridge.drain()
    assert len(drained) == 1
    assert bridge.empty

    bridge(np.ones(512, dtype=np.float32) * 2)
    bridge.close()

    # Collect chunks from bridge iterator
    chunks = list(bridge)
    assert len(chunks) == 1
    assert np.allclose(chunks[0], 2.0)


def test_streambridge_pcm_bytes():
    """Test StreamBridge decoding of raw 16-bit PCM bytes."""
    bridge = librosa.util.StreamBridge(
        orig_sr=22050, channels=1, in_dtype=np.int16
    )
    int_data = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    bridge(int_data.tobytes())
    bridge.close()

    chunks = list(bridge)
    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert np.isclose(chunks[0][1], 0.5, atol=1e-3)
    assert np.isclose(chunks[0][2], -0.5, atol=1e-3)


def test_streambridge_channel_transposition():
    """Test StreamBridge interleaving and channel transposition."""
    bridge = librosa.util.StreamBridge(
        orig_sr=22050, channels=2, channels_first=False
    )
    data = np.zeros((512, 2), dtype=np.float32)
    data[:, 0] = 1.0
    data[:, 1] = 2.0
    bridge(data)
    bridge.close()

    chunks = list(bridge)
    assert len(chunks) == 1
    assert chunks[0].shape == (2, 512)
    assert np.allclose(chunks[0][0], 1.0)
    assert np.allclose(chunks[0][1], 2.0)


def test_streambridge_overflow_policy():
    """Test StreamBridge buffer overflow policies."""
    # Drop oldest policy
    bridge = librosa.util.StreamBridge(
        orig_sr=22050, maxsize=2, overflow_policy="drop_oldest"
    )
    bridge.push(np.array([1.0], dtype=np.float32))
    bridge.push(np.array([2.0], dtype=np.float32))
    bridge.push(np.array([3.0], dtype=np.float32))
    bridge.close()

    chunks = list(bridge)
    assert len(chunks) == 2
    assert chunks[0][0] == 2.0
    assert chunks[1][0] == 3.0

    # Error policy
    bridge_err = librosa.util.StreamBridge(
        orig_sr=22050, maxsize=1, overflow_policy="error"
    )
    bridge_err.push(np.array([1.0], dtype=np.float32))
    with pytest.raises(BufferError):
        bridge_err.push(np.array([2.0], dtype=np.float32))


def test_streambridge_with_librosa_stream():
    """Test connecting StreamBridge directly into librosa.stream."""
    bridge = librosa.util.StreamBridge(orig_sr=22050, channels=1)

    for i in range(8):
        bridge(np.ones(512, dtype=np.float32) * i)
    bridge.close()

    blocks = list(
        librosa.stream(
            bridge, block_length=1, frame_length=1024, hop_length=512, sr=22050
        )
    )
    assert len(blocks) == 8


def test_astream_and_async_bridge():
    """Test librosa.astream asynchronous consumption with StreamBridge."""
    async def _test():
        bridge = librosa.util.StreamBridge(orig_sr=22050, channels=1)

        # Push chunks into bridge
        for i in range(6):
            bridge(np.ones(512, dtype=np.float32) * i)
        bridge.close()

        blocks = []
        async for block in librosa.astream(
            bridge, block_length=1, frame_length=1024, hop_length=512, sr=22050
        ):
            blocks.append(block)

        assert len(blocks) == 6
        assert blocks[0].shape == (1024,)

    asyncio.run(_test())


def test_async_capture_mock():
    """Test async_capture context manager lifecycle."""
    async def _test():
        mock_sd = mock.MagicMock()
        mock_input_stream = mock.MagicMock()
        mock_sd.InputStream.return_value = mock_input_stream

        with mock.patch.dict("sys.modules", {"sounddevice": mock_sd}):
            async with librosa.util.async_capture(
                samplerate=22050, channels=1
            ) as stream:
                assert isinstance(stream, librosa.util.StreamBridge)
                mock_sd.InputStream.assert_called_once()
                mock_input_stream.__enter__.assert_called_once()

            mock_input_stream.__exit__.assert_called_once()
            assert stream._closed

    asyncio.run(_test())
