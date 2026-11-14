#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for StreamBridge and librosa.stream."""
from __future__ import annotations

import queue
import threading

import numpy as np
import pytest
import soundfile as sf

import librosa
from librosa.core.streaming import StreamBridge


def test_streaming_imports():
    """Test submodule exports and core/top-level aliases."""
    import librosa.core.streaming as streaming

    assert streaming.StreamBridge is StreamBridge
    assert librosa.core.StreamBridge is StreamBridge
    assert librosa.StreamBridge is StreamBridge
    assert hasattr(librosa, "stream")
    assert hasattr(StreamBridge, "stream")

    # Internal buffer and removed helpers should not be exported publicly
    assert not hasattr(librosa, "astream")
    assert not hasattr(StreamBridge, "astream")
    assert not hasattr(librosa.util, "StreamBridge")
    assert not hasattr(librosa, "StreamBuffer")
    assert not hasattr(librosa.util, "StreamBuffer")
    assert not hasattr(librosa.util, "async_capture")


def test_streambridge_validation():
    """Test parameter validation on StreamBridge."""
    with pytest.raises(TypeError):
        StreamBridge()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        StreamBridge(22050)  # type: ignore[misc]
    with pytest.raises(librosa.ParameterError):
        StreamBridge(sr=-1)
    with pytest.raises(librosa.ParameterError):
        StreamBridge(sr=np.nan)
    with pytest.raises(librosa.ParameterError):
        StreamBridge(sr=22050, channels=0)
    with pytest.raises(librosa.ParameterError):
        StreamBridge(sr=22050, maxsize=-1)
    with pytest.raises(librosa.ParameterError):
        StreamBridge(sr=22050, overflow_policy="invalid")

    bridge = StreamBridge(sr=22050, channels=1)
    with pytest.raises(librosa.ParameterError):
        bridge.push("not_an_array_or_bytes")  # type: ignore[arg-type]
    # 2-channel input pushed into 1-channel bridge should raise ParameterError
    with pytest.raises(librosa.ParameterError):
        bridge.push(np.ones((64, 2), dtype=np.float32))
    # 3D input should raise ParameterError
    with pytest.raises(librosa.ParameterError):
        bridge.push(np.ones((1, 64, 1), dtype=np.float32))

    bridge_2ch = StreamBridge(sr=22050, channels=2)
    with pytest.raises(librosa.ParameterError):
        bridge_2ch.push(np.ones(63, dtype=np.float32))

    bridge.close()
    # Direct push on closed bridge raises RuntimeError, while callback returns paComplete (1)
    with pytest.raises(RuntimeError):
        bridge.push(np.ones(64, dtype=np.float32))
    assert bridge(np.ones(64, dtype=np.float32)) == (None, 1)


def test_streambridge_sync_and_timeout():
    """Test StreamBridge synchronous context manager, iteration, and timeout."""
    with StreamBridge(sr=22050, channels=1) as bridge:
        ret = bridge(np.ones((512, 1), dtype=np.float32) * 2)
        assert ret == (None, 0)

    chunks = list(bridge)
    assert len(chunks) == 1
    assert chunks[0].shape == (512,)
    assert np.allclose(chunks[0], 2.0)

    # Timeout when empty and open
    bridge_timeout = StreamBridge(sr=22050, timeout=0.01)
    with pytest.raises(queue.Empty):
        next(bridge_timeout)


def test_streambridge_pcm_and_float_bytes():
    """Test StreamBridge decoding of raw PCM and float byte buffers."""
    # 16-bit integer PCM
    with StreamBridge(sr=22050, channels=1, dtype=np.int16) as bridge:
        int_data = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        bridge(int_data.tobytes())
        bridge(int_data)

    chunks = list(bridge)
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.dtype == np.float32
        assert np.isclose(chunk[1], 0.5, atol=1e-3)
        assert np.isclose(chunk[2], -0.5, atol=1e-3)

    # Float32 raw bytes
    with StreamBridge(sr=22050, channels=1, dtype=np.float32) as bridge_f32:
        f_data = np.array([0.25, -0.75], dtype=np.float32)
        bridge_f32(f_data.tobytes())

    chunks_f32 = list(bridge_f32)
    assert len(chunks_f32) == 1
    assert np.allclose(chunks_f32[0], f_data)


def test_streambridge_channel_transposition():
    """Test StreamBridge channel transposition for 2D and 1D interleaved buffers."""
    # 2D (samples, channels) with channels_first=False
    with StreamBridge(sr=22050, channels=2, channels_first=False) as bridge:
        data = np.zeros((512, 2), dtype=np.float32)
        data[:, 0] = 1.0
        data[:, 1] = 2.0
        bridge(data)
        # 1D interleaved stereo buffer
        interleaved = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        bridge(interleaved)

    chunks = list(bridge)
    assert len(chunks) == 2
    assert chunks[0].shape == (2, 512)
    assert np.allclose(chunks[0][0], 1.0)
    assert np.allclose(chunks[0][1], 2.0)
    assert chunks[1].shape == (2, 2)
    assert np.allclose(chunks[1][0], 1.0)
    assert np.allclose(chunks[1][1], 2.0)

    # channels_first=True
    with StreamBridge(sr=22050, channels=2, channels_first=True) as bridge_cf:
        flat_cf = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        bridge_cf(flat_cf)

    chunks_cf = list(bridge_cf)
    assert chunks_cf[0].shape == (2, 2)
    assert np.allclose(chunks_cf[0][0], 1.0)
    assert np.allclose(chunks_cf[0][1], 2.0)


def test_streambridge_overflow_policies():
    """Test all StreamBridge buffer overflow policies."""
    # drop_oldest
    with StreamBridge(sr=22050, maxsize=2, overflow_policy="drop_oldest") as b_old:
        b_old.push(np.array([1.0], dtype=np.float32))
        b_old.push(np.array([2.0], dtype=np.float32))
        b_old.push(np.array([3.0], dtype=np.float32))
    chunks = list(b_old)
    assert [c[0] for c in chunks] == [2.0, 3.0]

    # drop_newest
    with StreamBridge(sr=22050, maxsize=2, overflow_policy="drop_newest") as b_new:
        b_new.push(np.array([1.0], dtype=np.float32))
        b_new.push(np.array([2.0], dtype=np.float32))
        b_new.push(np.array([3.0], dtype=np.float32))
    chunks = list(b_new)
    assert [c[0] for c in chunks] == [1.0, 2.0]

    # warn
    with StreamBridge(sr=22050, maxsize=1, overflow_policy="warn") as b_warn:
        b_warn.push(np.array([1.0], dtype=np.float32))
        with pytest.warns(RuntimeWarning, match="StreamBridge queue is full"):
            b_warn.push(np.array([2.0], dtype=np.float32))
    chunks = list(b_warn)
    assert [c[0] for c in chunks] == [2.0]

    # error
    with StreamBridge(sr=22050, maxsize=1, overflow_policy="error") as b_err:
        b_err.push(np.array([1.0], dtype=np.float32))
        with pytest.raises(BufferError):
            b_err.push(np.array([2.0], dtype=np.float32))


def test_stream_warm_up_and_multichannel():
    """Test warm_up and multichannel handling through librosa.stream(bridge)."""
    # warm_up=True yields immediately on the first hop-sized chunk before closing
    bridge = StreamBridge(sr=22050, channels=1)
    bridge(np.ones(512, dtype=np.float32))

    stream_gen = bridge.stream(
        block_length=1,
        frame_length=1024,
        hop_length=512,
        warm_up=True,
    )
    first_block = next(stream_gen)
    assert first_block.shape == (1024,)
    assert np.allclose(first_block[:512], 0.0)
    assert np.allclose(first_block[512:], 1.0)

    bridge.close()
    remaining_blocks = list(stream_gen)
    assert len(remaining_blocks) == 1
    assert remaining_blocks[0].shape == (512,)
    assert np.allclose(remaining_blocks[0], 1.0)

    # 1-channel with mono=False on 2D callback input (samples, 1)
    with StreamBridge(sr=22050, channels=1) as bridge_1ch:
        bridge_1ch(np.ones((1024, 1), dtype=np.float32))

    blocks_1ch = list(
        librosa.stream(
            bridge_1ch,
            block_length=1,
            frame_length=1024,
            hop_length=512,
            sr=None,
            mono=False,
        )
    )
    assert blocks_1ch[0].shape == (1024,)

    # Stereo downmix (mono=True) and stereo preservation (mono=False)
    stereo_chunk = np.zeros((512, 2), dtype=np.float32)
    stereo_chunk[:, 0] = 1.0
    stereo_chunk[:, 1] = 3.0

    with StreamBridge(sr=22050, channels=2) as b_mono:
        b_mono(stereo_chunk)
        b_mono(stereo_chunk)

    blocks_mono = list(
        librosa.stream(
            b_mono,
            block_length=1,
            frame_length=1024,
            hop_length=512,
            sr=None,
            mono=True,
        )
    )
    assert blocks_mono[0].shape == (1024,)
    assert np.allclose(blocks_mono[0], 2.0)

    with StreamBridge(sr=22050, channels=2) as b_stereo:
        b_stereo(stereo_chunk)
        b_stereo(stereo_chunk)

    blocks_stereo = list(
        librosa.stream(
            b_stereo,
            block_length=1,
            frame_length=1024,
            hop_length=512,
            sr=None,
            mono=False,
        )
    )
    assert blocks_stereo[0].shape == (2, 1024)
    assert np.allclose(blocks_stereo[0][0], 1.0)
    assert np.allclose(blocks_stereo[0][1], 3.0)


@pytest.mark.parametrize("sr", [None, 11025])
@pytest.mark.parametrize("offset", [0.0, 0.5])
@pytest.mark.parametrize("duration", [None, 1.0])
@pytest.mark.parametrize("fill_value", [None, 0.0])
def test_stream_file_and_bridge_parity(sr, offset, duration, fill_value):
    """Verify bit-for-bit parity between file streaming and StreamBridge streaming."""
    filename = librosa.ex("trumpet")
    with sf.SoundFile(filename) as sfo:
        native_sr = sfo.samplerate
        channels = sfo.channels
        bridge = StreamBridge(sr=native_sr, channels=channels)
        for chunk in sfo.blocks(blocksize=400, dtype=np.float32, always_2d=True):
            bridge(chunk)
        bridge.close()

    kwargs = dict(
        block_length=2,
        frame_length=1024,
        hop_length=512,
        sr=sr,
        offset=offset,
        duration=duration,
        fill_value=fill_value,
    )
    file_blocks = list(librosa.stream(filename, **kwargs))
    bridge_blocks = list(librosa.stream(bridge, **kwargs))

    assert len(file_blocks) == len(bridge_blocks)
    for fb, bb in zip(file_blocks, bridge_blocks, strict=True):
        assert fb.shape == bb.shape
        assert np.allclose(fb, bb, atol=1e-5)


def test_threaded_producer_and_buffer_resize():
    """Test background producer thread with StreamBridge.stream and buffer resizing."""
    bridge = StreamBridge(sr=22050, channels=1, maxsize=4, overflow_policy="error")

    def producer():
        bridge(np.array([], dtype=np.float32))
        for i in range(6):
            bridge(np.ones(512, dtype=np.float32) * i)
        bridge.close()

    t = threading.Thread(target=producer)
    t.start()
    blocks = list(
        bridge.stream(
            block_length=1,
            frame_length=1024,
            hop_length=512,
            sr=22050,
            duration=0.1,
            dtype=np.float64,
        )
    )
    t.join()

    assert len(blocks) > 0
    assert blocks[0].dtype == np.float64
    assert blocks[0].shape == (1024,)

    # Large chunk causing internal buffer resize
    with StreamBridge(sr=22050, channels=1) as pre_bridge:
        pre_bridge(np.ones(8192, dtype=np.float32))
    pre_blocks = list(
        pre_bridge.stream(
            block_length=1,
            frame_length=1024,
            hop_length=512,
        )
    )
    assert len(pre_blocks) == 16


def test_stream_invalid_sources_and_bounds():
    """Test rejection of unsupported sources and negative stream bounds."""
    with pytest.raises(librosa.ParameterError):
        next(
            librosa.stream(
                [np.ones(512, dtype=np.float32)],  # type: ignore[arg-type]
                block_length=1,
                frame_length=1024,
                hop_length=512,
                sr=None,
            )
        )

    bridge = StreamBridge(sr=22050)
    with pytest.raises(librosa.ParameterError):
        next(
            librosa.stream(
                bridge,
                block_length=1,
                frame_length=1024,
                hop_length=512,
                sr=None,
                offset=-1.0,
            )
        )
    with pytest.raises(librosa.ParameterError):
        next(
            librosa.stream(
                bridge,
                block_length=1,
                frame_length=1024,
                hop_length=512,
                sr=None,
                duration=-1.0,
            )
        )
    with pytest.raises(librosa.ParameterError):
        next(
            librosa.stream(
                librosa.ex("trumpet"),
                block_length=1,
                frame_length=1024,
                hop_length=512,
                sr=None,
                duration=-1.0,
            )
        )
