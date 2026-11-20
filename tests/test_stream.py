#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for StreamBridge and librosa.stream."""
from __future__ import annotations

import os
import queue
import threading
import time
import warnings

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
    assert streaming.stream is librosa.stream
    assert librosa.core.stream is librosa.stream
    assert hasattr(StreamBridge, "stream")


def test_streambridge_validation():
    """Test parameter validation on StreamBridge."""
    with pytest.raises(TypeError):
        StreamBridge()  # type: ignore[call-arg]
    with pytest.raises(librosa.ParameterValueError):
        StreamBridge(orig_sr=-1)
    with pytest.raises(librosa.ParameterValueError):
        StreamBridge(orig_sr=np.nan)
    with pytest.raises(librosa.ParameterValueError):
        StreamBridge(orig_sr=22050, channels=0)
    with pytest.raises(librosa.ParameterValueError):
        StreamBridge(orig_sr=22050, channels=1.5)  # type: ignore[arg-type]
    with pytest.raises(librosa.ParameterValueError):
        StreamBridge(orig_sr=22050, maxsize=-1)
    with pytest.raises(librosa.ParameterTypeError):
        StreamBridge(orig_sr=22050, maxsize=1.5)  # type: ignore[arg-type]
    with pytest.raises(librosa.ParameterTypeError):
        StreamBridge(orig_sr=22050, maxsize="invalid")  # type: ignore[arg-type]
    with pytest.raises(librosa.ParameterValueError):
        StreamBridge(orig_sr=22050, overflow_policy="invalid")

    # Verify numpy integer scalars are accepted
    assert StreamBridge(orig_sr=22050, channels=np.int32(1), maxsize=np.int64(0)).maxsize == 0
    assert StreamBridge(orig_sr=22050, maxsize=np.int64(5)).maxsize == 5

    bridge = StreamBridge(orig_sr=22050, channels=1)
    with pytest.raises(librosa.ParameterTypeError):
        bridge.push("not_an_array_or_bytes")  # type: ignore[arg-type]
    # 2-channel input pushed into 1-channel bridge should raise ParameterError (float and int16)
    with pytest.raises(librosa.ParameterShapeError):
        bridge.push(np.ones((64, 2), dtype=np.float32))
    with pytest.raises(librosa.ParameterShapeError):
        bridge.push(np.ones((64, 2), dtype=np.int16))
    # 3D input should raise ParameterError
    with pytest.raises(librosa.ParameterShapeError):
        bridge.push(np.ones((1, 64, 1), dtype=np.float32))

    bridge_2ch = StreamBridge(orig_sr=22050, channels=2)
    with pytest.raises(librosa.ParameterShapeError):
        bridge_2ch.push(np.ones(63, dtype=np.float32))

    bridge.close()
    # Direct push on closed bridge raises RuntimeError, while callback returns paComplete (1)
    with pytest.raises(librosa.StreamClosedError):
        bridge.push(np.ones(64, dtype=np.float32))
    assert bridge(np.ones(64, dtype=np.float32)) == (None, 1)


def test_streambridge_sync_and_timeout():
    """Test StreamBridge synchronous context manager, iteration, and timeout."""
    with StreamBridge(orig_sr=22050, channels=1) as bridge:
        ret = bridge(np.ones((512, 1), dtype=np.float32) * 2)
        assert ret == (None, 0)

    chunks = list(bridge)
    assert len(chunks) == 1
    assert chunks[0].shape == (512,)
    assert np.allclose(chunks[0], 2.0)

    # Timeout when empty and open
    bridge_timeout = StreamBridge(orig_sr=22050, timeout=0.01)
    with pytest.raises(librosa.StreamTimeoutError):
        next(bridge_timeout)


def test_streambridge_pcm_and_float_bytes():
    """Test StreamBridge decoding of raw PCM and float byte buffers."""
    # 16-bit integer PCM (bytes, 1D array, and non-contiguous 2D view)
    with StreamBridge(orig_sr=22050, channels=1, dtype=np.int16) as bridge:
        int_data = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        bridge(int_data.tobytes())
        bridge(int_data)
        int_2d = np.column_stack([int_data, int_data])
        bridge(int_2d[:, :1])  # non-contiguous (4, 1) view

    chunks = list(bridge)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.dtype == np.float32
        assert chunk.shape == (4,)
        assert np.isclose(chunk[1], 0.5, atol=1e-3)
        assert np.isclose(chunk[2], -0.5, atol=1e-3)

    # Unsigned 8-bit integer PCM (128 is silence -> 0.0, 0 -> -1.0, 192 -> 0.5)
    with StreamBridge(orig_sr=22050, channels=1, dtype=np.uint8) as bridge_u8:
        u8_data = np.array([128, 0, 192], dtype=np.uint8)
        bridge_u8(u8_data.tobytes())
        bridge_u8(u8_data)

    chunks_u8 = list(bridge_u8)
    assert len(chunks_u8) == 2
    for chunk in chunks_u8:
        assert chunk.dtype == np.float32
        assert np.isclose(chunk[0], 0.0)
        assert np.isclose(chunk[1], -1.0)
        assert np.isclose(chunk[2], 0.5)

    # Float32 raw bytes
    with StreamBridge(orig_sr=22050, channels=1, dtype=np.float32) as bridge_f32:
        f_data = np.array([0.25, -0.75], dtype=np.float32)
        bridge_f32(f_data.tobytes())

    chunks_f32 = list(bridge_f32)
    assert len(chunks_f32) == 1
    assert np.allclose(chunks_f32[0], f_data)


def test_streambridge_channel_transposition():
    """Test StreamBridge channel transposition for 2D and 1D interleaved buffers."""
    # 2D (samples, channels) with channels_first=False
    with StreamBridge(orig_sr=22050, channels=2, channels_first=False) as bridge:
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
    with StreamBridge(orig_sr=22050, channels=2, channels_first=True) as bridge_cf:
        flat_cf = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        bridge_cf(flat_cf)

    chunks_cf = list(bridge_cf)
    assert chunks_cf[0].shape == (2, 2)
    assert np.allclose(chunks_cf[0][0], 1.0)
    assert np.allclose(chunks_cf[0][1], 2.0)


def test_streambridge_overflow_policies():
    """Test all StreamBridge buffer overflow policies."""
    # drop_oldest
    with StreamBridge(orig_sr=22050, maxsize=2, overflow_policy="drop_oldest") as b_old:
        b_old.push(np.array([1.0], dtype=np.float32))
        b_old.push(np.array([2.0], dtype=np.float32))
        b_old.push(np.array([3.0], dtype=np.float32))
    chunks = list(b_old)
    assert [c[0] for c in chunks] == [2.0, 3.0]

    # drop_newest
    with StreamBridge(orig_sr=22050, maxsize=2, overflow_policy="drop_newest") as b_new:
        b_new.push(np.array([1.0], dtype=np.float32))
        b_new.push(np.array([2.0], dtype=np.float32))
        b_new.push(np.array([3.0], dtype=np.float32))
    chunks = list(b_new)
    assert [c[0] for c in chunks] == [1.0, 2.0]

    # warn
    b_warn = StreamBridge(orig_sr=22050, maxsize=1, overflow_policy="warn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b_warn.push(np.array([1.0], dtype=np.float32))
        b_warn.push(np.array([2.0], dtype=np.float32))
    b_warn.close()

    with pytest.warns(RuntimeWarning, match=r"StreamBridge queue is full; dropped 1 chunk\(s\)"):
        chunks = list(b_warn)
    assert [c[0] for c in chunks] == [2.0]

    b_warn2 = StreamBridge(orig_sr=22050, maxsize=1, overflow_policy="warn")
    b_warn2.push(np.array([1.0], dtype=np.float32))
    b_warn2.push(np.array([2.0], dtype=np.float32))
    b_warn2.push(np.array([3.0], dtype=np.float32))
    b_warn2.close()
    with pytest.warns(RuntimeWarning, match=r"StreamBridge queue is full; dropped 2 chunk\(s\)"):
        chunks2 = list(b_warn2)
    assert [c[0] for c in chunks2] == [3.0]

    # error
    with StreamBridge(orig_sr=22050, maxsize=1, overflow_policy="error") as b_err:
        b_err.push(np.array([1.0], dtype=np.float32))
        with pytest.raises(librosa.StreamBufferError):
            b_err.push(np.array([2.0], dtype=np.float32))


def test_stream_warm_up_empty():
    """Test warm_up on empty bridge or zero duration yields nothing."""
    with StreamBridge(orig_sr=22050, channels=1) as bridge:
        pass
    blocks = list(
        bridge.stream(block_length=1, frame_length=1024, hop_length=512, warm_up=True)
    )
    assert blocks == []

    with StreamBridge(orig_sr=22050, channels=1) as bridge:
        bridge.push(np.ones(512, dtype=np.float32))
    blocks_zero = list(
        bridge.stream(
            block_length=1,
            frame_length=1024,
            hop_length=512,
            warm_up=True,
            duration=0.0,
        )
    )
    assert blocks_zero == []


def test_stream_warm_up_and_multichannel():
    """Test warm_up and multichannel handling through librosa.stream(bridge)."""
    # warm_up=True yields immediately on the first hop-sized chunk before closing
    bridge = StreamBridge(orig_sr=22050, channels=1)
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
    with StreamBridge(orig_sr=22050, channels=1) as bridge_1ch:
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

    with StreamBridge(orig_sr=22050, channels=2) as b_mono:
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

    with StreamBridge(orig_sr=22050, channels=2) as b_stereo:
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
        bridge = StreamBridge(orig_sr=native_sr, channels=channels)
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
    bridge = StreamBridge(orig_sr=22050, channels=1)

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
    with StreamBridge(orig_sr=22050, channels=1) as pre_bridge:
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
    with pytest.raises(librosa.ParameterTypeError):
        next(
            librosa.stream(
                [np.ones(512, dtype=np.float32)],
                block_length=1,
                frame_length=1024,
                hop_length=512,
                sr=None,
            )
        )

    bridge = StreamBridge(orig_sr=22050)
    with pytest.raises(librosa.ParameterValueError):
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
    with pytest.raises(librosa.ParameterValueError):
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
    with pytest.raises(librosa.ParameterValueError):
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

    from librosa.core.streaming import _StreamBuffer

    with pytest.raises(librosa.ParameterValueError, match="orig_sr"):
        _StreamBuffer(
            frame_length=1024,
            hop_length=512,
            block_length=1,
            orig_sr=-1,
        )

    with pytest.raises(librosa.ParameterValueError, match="channels"):
        _StreamBuffer(
            frame_length=1024,
            hop_length=512,
            block_length=1,
            orig_sr=22050,
            channels=0,
        )

    # Zero duration on an open bridge returns immediately without blocking or consuming chunks
    open_bridge = StreamBridge(orig_sr=22050, timeout=0.05)
    open_bridge.push(np.ones(1024, dtype=np.float32))
    assert (
        list(
            open_bridge.stream(
                block_length=1,
                frame_length=1024,
                hop_length=512,
                duration=0.0,
            )
        )
        == []
    )
    # The queued chunk was not consumed by the zero-duration stream
    open_bridge.close()
    assert len(list(open_bridge)) == 1

    # Non-integer sample advance ratio raises ParameterError
    bridge_48k = StreamBridge(orig_sr=48000)
    with pytest.raises(librosa.ParameterValueError, match=r"fractional sample count"):
        next(
            librosa.stream(
                bridge_48k,
                block_length=1,
                frame_length=1024,
                hop_length=512,
                sr=22050,
            )
        )


def test_streambridge_concurrent_close():
    """Test that concurrent close from a separate thread cleanly terminates consumer iteration."""
    bridge = StreamBridge(orig_sr=22050, channels=1)
    consumed_blocks = []
    started_event = threading.Event()

    def consumer():
        started_event.set()
        for block in bridge.stream(block_length=1, frame_length=1024, hop_length=512):
            consumed_blocks.append(block)

    t = threading.Thread(target=consumer)
    t.start()

    assert started_event.wait(timeout=2.0)
    time.sleep(0.05)
    bridge.close()
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert consumed_blocks == []


def test_streambridge_concurrent_close_with_data():
    """Test concurrent close drains pending chunks before stopping."""
    bridge = StreamBridge(orig_sr=22050, channels=1)
    bridge.push(np.ones(1024, dtype=np.float32))
    consumed_blocks = []
    started_event = threading.Event()

    def consumer():
        started_event.set()
        for block in bridge.stream(block_length=1, frame_length=1024, hop_length=512):
            consumed_blocks.append(block)

    t = threading.Thread(target=consumer)
    t.start()

    assert started_event.wait(timeout=2.0)
    time.sleep(0.05)
    bridge.close()
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert len(consumed_blocks) >= 1


def test_stream_default_sr():
    """Test default sr=22050 behavior in stream and StreamBridge matching load()."""
    filename = os.path.join("tests", "test_audio.ogg")

    # 1. librosa.stream with default sr should match librosa.load with default sr (22050)
    y_load_default, sr_default = librosa.load(filename)
    assert sr_default == 22050

    stream_default = librosa.stream(
        filename, block_length=16, frame_length=1024, hop_length=512
    )
    blocks = list(stream_default)
    # Reassemble blocks using hop advance
    y_stream = []
    for i, b in enumerate(blocks):
        if i == len(blocks) - 1:
            y_stream.append(b)
        else:
            y_stream.append(b[: 16 * 512])
    y_stream_concat = np.concatenate(y_stream)
    assert np.allclose(y_stream_concat[: len(y_load_default)], y_load_default[: len(y_stream_concat)], atol=1e-3)

    # 2. librosa.stream with sr=None should match librosa.load with sr=None (native)
    y_load_native, sr_native = librosa.load(filename, sr=None)
    assert sr_native != 22050  # test_audio.ogg is 44100 Hz

    stream_native = librosa.stream(
        filename, block_length=16, frame_length=1024, hop_length=512, sr=None
    )
    blocks_native = list(stream_native)
    y_stream_nat = []
    for i, b in enumerate(blocks_native):
        if i == len(blocks_native) - 1:
            y_stream_nat.append(b)
        else:
            y_stream_nat.append(b[: 16 * 512])
    y_stream_nat_concat = np.concatenate(y_stream_nat)
    assert np.allclose(y_stream_nat_concat[: len(y_load_native)], y_load_native[: len(y_stream_nat_concat)], atol=1e-3)

    # 3. StreamBridge orig_sr attribute and sr alias property
    bridge = StreamBridge(orig_sr=44100, channels=1)
    assert bridge.orig_sr == 44100
    assert bridge.sr == 44100

    # StreamBridge.stream default sr=22050 resamples from 44100
    # 44100 to 22050 is a 2:1 ratio. Hop length 512 at 22050 corresponds to 1024 samples at 44100.
    test_signal = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
    bridge.push(test_signal)
    bridge.close()

    # Streaming with default sr=22050
    resampled_blocks = list(bridge.stream(block_length=1, frame_length=1024, hop_length=512))
    assert len(resampled_blocks) > 0

    # Streaming from another 44100 bridge with sr=None preserves native 44100
    bridge_native = StreamBridge(orig_sr=44100, channels=1)
    bridge_native.push(test_signal[:2048])
    bridge_native.close()
    native_blocks = list(bridge_native.stream(block_length=1, frame_length=1024, hop_length=512, sr=None))
    assert len(native_blocks) > 0


def test_streambridge_enqueue_race_closed(monkeypatch):
    bridge = StreamBridge(orig_sr=22050)
    data = np.ones(512, dtype=np.float32)

    def close_during_normalize(x):
        bridge.close()
        return np.ones((512, 1), dtype=np.float32)

    monkeypatch.setattr(bridge, "_normalize", close_during_normalize)
    assert bridge._enqueue(data, raise_on_closed=False) is False


