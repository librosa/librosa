#!/usr/bin/env python
# CREATED:2014-01-18 14:09:05 by Brian McFee <brm2132@columbia.edu>
# unit tests for util routines

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except:
    pass

import platform
import numpy as np
import scipy.sparse
import pytest
import warnings
import librosa

from test_core import srand

np.set_printoptions(precision=3)


def test_example_audio_file():

    assert os.path.exists(librosa.util.example_audio_file())


@pytest.mark.parametrize("frame_length", [4, 8])
@pytest.mark.parametrize("hop_length", [2, 4])
@pytest.mark.parametrize("y", [np.random.randn(32)])
@pytest.mark.parametrize("axis", [0, -1])
def test_frame1d(frame_length, hop_length, axis, y):

    y_frame = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length, axis=axis)

    if axis == -1:
        y_frame = y_frame.T

    for i in range(y_frame.shape[0]):
        assert np.allclose(y_frame[i], y[i * hop_length : (i * hop_length + frame_length)])


@pytest.mark.parametrize("frame_length", [4, 8])
@pytest.mark.parametrize("hop_length", [2, 4])
@pytest.mark.parametrize(
    "y, axis", [(np.asfortranarray(np.random.randn(16, 32)), -1), (np.ascontiguousarray(np.random.randn(16, 32)), 0)]
)
def test_frame2d(frame_length, hop_length, axis, y):

    y_frame = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length, axis=axis)

    if axis == -1:
        y_frame = y_frame.T
        y = y.T

    for i in range(y_frame.shape[0]):
        assert np.allclose(y_frame[i], y[i * hop_length : (i * hop_length + frame_length)])


def test_frame_0stride():
    x = np.arange(10)
    xpad = x[np.newaxis]

    xpad2 = np.atleast_2d(x)

    xf = librosa.util.frame(x, 3, 1)
    xfpad = librosa.util.frame(xpad, 3, 1)
    xfpad2 = librosa.util.frame(xpad2, 3, 1)

    assert np.allclose(xf, xfpad)
    assert np.allclose(xf, xfpad2)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_frame_badtype():
    librosa.util.frame([1, 2, 3, 4], frame_length=2, hop_length=1)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("x", [np.arange(16)])
def test_frame_too_short(x, axis):
    librosa.util.frame(x, frame_length=17, hop_length=1, axis=axis)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_frame_bad_hop():
    librosa.util.frame(np.arange(16), frame_length=4, hop_length=0)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("axis", [1, 2])
def test_frame_bad_axis(axis):
    librosa.util.frame(np.zeros((3, 3, 3)), frame_length=2, hop_length=1, axis=axis)


@pytest.mark.parametrize("x_bad, axis", 
                        [(np.zeros((4, 10), order="C"), -1), 
                         (np.zeros((4, 10), order="F"), 0)])
def test_frame_bad_contiguity(x_bad, axis):
    # Populate fixture with random data
    x_bad += np.random.randn(*x_bad.shape)

    # And make a contiguous copy of it
    if axis == 0:
        x_good = np.ascontiguousarray(x_bad)
    else:
        x_good = np.asfortranarray(x_bad)

    # Verify that the aligned data is good
    assert np.allclose(x_bad, x_good)

    # The test here checks two things:
    #   1) that output is identical if we provide properly contiguous input
    #   2) that a warning is issued if the input is not properly contiguous
    x_good_f = librosa.util.frame(x_good, frame_length=2, hop_length=1, axis=axis)
    with pytest.warns(UserWarning):
        x_bad_f = librosa.util.frame(x_bad, frame_length=2, hop_length=1, axis=axis)
        assert np.allclose(x_good_f, x_bad_f)


@pytest.mark.parametrize("y", [np.ones((16,)), np.ones((16, 16))])
@pytest.mark.parametrize("m", [0, 10])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("mode", ["constant", "edge", "reflect"])
def test_pad_center(y, m, axis, mode):

    n = m + y.shape[axis]
    y_out = librosa.util.pad_center(y, n, axis=axis, mode=mode)

    n_len = y.shape[axis]
    n_pad = int((n - n_len) / 2)

    eq_slice = [slice(None)] * y.ndim
    eq_slice[axis] = slice(n_pad, n_pad + n_len)

    assert np.allclose(y, y_out[tuple(eq_slice)])


@pytest.mark.parametrize("y", [np.ones((16,)), np.ones((16, 16))])
@pytest.mark.parametrize("n", [0, 10])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("mode", ["constant", "edge", "reflect"])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_pad_center_fail(y, n, axis, mode):
    librosa.util.pad_center(y, n, axis=axis, mode=mode)


@pytest.mark.parametrize("y", [np.ones((16,)), np.ones((16, 16))])
@pytest.mark.parametrize("m", [-5, 0, 5])
@pytest.mark.parametrize("axis", [0, -1])
def test_fix_length(y, m, axis):
    n = m + y.shape[axis]

    y_out = librosa.util.fix_length(y, n, axis=axis)

    eq_slice = [slice(None)] * y.ndim
    eq_slice[axis] = slice(y.shape[axis])

    if n > y.shape[axis]:
        assert np.allclose(y, y_out[tuple(eq_slice)])
    else:
        assert np.allclose(y[tuple(eq_slice)], y)


@pytest.mark.parametrize("frames", [np.arange(20, 100, step=15)])
@pytest.mark.parametrize("x_min", [0, 20])
@pytest.mark.parametrize("x_max", [20, 70, 120])
@pytest.mark.parametrize("pad", [False, True])
def test_fix_frames(frames, x_min, x_max, pad):

    f_fix = librosa.util.fix_frames(frames, x_min=x_min, x_max=x_max, pad=pad)

    if x_min is not None:
        if pad:
            assert f_fix[0] == x_min
        assert np.all(f_fix >= x_min)

    if x_max is not None:
        if pad:
            assert f_fix[-1] == x_max
        assert np.all(f_fix <= x_max)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("frames", [np.arange(-20, 100)])
@pytest.mark.parametrize("x_min", [None, 0, 20])
@pytest.mark.parametrize("x_max", [None, 0, 20])
@pytest.mark.parametrize("pad", [False, True])
def test_fix_frames_fail_negative(frames, x_min, x_max, pad):
    librosa.util.fix_frames(frames, x_min, x_max, pad)


@pytest.mark.parametrize("norm", [np.inf, -np.inf, 0, 0.5, 1.0, 2.0, None])
@pytest.mark.parametrize("ndims,axis", [(1, 0), (1, -1), (2, 0), (2, 1), (2, -1), (3, 0), (3, 1), (3, 2), (3, -1)])
def test_normalize(ndims, norm, axis):
    srand()
    X = np.random.randn(*([4] * ndims))
    X_norm = librosa.util.normalize(X, norm=norm, axis=axis)

    # Shape and dtype checks
    assert X_norm.dtype == X.dtype
    assert X_norm.shape == X.shape

    if norm is None:
        assert np.allclose(X, X_norm)
        return

    X_norm = np.abs(X_norm)

    if norm == np.inf:
        values = np.max(X_norm, axis=axis)
    elif norm == -np.inf:
        values = np.min(X_norm, axis=axis)
    elif norm == 0:
        # XXX: normalization here isn't quite right
        values = np.ones(1)

    else:
        values = np.sum(X_norm ** norm, axis=axis) ** (1.0 / norm)

    assert np.allclose(values, np.ones_like(values))


@pytest.mark.parametrize("norm", ["inf", -0.5, -2])
@pytest.mark.parametrize("X", [np.ones((3, 3))])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_normalize_badnorm(X, norm):
    librosa.util.normalize(X, norm=norm)


@pytest.mark.parametrize("badval", [np.nan, np.inf, -np.inf])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_normalize_bad_input(badval):
    X = np.ones((3, 3))
    X[0] = badval
    librosa.util.normalize(X, norm=np.inf, axis=0)


@pytest.mark.parametrize("fill", [7, "foo"])
@pytest.mark.parametrize("X", [np.ones((2, 2))])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_normalize_badfill(X, fill):
    librosa.util.normalize(X, fill=fill)


@pytest.mark.parametrize("x", [np.asarray([[0, 1, 2, 3]])])
@pytest.mark.parametrize(
    "threshold, result",
    [(None, [[0, 1, 1, 1]]), (1, [[0, 1, 1, 1]]), (2, [[0, 1, 1, 1]]), (3, [[0, 1, 2, 1]]), (4, [[0, 1, 2, 3]])],
)
def test_normalize_threshold(x, threshold, result):
    assert np.allclose(librosa.util.normalize(x, threshold=threshold), result)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("x", [np.asarray([[0, 1, 2, 3]])])
@pytest.mark.parametrize("threshold", [0, -1])
def test_normalize_threshold_fail(x, threshold):
    librosa.util.normalize(x, threshold=threshold)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_normalize_fill_l0():
    X = np.ones((2, 2))
    librosa.util.normalize(X, fill=True, norm=0)


@pytest.mark.parametrize("norm", [1, 2, np.inf])
@pytest.mark.parametrize("X", [np.zeros((3, 3))])
def test_normalize_fill_allaxes(X, norm):
    Xn = librosa.util.normalize(X, fill=True, axis=None, norm=norm)
    if norm is np.inf:
        assert np.allclose(Xn, 1)
    else:
        assert np.allclose(np.sum(Xn ** norm) ** (1.0 / norm), 1)


@pytest.mark.parametrize("norm", [1, 2, np.inf])
@pytest.mark.parametrize("X", [np.zeros((3, 3))])
def test_normalize_nofill(X, norm):
    Xn = librosa.util.normalize(X, fill=False, norm=norm)
    assert np.allclose(Xn, 0)


@pytest.mark.parametrize("X", [np.asarray([[0.0, 1], [0, 1]])])
@pytest.mark.parametrize("norm,value", [(1, 0.5), (2, np.sqrt(2) / 2), (np.inf, 1)])
@pytest.mark.parametrize("threshold", [0.5, 2])
def test_normalize_fill(X, threshold, norm, value):
    Xn = librosa.util.normalize(X, fill=True, norm=norm, threshold=threshold)
    assert np.allclose(Xn, value)


@pytest.mark.parametrize("ndim", [1, 3])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("value", [None, np.min, np.mean, np.max])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_axis_sort_badndim(ndim, axis, index, value):
    data = np.zeros([2] * ndim)
    librosa.util.axis_sort(data, axis=axis, index=index, value=value)


@pytest.mark.parametrize("ndim", [2])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("value", [None, np.min, np.mean, np.max])
def test_axis_sort(ndim, axis, index, value):
    srand()
    data = np.random.randn(*([10] * ndim))
    if index:
        Xsorted, idx = librosa.util.axis_sort(data, axis=axis, index=index, value=value)

        cmp_slice = [slice(None)] * ndim
        cmp_slice[axis] = idx

        assert np.allclose(data[tuple(cmp_slice)], Xsorted)

    else:
        Xsorted = librosa.util.axis_sort(data, axis=axis, index=index, value=value)

    compare_axis = np.mod(1 - axis, 2)

    if value is None:
        value = np.argmax

    sort_values = value(Xsorted, axis=compare_axis)

    assert np.allclose(sort_values, np.sort(sort_values))


@pytest.mark.parametrize(
    "int_from, int_to",
    [
        (np.asarray([[0, 2], [0, 4], [3, 6]]), np.empty((0, 2), dtype=int)),
        (np.empty((0, 2), dtype=int), np.asarray([[0, 2], [0, 4], [3, 6]])),
    ],
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_match_intervals_empty(int_from, int_to):
    librosa.util.match_intervals(int_from, int_to)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_match_intervals_strict_fail():
    int_from = np.asarray([[0, 3], [2, 4], [5, 7]])
    int_to = np.asarray([[0, 2], [0, 4]])
    librosa.util.match_intervals(int_from, int_to, strict=True)


@pytest.mark.parametrize("int_from", [np.asarray([[0, 3], [2, 4], [5, 7]])])
@pytest.mark.parametrize("int_to", [np.asarray([[0, 2], [0, 4], [3, 6]])])
@pytest.mark.parametrize("matches", [np.asarray([1, 1, 2])])
def test_match_intervals_strict(int_from, int_to, matches):

    test_matches = librosa.util.match_intervals(int_from, int_to, strict=True)
    assert np.array_equal(matches, test_matches)


@pytest.mark.parametrize("int_from", [np.asarray([[0, 3], [2, 4], [5, 7]])])
@pytest.mark.parametrize(
    "int_to,matches",
    [
        (np.asarray([[0, 2], [0, 4], [3, 6]]), np.asarray([1, 1, 2])),
        (np.asarray([[0, 2], [0, 4]]), np.asarray([1, 1, 1])),
    ],
)
def test_match_intervals_nonstrict(int_from, int_to, matches):
    test_matches = librosa.util.match_intervals(int_from, int_to, strict=False)
    assert np.array_equal(matches, test_matches)


@pytest.mark.parametrize("n", [1, 5, 20, 100])
@pytest.mark.parametrize("m", [1, 5, 20, 100])
def test_match_events(n, m):

    srand()
    ev1 = np.abs(np.random.randn(n))
    ev2 = np.abs(np.random.randn(m))

    match = librosa.util.match_events(ev1, ev2)

    for i in range(len(match)):
        values = np.asarray([np.abs(ev1[i] - e2) for e2 in ev2])
        assert not np.any(values < values[match[i]])


@pytest.mark.parametrize("ev1,ev2", [(np.array([]), np.arange(5)), (np.arange(5), np.array([]))])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_match_events_failempty(ev1, ev2):
    librosa.util.match_events(ev1, ev2)


@pytest.mark.parametrize("events_from", [np.asarray([5, 15, 25])])
@pytest.mark.parametrize("events_to", [np.asarray([0, 10, 20, 30])])
@pytest.mark.parametrize("left,right,target", [(False, True, [10, 20, 30]), (True, False, [0, 10, 20])])
def test_match_events_onesided(events_from, events_to, left, right, target):

    events_from = np.asarray(events_from)
    events_to = np.asarray(events_to)
    match = librosa.util.match_events(events_from, events_to, left=left, right=right)

    assert np.allclose(target, events_to[match])


def test_match_events_twosided():
    events_from = np.asarray([5, 15, 25])
    events_to = np.asarray([5, 15, 25, 30])
    match = librosa.util.match_events(events_from, events_to, left=False, right=False)
    assert np.allclose(match, [0, 1, 2])


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "events_from,events_to,left,right",
    [
        ([40, 15, 25], [0, 10, 20, 30], False, True),  # right-sided fail
        ([-1, 15, 25], [0, 10, 20, 30], True, False),  # left-sided fail
        ([-1, 15, 25], [0, 10, 20, 30], False, False),  # two-sided fail
    ],
)
def test_match_events_onesided_fail(events_from, events_to, left, right):
    events_from = np.asarray(events_from)
    events_to = np.asarray(events_to)
    librosa.util.match_events(events_from, events_to, left=left, right=right)


@pytest.mark.parametrize("ndim, axis", [(n, m) for n in range(1, 5) for m in range(n)])
def test_localmax(ndim, axis):

    srand()

    data = np.random.randn(*([7] * ndim))
    lm = librosa.util.localmax(data, axis=axis)

    for hits in np.argwhere(lm):
        for offset in [-1, 1]:
            compare_idx = hits.copy()
            compare_idx[axis] += offset

            if compare_idx[axis] < 0:
                continue

            if compare_idx[axis] >= data.shape[axis]:
                continue

            if offset < 0:
                assert data[tuple(hits)] > data[tuple(compare_idx)]
            else:
                assert data[tuple(hits)] >= data[tuple(compare_idx)]


@pytest.mark.parametrize("x", [np.random.randn(_) ** 2 for _ in [1, 5, 10, 100]])
@pytest.mark.parametrize("pre_max", [0, 1, 10])
@pytest.mark.parametrize("post_max", [1, 10])
@pytest.mark.parametrize("pre_avg", [0, 1, 10])
@pytest.mark.parametrize("post_avg", [1, 10])
@pytest.mark.parametrize("wait", [0, 1, 10])
@pytest.mark.parametrize("delta", [0.05, 100.0])
def test_peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    peaks = librosa.util.peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait)

    for i in peaks:
        # Test 1: is it a peak in this window?
        s = i - pre_max
        if s < 0:
            s = 0
        t = i + post_max

        diff = x[i] - np.max(x[s:t])
        assert diff > 0 or np.isclose(diff, 0, rtol=1e-3, atol=1e-4)

        # Test 2: is it a big enough peak to count?
        s = i - pre_avg
        if s < 0:
            s = 0
        t = i + post_avg

        diff = x[i] - (delta + np.mean(x[s:t]))
        assert diff > 0 or np.isclose(diff, 0, rtol=1e-3, atol=1e-4)

    # Test 3: peak separation
    assert not np.any(np.diff(peaks) <= wait)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("x", [np.random.randn(_) ** 2 for _ in [1, 5, 10, 100]])
@pytest.mark.parametrize(
    "pre_max,post_max,pre_avg,post_avg,delta,wait",
    [
        (-1, 1, 1, 1, 0.05, 1),  # negative pre-max
        (1, -1, 1, 1, 0.05, 1),  # negative post-max
        (1, 0, 1, 1, 0.05, 1),  # 0 post-max
        (1, 1, -1, 1, 0.05, 1),  # negative pre-avg
        (1, 1, 1, -1, 0.05, 1),  # negative post-avg
        (1, 1, 1, 0, 0.05, 1),  # zero post-avg
        (1, 1, 1, 1, -0.05, 1),  # negative delta
        (1, 1, 1, 1, 0.05, -1),  # negative wait
    ],
)
def test_peak_pick_fail(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    librosa.util.peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_peak_pick_shape_fail():
    # Can't pick peaks on 2d inputs
    librosa.util.peak_pick(np.eye(2), 1, 1, 1, 1, 0.5, 1)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("ndim", [3, 4])
def test_sparsify_rows_ndimfail(ndim):
    X = np.zeros([2] * ndim)
    librosa.util.sparsify_rows(X)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("quantile", [1.0, -1, 2.0])
@pytest.mark.parametrize("X", [np.ones((3, 3))])
def test_sparsify_rows_badquantile(X, quantile):
    librosa.util.sparsify_rows(X, quantile=quantile)


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("d", [1, 5, 10, 100])
@pytest.mark.parametrize("q", [0.0, 0.01, 0.25, 0.5, 0.99])
def test_sparsify_rows(ndim, d, q):
    srand()

    X = np.random.randn(*([d] * ndim)) ** 4

    X = np.asarray(X)

    xs = librosa.util.sparsify_rows(X, quantile=q)

    if ndim == 1:
        X = X.reshape((1, -1))

    assert np.allclose(xs.shape, X.shape)

    # And make sure that xs matches X on nonzeros
    xsd = np.asarray(xs.todense())

    for i in range(xs.shape[0]):
        assert np.allclose(xsd[i, xs[i].indices], X[i, xs[i].indices])

    # Compute row-wise magnitude marginals
    v_in = np.sum(np.abs(X), axis=-1)
    v_out = np.sum(np.abs(xsd), axis=-1)

    # Ensure that v_out retains 1-q fraction of v_in
    assert np.all(v_out >= (1.0 - q) * v_in)


@pytest.mark.parametrize(
    "searchdir", [os.path.join(os.path.curdir, "tests"), os.path.join(os.path.curdir, "tests", "data")]
)
@pytest.mark.parametrize("ext", [None, "wav", "WAV", ["wav"], ["WAV"]])
@pytest.mark.parametrize("recurse", [True])
@pytest.mark.parametrize("case_sensitive", list({False} | {platform.system() != "Windows"}))
@pytest.mark.parametrize("limit", [None, 1, 2])
@pytest.mark.parametrize("offset", [0, 1, -1])
@pytest.mark.parametrize(
    "output",
    [
        [
            os.path.join(os.path.abspath(os.path.curdir), "tests", "data", s)
            for s in ["test1_22050.mp3", "test1_22050.wav", "test1_44100.wav", "test2_8000.wav"]
        ]
    ],
)
def test_find_files(searchdir, ext, recurse, case_sensitive, limit, offset, output):
    files = librosa.util.find_files(
        searchdir, ext=ext, recurse=recurse, case_sensitive=case_sensitive, limit=limit, offset=offset
    )

    targets = output
    if ext is not None:
        # If we're only seeking wavs, bump off the mp3 file
        targets = targets[1:]

    s1 = slice(offset, None)
    s2 = slice(limit)

    if case_sensitive and ext not in (None, "wav", ["wav"]):
        assert len(files) == 0
    else:
        assert set(files) == set(targets[s1][s2])


def test_find_files_nonrecurse():
    files = librosa.util.find_files(os.path.join(os.path.curdir, "tests"), recurse=False)
    assert len(files) == 0


# fail if ext is not none, we're case-sensitive, and looking for WAV
@pytest.mark.parametrize("ext", ["WAV", ["WAV"]])
def test_find_files_case_sensitive(ext):
    files = librosa.util.find_files(os.path.join(os.path.curdir, "tests"), ext=ext, case_sensitive=True)
    # On windows, this test won't work
    if platform.system() != "Windows":
        assert len(files) == 0


@pytest.mark.parametrize("x_in", np.linspace(-2, 2, num=6))
@pytest.mark.parametrize("cast", [None, np.floor, np.ceil])
def test_valid_int(x_in, cast):

    z = librosa.util.valid_int(x_in, cast)

    assert isinstance(z, int)
    if cast is None:
        assert z == int(np.floor(x_in))
    else:
        assert z == int(cast(x_in))


@pytest.mark.parametrize("x", np.linspace(-2, 2, num=3))
@pytest.mark.parametrize("cast", [7])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_valid_int_fail(x, cast):
    # Test with a non-callable cast operator
    librosa.util.valid_int(x, cast)


@pytest.mark.parametrize(
    "ivals", [np.asarray([[0, 1], [1, 2]]), np.asarray([[0, 0], [1, 1]]), np.asarray([[0, 2], [1, 2]])]
)
def test_valid_intervals(ivals):
    librosa.util.valid_intervals(ivals)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "ivals", [np.asarray([]), np.arange(2), np.ones((2, 2, 2)), np.ones((2, 3))]  # ndim=0  # ndim=1  # ndim=3
)  # ndim=2, shape[1] != 2
def test_valid_intervals_badshape(ivals):
    #   fail if ndim != 2 or shape[1] != 2
    librosa.util.valid_intervals(ivals)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("intval", [np.asarray([[0, 1], [2, 1]])])
def test_valid_intervals_fail(intval):
    # Test for issue #712: intervals must have non-negative duration
    librosa.util.valid_intervals(intval)


def test_warning_deprecated():
    @librosa.util.decorators.deprecated("old_version", "new_version")
    def __dummy():
        return True

    with warnings.catch_warnings(record=True) as out:
        x = __dummy()

        # Make sure we still get the right value
        assert x is True

        # And that the warning triggered
        assert len(out) > 0

        # And that the category is correct
        assert out[0].category is DeprecationWarning

        # And that it says the right thing (roughly)
        assert "deprecated" in str(out[0].message).lower()


def test_warning_moved():
    @librosa.util.decorators.moved("from", "old_version", "new_version")
    def __dummy():
        return True

    with warnings.catch_warnings(record=True) as out:
        x = __dummy()

        # Make sure we still get the right value
        assert x is True

        # And that the warning triggered
        assert len(out) > 0

        # And that the category is correct
        assert out[0].category is DeprecationWarning

        # And that it says the right thing (roughly)
        assert "moved" in str(out[0].message).lower()


def test_warning_rename_kw_pass():

    warnings.resetwarnings()
    warnings.simplefilter("always")

    ov = librosa.util.Deprecated()
    nv = 23

    with warnings.catch_warnings(record=True) as out:
        v = librosa.util.rename_kw("old", ov, "new", nv, "0", "1")

        assert v == nv

        # Make sure no warning triggered
        assert len(out) == 0


def test_warning_rename_kw_fail():

    warnings.resetwarnings()
    warnings.simplefilter("always")

    ov = 27
    nv = 23

    with warnings.catch_warnings(record=True) as out:
        v = librosa.util.rename_kw("old", ov, "new", nv, "0", "1")

        assert v == ov

        # Make sure the warning triggered
        assert len(out) > 0

        # And that the category is correct
        assert out[0].category is DeprecationWarning

        # And that it says the right thing (roughly)
        assert "renamed" in str(out[0].message).lower()


@pytest.mark.parametrize("idx", [np.arange(10, 90, 10), np.arange(10, 90, 15)])
@pytest.mark.parametrize("idx_min", [None, 5, 15])
@pytest.mark.parametrize("idx_max", [None, 85, 100])
@pytest.mark.parametrize("step", [None, 2])
@pytest.mark.parametrize("pad", [False, True])
def test_index_to_slice(idx, idx_min, idx_max, step, pad):

    slices = librosa.util.index_to_slice(idx, idx_min=idx_min, idx_max=idx_max, step=step, pad=pad)

    if pad:
        if idx_min is not None:
            assert slices[0].start == idx_min
            if idx.min() != idx_min:
                slices = slices[1:]
        if idx_max is not None:
            assert slices[-1].stop == idx_max
            if idx.max() != idx_max:
                slices = slices[:-1]

    if idx_min is not None:
        idx = idx[idx >= idx_min]

    if idx_max is not None:
        idx = idx[idx <= idx_max]

    idx = np.unique(idx)
    assert len(slices) == len(idx) - 1

    for sl, start, stop in zip(slices, idx, idx[1:]):
        assert sl.start == start
        assert sl.stop == stop
        assert sl.step == step


@pytest.mark.parametrize("aggregate", [None, np.mean, np.sum])
@pytest.mark.parametrize("ndim,axis", [(1, 0), (1, -1), (2, 0), (2, 1), (2, -1), (3, 0), (3, 2), (3, -1)])
def test_sync(aggregate, ndim, axis):
    data = np.ones([6] * ndim, dtype=np.float)

    # Make some slices that don't fill the entire dimension
    slices = [slice(1, 3), slice(3, 4)]
    dsync = librosa.util.sync(data, slices, aggregate=aggregate, axis=axis)

    # Check the axis shapes
    assert dsync.shape[axis] == len(slices)

    s_test = list(dsync.shape)
    del s_test[axis]
    s_orig = list(data.shape)
    del s_orig[axis]
    assert s_test == s_orig

    # The first slice will sum to 2 and have mean 1
    idx = [slice(None)] * ndim
    idx[axis] = 0
    if aggregate is np.sum:
        assert np.allclose(dsync[idx], 2)
    else:
        assert np.allclose(dsync[idx], 1)

    # The second slice will sum to 1 and have mean 1
    idx[axis] = 1
    assert np.allclose(dsync[idx], 1)


@pytest.mark.parametrize("aggregate", [np.mean, np.max])
def test_sync_slices(aggregate):
    x = np.arange(8, dtype=float)
    slices = [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)]
    xsync = librosa.util.sync(x, slices, aggregate=aggregate)
    if aggregate is np.mean:
        assert np.allclose(xsync, [0.5, 2.5, 4.5, 6.5])
    elif aggregate is np.max:
        assert np.allclose(xsync, [1, 3, 5, 7])
    else:
        assert False


@pytest.mark.parametrize("aggregate", [np.mean, np.max])
@pytest.mark.parametrize("atype", [list, np.asarray])
def test_sync_frames(aggregate, atype):
    x = np.arange(8, dtype=float)
    frames = atype([0, 2, 4, 6, 8])
    xsync = librosa.util.sync(x, frames, aggregate=aggregate)
    if aggregate is np.mean:
        assert np.allclose(xsync, [0.5, 2.5, 4.5, 6.5])
    elif aggregate is np.max:
        assert np.allclose(xsync, [1, 3, 5, 7])
    else:
        assert False


@pytest.mark.parametrize("atype", [list, np.asarray])
@pytest.mark.parametrize("pad", [False, True])
def test_sync_frames_pad(atype, pad):
    x = np.arange(8, dtype=float)
    frames = atype([2, 4, 6])
    xsync = librosa.util.sync(x, frames, pad=pad)
    if pad:
        assert np.allclose(xsync, [0.5, 2.5, 4.5, 6.5])
    else:
        assert np.allclose(xsync, [2.5, 4.5])


@pytest.mark.parametrize("data", [np.mod(np.arange(135), 5)])
@pytest.mark.parametrize("idx", [["foo", "bar"], [None], [slice(None), None]])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_sync_fail(data, idx):
    librosa.util.sync(data, idx)


@pytest.mark.parametrize("power", [1, 2, 50, 100, np.inf])
@pytest.mark.parametrize("split_zeros", [False, True])
def test_softmask(power, split_zeros):

    srand()

    X = np.abs(np.random.randn(10, 10))
    X_ref = np.abs(np.random.randn(10, 10))

    # Zero out some rows
    X[3, :] = 0
    X_ref[3, :] = 0

    M = librosa.util.softmask(X, X_ref, power=power, split_zeros=split_zeros)

    assert np.all(0 <= M) and np.all(M <= 1)

    if split_zeros and np.isfinite(power):
        assert np.allclose(M[3, :], 0.5)
    else:
        assert not np.any(M[3, :]), M[3]


def test_softmask_int():
    X = 2 * np.ones((3, 3), dtype=np.int32)
    X_ref = np.vander(np.arange(3))

    M1 = librosa.util.softmask(X, X_ref, power=1)
    M2 = librosa.util.softmask(X_ref, X, power=1)

    assert np.allclose(M1 + M2, 1)


@pytest.mark.parametrize(
    "x,x_ref,power,split_zeros",
    [
        (-np.ones(3), np.ones(3), 1, False),
        (np.ones(3), -np.ones(3), 1, False),
        (np.ones(3), np.ones(4), 1, False),
        (np.ones(3), np.ones(3), 0, False),
        (np.ones(3), np.ones(3), -1, False),
    ],
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_softmask_fail(x, x_ref, power, split_zeros):
    librosa.util.softmask(x, x_ref, power=power, split_zeros=split_zeros)


@pytest.mark.parametrize(
    "x,value",
    [
        (1, np.finfo(np.float32).tiny),
        (np.ones(3, dtype=int), np.finfo(np.float32).tiny),
        (np.ones(3, dtype=np.float32), np.finfo(np.float32).tiny),
        (1.0, np.finfo(np.float64).tiny),
        (np.ones(3, dtype=np.float64), np.finfo(np.float64).tiny),
        (1j, np.finfo(np.complex128).tiny),
        (np.ones(3, dtype=np.complex64), np.finfo(np.complex64).tiny),
        (np.ones(3, dtype=np.complex128), np.finfo(np.complex128).tiny),
    ],
)
def test_tiny(x, value):
    assert value == librosa.util.tiny(x)


def test_util_fill_off_diagonal_8_8():
    # Case 1: Square matrix (N=M)
    mut_x = np.ones((8, 8))
    librosa.util.fill_off_diagonal(mut_x, 0.25)

    gt_x = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ]
    )

    assert np.array_equal(mut_x, gt_x)
    assert np.array_equal(mut_x, gt_x.T)


def test_util_fill_off_diagonal_8_12():
    # Case 2a: N!=M
    mut_x = np.ones((8, 12))
    librosa.util.fill_off_diagonal(mut_x, 0.25)

    gt_x = np.array(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
    )

    assert np.array_equal(mut_x, gt_x)

    # Case 2b: (N!=M).T
    mut_x = np.ones((8, 12)).T
    librosa.util.fill_off_diagonal(mut_x, 0.25)

    assert np.array_equal(mut_x, gt_x.T)


@pytest.mark.parametrize("dtype_A", [np.float32, np.float64])
@pytest.mark.parametrize("dtype_B", [np.float32, np.float64])
def test_nnls_vector(dtype_A, dtype_B):
    srand()

    # Make a random basis
    A = np.random.randn(5, 7).astype(dtype_A)

    # Make a random latent vector
    x = np.random.randn(A.shape[1]) ** 2

    B = A.dot(x).astype(dtype_B)

    x_rec = librosa.util.nnls(A, B)

    assert np.all(x_rec >= 0)
    assert np.sqrt(np.mean((B - A.dot(x_rec)) ** 2)) <= 1e-6


@pytest.mark.parametrize("dtype_A", [np.float32, np.float64])
@pytest.mark.parametrize("dtype_B", [np.float32, np.float64])
@pytest.mark.parametrize("x_size", [3, 30])
def test_nnls_matrix(dtype_A, dtype_B, x_size):
    srand()

    # Make a random basis
    A = np.random.randn(5, 7).astype(dtype_A)

    # Make a random latent matrix
    #   when x_size is 3, B is 7x3 (smaller than A)
    x = np.random.randn(A.shape[1], x_size) ** 2

    B = A.dot(x).astype(dtype_B)

    x_rec = librosa.util.nnls(A, B)

    assert np.all(x_rec >= 0)
    assert np.sqrt(np.mean((B - A.dot(x_rec)) ** 2)) <= 1e-5


@pytest.mark.parametrize("dtype_A", [np.float32, np.float64])
@pytest.mark.parametrize("dtype_B", [np.float32, np.float64])
@pytest.mark.parametrize("x_size", [16, 64, 256])
def test_nnls_multiblock(dtype_A, dtype_B, x_size):
    srand()

    # Make a random basis
    A = np.random.randn(7, 1025).astype(dtype_A)

    # Make a random latent matrix
    #   when x_size is 3, B is 7x3 (smaller than A)
    x = np.random.randn(A.shape[1], x_size) ** 2

    B = A.dot(x).astype(dtype_B)

    x_rec = librosa.util.nnls(A, B)

    assert np.all(x_rec >= 0)
    assert np.sqrt(np.mean((B - A.dot(x_rec)) ** 2)) <= 1e-4


@pytest.fixture
def psig():

    # [[0, 1, 2, 3, 4]]
    # axis=1 or -1 ==> [-1.5, 1, 1, 1, -1.5]
    # axis=0 ==> [0, 0, 0, 0, 0]
    return np.arange(0, 5, dtype=float)[np.newaxis]


@pytest.mark.parametrize("edge_order", [1, 2])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_cyclic_gradient(psig, edge_order, axis):
    grad = librosa.util.cyclic_gradient(psig, edge_order=edge_order, axis=axis)

    assert grad.shape == psig.shape
    assert grad.dtype == psig.dtype

    # Check the values
    if axis == 0:
        assert np.allclose(grad, 0)
    else:
        assert np.allclose(grad, [-1.5, 1, 1, 1, -1.5])


def test_shear_dense():

    E = np.eye(3)

    E_shear = librosa.util.shear(E, factor=1, axis=0)
    assert np.allclose(E_shear, np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

    E_shear = librosa.util.shear(E, factor=1, axis=1)
    assert np.allclose(E_shear, np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

    E_shear = librosa.util.shear(E, factor=-1, axis=1)
    assert np.allclose(E_shear, np.asarray([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))

    E_shear = librosa.util.shear(E, factor=-1, axis=0)
    assert np.allclose(E_shear, np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0]]))


@pytest.mark.parametrize("fmt", ["csc", "csr", "lil", "dok"])
def test_shear_sparse(fmt):
    E = scipy.sparse.identity(3, format=fmt)

    E_shear = librosa.util.shear(E, factor=1, axis=0)
    assert E_shear.format == fmt
    assert np.allclose(E_shear.toarray(), np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

    E_shear = librosa.util.shear(E, factor=1, axis=1)
    assert E_shear.format == fmt
    assert np.allclose(E_shear.toarray(), np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

    E_shear = librosa.util.shear(E, factor=-1, axis=1)
    assert E_shear.format == fmt
    assert np.allclose(E_shear.toarray(), np.asarray([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))

    E_shear = librosa.util.shear(E, factor=-1, axis=0)
    assert E_shear.format == fmt
    assert np.allclose(E_shear.toarray(), np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0]]))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_shear_badfactor():
    librosa.util.shear(np.eye(3), factor=None)


def test_stack_contig():
    x1 = np.ones(3)
    x2 = -np.ones(3)

    xs = librosa.util.stack([x1, x2], axis=0)

    assert xs.flags["F_CONTIGUOUS"]
    assert np.allclose(xs, [[1, 1, 1], [-1, -1, -1]])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stack_fail_shape():
    x1 = np.ones(3)

    x2 = np.ones(2)
    librosa.util.stack([x1, x2])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stack_fail_empty():
    librosa.util.stack([])


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("x", [np.random.randn(5, 10, 20)])
def test_stack_consistent(x, axis):
    xs = librosa.util.stack([x, x], axis=axis)
    xsnp = np.stack([x, x], axis=axis)

    assert np.allclose(xs, xsnp)
    if axis != 0:
        assert xs.flags["C_CONTIGUOUS"]
