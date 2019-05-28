#!/usr/bin/env python
# CREATED:2014-01-18 14:09:05 by Brian McFee <brm2132@columbia.edu>
# unit tests for util routines

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import platform
import numpy as np
import scipy.sparse
import pytest
import six
import warnings
import librosa

from test_core import srand

np.set_printoptions(precision=3)


def test_example_audio_file():

    assert os.path.exists(librosa.util.example_audio_file())


def test_frame():

    # Generate a random time series
    def __test(P):
        srand()
        frame, hop = P

        y = np.random.randn(8000)
        y_frame = librosa.util.frame(y, frame_length=frame, hop_length=hop)

        for i in range(y_frame.shape[1]):
            assert np.allclose(y_frame[:, i], y[i * hop:(i * hop + frame)])

    for frame in [256, 1024, 2048]:
        for hop_length in [64, 256, 512]:
            yield (__test, [frame, hop_length])


def test_frame_fail():

    __test = pytest.mark.xfail(librosa.util.frame, raises=librosa.ParameterError)

    # First fail, not an ndarray
    yield __test, list(range(10)), 5, 1

    # Second fail: wrong ndims
    yield __test, np.zeros((10, 10)), 5, 1

    # Third fail: too short
    yield __test, np.zeros(10), 20, 1

    # Fourth fail: bad hop length
    yield __test, np.zeros(10), 20, -1

    # Fifth fail: discontiguous input
    yield __test, np.zeros(20)[::2], 10, 1


def test_pad_center():

    def __test(y, n, axis, mode):

        y_out = librosa.util.pad_center(y, n, axis=axis, mode=mode)

        n_len = y.shape[axis]
        n_pad = int((n - n_len) / 2)

        eq_slice = [slice(None)] * y.ndim
        eq_slice[axis] = slice(n_pad, n_pad + n_len)

        assert np.allclose(y, y_out[tuple(eq_slice)])

    @pytest.mark.xfail(raises=librosa.ParameterError)
    def __test_fail(y, n, axis, mode):
        librosa.util.pad_center(y, n, axis=axis, mode=mode)

    for shape in [(16,), (16, 16)]:
        y = np.ones(shape)

        for axis in [0, -1]:
            for mode in ['constant', 'edge', 'reflect']:
                for n in [0, 10]:
                    yield __test, y, n + y.shape[axis], axis, mode

                for n in [0, 10]:
                    yield __test_fail, y, n, axis, mode


def test_fix_length():

    def __test(y, n, axis):

        y_out = librosa.util.fix_length(y, n, axis=axis)

        eq_slice = [slice(None)] * y.ndim
        eq_slice[axis] = slice(y.shape[axis])

        if n > y.shape[axis]:
            assert np.allclose(y, y_out[tuple(eq_slice)])
        else:
            assert np.allclose(y[tuple(eq_slice)], y)

    for shape in [(16,), (16, 16)]:
        y = np.ones(shape)

        for axis in [0, -1]:
            for n in [-5, 0, 5]:
                yield __test, y, n + y.shape[axis], axis


def test_fix_frames():
    srand()

    @pytest.mark.xfail(raises=librosa.ParameterError)
    def __test_fail(frames, x_min, x_max, pad):
        librosa.util.fix_frames(frames, x_min, x_max, pad)

    def __test_pass(frames, x_min, x_max, pad):

        f_fix = librosa.util.fix_frames(frames,
                                        x_min=x_min,
                                        x_max=x_max,
                                        pad=pad)

        if x_min is not None:
            if pad:
                assert f_fix[0] == x_min
            assert np.all(f_fix >= x_min)

        if x_max is not None:
            if pad:
                assert f_fix[-1] == x_max
            assert np.all(f_fix <= x_max)

    for low in [-20, 0, 20]:
        for high in [low + 20, low + 50, low + 100]:

            frames = np.random.randint(low, high=high, size=15)

            for x_min in [None, 0, 20]:
                for x_max in [None, 20, 100]:
                    for pad in [False, True]:
                        if np.any(frames < 0):
                            yield __test_fail, frames, x_min, x_max, pad
                        else:
                            yield __test_pass, frames, x_min, x_max, pad


def test_normalize():
    srand()

    def __test_pass(X, norm, axis):
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
            values = np.sum(X_norm**norm, axis=axis)**(1./norm)

        assert np.allclose(values, np.ones_like(values))

    @pytest.mark.xfail(raises=librosa.ParameterError)
    def _test_fail(X, norm, axis):
        librosa.util.normalize(X, norm=norm, axis=axis)

    __test_fail = pytest.mark.xfail(_test_fail, raises=librosa.ParameterError)

    for ndims in [1, 2, 3]:
        X = np.random.randn(* ([16] * ndims))

        for axis in range(X.ndim):
            for norm in [np.inf, -np.inf, 0, 0.5, 1.0, 2.0, None]:
                yield __test_pass, X, norm, axis

            for norm in ['inf', -0.5, -2]:
                yield __test_fail, X, norm, axis

        # And test for non-finite failure
        Xnan = X.copy()

        Xnan[0] = np.nan
        yield __test_fail, Xnan, np.inf, 0

        Xinf = X.copy()
        Xinf[0] = np.inf
        yield __test_fail, Xinf, np.inf, 0


def test_normalize_threshold():

    x = np.asarray([[0, 1, 2, 3]])

    def __test(threshold, result):
        assert np.allclose(librosa.util.normalize(x, threshold=threshold),
                           result)

    yield __test, None, [[0, 1, 1, 1]]
    yield __test, 1, [[0, 1, 1, 1]]
    yield __test, 2, [[0, 1, 1, 1]]
    yield __test, 3, [[0, 1, 2, 1]]
    yield __test, 4, [[0, 1, 2, 3]]
    tf = pytest.mark.xfail(__test, raises=librosa.ParameterError)
    yield tf, 0, [[0, 1, 1, 1]]
    yield tf, -1, [[0, 1, 1, 1]]


def test_normalize_fill():

    def __test(fill, norm, threshold, axis, x, result):
        xn = librosa.util.normalize(x,
                                    axis=axis,
                                    fill=fill,
                                    threshold=threshold,
                                    norm=norm)
        assert np.allclose(xn, result), (xn, np.asarray(result))

    x = np.asarray([[0, 1, 2, 3]], dtype=np.float32)

    axis = 0
    norm = np.inf
    threshold = 2
    # Test with inf norm
    yield __test, None, norm, threshold, axis, x, [[0, 1, 1, 1]]
    yield __test, False, norm, threshold, axis, x, [[0, 0, 1, 1]]
    yield __test, True, norm, threshold, axis, x, [[1, 1, 1, 1]]

    # Test with l0 norm
    norm = 0
    yield __test, None, norm, threshold, axis, x, [[0, 1, 2, 3]]
    yield __test, False, norm, threshold, axis, x, [[0, 0, 0, 0]]
    tf = pytest.mark.xfail(__test, raises=librosa.ParameterError)
    yield tf, True, norm, threshold, axis, x, [[0, 0, 0, 0]]

    # Test with l1 norm
    norm = 1
    yield __test, None, norm, threshold, axis, x, [[0, 1, 1, 1]]
    yield __test, False, norm, threshold, axis, x, [[0, 0, 1, 1]]
    yield __test, True, norm, threshold, axis, x, [[1, 1, 1, 1]]

    # And with l2 norm
    norm = 2
    x = np.repeat(x, 2, axis=0)
    s = np.sqrt(2)/2

    # First two columns are left as is, second two map to sqrt(2)/2
    yield __test, None, norm, threshold, axis, x, [[0, 1, s, s], [0, 1, s, s]]

    # First two columns are zeroed, second two map to sqrt(2)/2
    yield __test, False, norm, threshold, axis, x, [[0, 0, s, s], [0, 0, s, s]]

    # All columns map to sqrt(2)/2
    yield __test, True, norm, threshold, axis, x, [[s, s, s, s], [s, s, s, s]]

    # And test the bad-fill case
    yield tf, 3, norm, threshold, axis, x, x

    # And an all-axes test
    axis = None
    threshold = None
    norm = 2
    yield __test, None, norm, threshold, axis, np.asarray([[3, 0], [0, 4]]), np.asarray([[0, 0], [0, 0]])
    yield __test, None, norm, threshold, axis, np.asarray([[3., 0], [0, 4]]), np.asarray([[0.6, 0], [0, 0.8]])


@pytest.mark.parametrize('ndim', [pytest.mark.xfail(1, raises=librosa.ParameterError),
                                  2,
                                  pytest.mark.xfail(3, raises=librosa.ParameterError)])
@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('index', [False, True])
@pytest.mark.parametrize('value', [None, np.min, np.mean, np.max])
def test_axis_sort(ndim, axis, index, value):
    srand()
    data = np.random.randn(*([10] * ndim))
    if index:
        Xsorted, idx = librosa.util.axis_sort(data,
                                              axis=axis,
                                              index=index,
                                              value=value)

        cmp_slice = [slice(None)] * ndim
        cmp_slice[axis] = idx

        assert np.allclose(data[tuple(cmp_slice)], Xsorted)

    else:
        Xsorted = librosa.util.axis_sort(data,
                                         axis=axis,
                                         index=index,
                                         value=value)

    compare_axis = np.mod(1 - axis, 2)

    if value is None:
        value = np.argmax

    sort_values = value(Xsorted, axis=compare_axis)

    assert np.allclose(sort_values, np.sort(sort_values))


def test_match_intervals_empty():


    @pytest.mark.xfail(raises=librosa.ParameterError)
    def __test(int_from, int_to):
        librosa.util.match_intervals(int_from, int_to)

    ints = np.asarray([[0, 2],
                       [0, 4],
                       [3, 6]])

    # true matches for the above
    yield __test, ints, ints[:0]
    yield __test, ints[:0], ints


@pytest.mark.parametrize('int_from', [np.asarray([[0, 3], [2, 4], [5, 7]])])
@pytest.mark.parametrize('int_to', [np.asarray([[0, 2], [0, 4], [3, 6]]),
                                    pytest.mark.xfail(np.asarray([[0, 2], [0, 4]]),
                                                      raises=librosa.ParameterError)])
@pytest.mark.parametrize('matches', [np.asarray([1, 1, 2])])
def test_match_intervals_strict(int_from, int_to, matches):

    test_matches = librosa.util.match_intervals(int_from, int_to, strict=True)
    assert np.array_equal(matches, test_matches)


def test_match_intervals_nonstrict():


    def __test(int_from, int_to, matches):
        test_matches = librosa.util.match_intervals(int_from, int_to, strict=False)
        assert np.array_equal(matches, test_matches)


    int_from = np.asarray([[0, 3],
                           [2, 4],
                           [5, 7]])

    int_to = np.asarray([[0, 2],
                         [0, 4],
                         [3, 6]])

    # true matches for the above
    matches = np.asarray([1, 1, 2])
    yield __test, int_from, int_to, matches

    # Without the [3, 6] interval, the source [5, 7] should match [2, 4]
    yield __test, int_from, int_to[:-1], np.asarray([1, 1, 1])


def test_match_events():

    def __make_events(n):
        srand()
        return np.abs(np.random.randn(n))

    def __is_best(y, ev1, ev2):
        for i in range(len(y)):
            values = np.asarray([np.abs(ev1[i] - e2) for e2 in ev2])
            if np.any(values < values[y[i]]):
                return False

        return True

    def __test(n, m):
        ev1 = __make_events(n)
        ev2 = __make_events(m)

        y_pred = librosa.util.match_events(ev1, ev2)

        assert __is_best(y_pred, ev1, ev2)

    @pytest.mark.xfail(raises=librosa.ParameterError)
    def __test_fail(n, m):
        ev1 = __make_events(n)
        ev2 = __make_events(m)
        librosa.util.match_events(ev1, ev2)

    for n in [0, 1, 5, 20, 100]:
        for m in [0, 1, 5, 20, 100]:
            if n == 0 or m == 0:
                yield __test_fail, n, m
            else:
                yield __test, n, m


def test_match_events_onesided():

    events_from = np.asarray([5, 15, 25])
    events_to = np.asarray([0, 10, 20, 30])

    def __test(events_from, events_to, left, right, target):
        match = librosa.util.match_events(events_from, events_to,
                                          left=left, right=right)

        assert np.allclose(target, events_to[match])

    yield __test, events_from, events_to, False, True, [10, 20, 30]
    yield __test, events_from, events_to, True, False, [0, 10, 20]

    # Make a right-sided fail
    tf = pytest.mark.xfail(__test, raises=librosa.ParameterError)

    ef = events_from.copy()
    ef[0] = 40
    yield tf, ef, events_to, False, True, [10, 20, 30]

    # Make a left-sided fail
    ef = events_from.copy()
    ef[0] = -1
    yield tf, ef, events_to, True, False, [10, 20, 30]

    # Make a two-sided fail
    ef = events_from.copy()
    ef[0] = -1
    yield tf, ef, events_to, False, False, [10, 20, 30]

    # Make a two-sided success
    et = events_to.copy()
    et[:-1] = events_from
    yield __test, events_from, et, False, False, events_from


def test_localmax():

    def __test(ndim, axis):
        srand()

        data = np.random.randn(*([20] * ndim))

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

    for ndim in range(1, 5):
        for axis in range(ndim):
            yield __test, ndim, axis


def test_peak_pick():

    def __test(n, pre_max, post_max, pre_avg, post_avg, delta, wait):
        srand()

        # Generate a test signal
        x = np.random.randn(n)**2

        peaks = librosa.util.peak_pick(x,
                                       pre_max, post_max,
                                       pre_avg, post_avg,
                                       delta, wait)

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
    def __test_shape_fail():
        x = np.eye(10)
        librosa.util.peak_pick(x, 1, 1, 1, 1, 0.5, 1)

    yield __test_shape_fail

    win_range = [-1, 0, 1, 10]
    __test_fail = pytest.mark.xfail(__test, raises=librosa.ParameterError)

    for n in [1, 5, 10, 100]:
        for pre_max in win_range:
            for post_max in win_range:
                for pre_avg in win_range:
                    for post_avg in win_range:
                        for wait in win_range:
                            for delta in [-1, 0.05, 100.0]:
                                tf = __test
                                if pre_max < 0:
                                    tf = __test_fail
                                if pre_avg < 0:
                                    tf = __test_fail
                                if delta < 0:
                                    tf = __test_fail
                                if wait < 0:
                                    tf = __test_fail
                                if post_max <= 0:
                                    tf = __test_fail
                                if post_avg <= 0:
                                    tf = __test_fail
                                yield (tf, n, pre_max, post_max,
                                       pre_avg, post_avg, delta, wait)


@pytest.mark.parametrize('ndim', [1, 2, pytest.mark.xfail(3, raises=librosa.ParameterError)])
@pytest.mark.parametrize('d', [1, 5, 10, 100])
@pytest.mark.parametrize('q', [0., 0.01, 0.25, 0.5, 0.99,
                               pytest.mark.xfail(1.0, raises=librosa.ParameterError),
                               pytest.mark.xfail(-1, raises=librosa.ParameterError),
                               pytest.mark.xfail(2.0, raises=librosa.ParameterError)])
def test_sparsify_rows(ndim, d, q):
    srand()

    X = np.random.randn(*([d] * ndim))**4

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


def test_files():

    # Expected output
    output = [os.path.join(os.path.abspath(os.path.curdir), 'tests', 'data', s)
              for s in ['test1_22050.mp3',
                        'test1_22050.wav',
                        'test1_44100.wav',
                        'test2_8000.wav']]

    def __test(searchdir, ext, recurse, case_sensitive, limit, offset):
        files = librosa.util.find_files(searchdir,
                                        ext=ext,
                                        recurse=recurse,
                                        case_sensitive=case_sensitive,
                                        limit=limit,
                                        offset=offset)

        targets = output
        if ext is not None:
            # If we're only seeking wavs, bump off the mp3 file
            targets = targets[1:]

        s1 = slice(offset, None)
        s2 = slice(limit)

        assert set(files) == set(targets[s1][s2])

    if platform.system() == 'Windows':
        cases = [False]
    else:
        cases = [False, True]

    for searchdir in [os.path.join(os.path.curdir, 'tests'),
                      os.path.join(os.path.curdir, 'tests', 'data')]:
        for ext in [None, 'wav', 'WAV', ['wav'], ['WAV']]:
            for recurse in [False, True]:
                for case_sensitive in cases:
                    for limit in [None, 1, 2]:
                        for offset in [0, 1, -1]:
                            tf = __test

                            if searchdir == os.path.join(os.path.curdir, 'tests') and not recurse:
                                tf = pytest.mark.xfail(__test, raises=AssertionError)

                            if (ext is not None and case_sensitive and
                                    (ext == 'WAV' or
                                     set(ext) == set(['WAV']))):

                                tf = pytest.mark.xfail(__test, raises=AssertionError)

                            yield (tf, searchdir, ext, recurse,
                                   case_sensitive, limit, offset)


def test_valid_int():

    def __test(x_in, cast):
        z = librosa.util.valid_int(x_in, cast)

        assert isinstance(z, int)
        if cast is None:
            assert z == int(np.floor(x_in))
        else:
            assert z == int(cast(x_in))

    __test_fail = pytest.mark.xfail(__test, raises=librosa.ParameterError)

    for x in np.linspace(-2, 2, num=6):
        for cast in [None, np.floor, np.ceil, 7]:
            if cast is None or six.callable(cast):
                yield __test, x, cast
            else:
                yield __test_fail, x, cast


def test_valid_intervals():

    def __test(intval):
        librosa.util.valid_intervals(intval)

    tf = pytest.mark.xfail(__test, raises=librosa.ParameterError)

    for d in range(1, 4):
        for n in range(1, 4):
            ivals = np.ones(d * [n])
            for m in range(1, 3):
                slices = [slice(m)] * d
                if m == 2 and d == 2 and n > 1:
                    yield __test, ivals[tuple(slices)]
                else:
                    yield tf, ivals[tuple(slices)]

    # Test for issue #712: intervals must have non-negative duration
    yield tf, np.asarray([[0, 1], [2, 1]])


def test_warning_deprecated():

    @librosa.util.decorators.deprecated('old_version', 'new_version')
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
        assert 'deprecated' in str(out[0].message).lower()


def test_warning_moved():

    @librosa.util.decorators.moved('from', 'old_version', 'new_version')
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
        assert 'moved' in str(out[0].message).lower()


def test_warning_rename_kw_pass():

    warnings.resetwarnings()
    warnings.simplefilter('always')

    ov = librosa.util.Deprecated()
    nv = 23

    with warnings.catch_warnings(record=True) as out:
        v = librosa.util.rename_kw('old', ov, 'new', nv, '0', '1')

        assert v == nv

        # Make sure no warning triggered
        assert len(out) == 0


def test_warning_rename_kw_fail():

    warnings.resetwarnings()
    warnings.simplefilter('always')

    ov = 27
    nv = 23

    with warnings.catch_warnings(record=True) as out:
        v = librosa.util.rename_kw('old', ov, 'new', nv, '0', '1')

        assert v == ov

        # Make sure the warning triggered
        assert len(out) > 0

        # And that the category is correct
        assert out[0].category is DeprecationWarning

        # And that it says the right thing (roughly)
        assert 'renamed' in str(out[0].message).lower()


def test_index_to_slice():

    def __test(idx, idx_min, idx_max, step, pad):

        slices = librosa.util.index_to_slice(idx,
                                             idx_min=idx_min,
                                             idx_max=idx_max,
                                             step=step,
                                             pad=pad)

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

    for indices in [np.arange(10, 90, 10), np.arange(10, 90, 15)]:
        for idx_min in [None, 5, 15]:
            for idx_max in [None, 85, 100]:
                for step in [None, 2]:
                    for pad in [False, True]:
                        yield __test, indices, idx_min, idx_max, step, pad


def test_sync():

    def __test_pass(axis, data, idx):
        # By default, mean aggregation
        dsync = librosa.util.sync(data, idx, axis=axis)
        if data.ndim == 1 or axis == -1:
            assert np.allclose(dsync, 2 * np.ones_like(dsync))
        else:
            assert np.allclose(dsync, data)

        # Explicit mean aggregation
        dsync = librosa.util.sync(data, idx, aggregate=np.mean, axis=axis)
        if data.ndim == 1 or axis == -1:
            assert np.allclose(dsync, 2 * np.ones_like(dsync))
        else:
            assert np.allclose(dsync, data)

        # Max aggregation
        dsync = librosa.util.sync(data, idx, aggregate=np.max, axis=axis)
        if data.ndim == 1 or axis == -1:
            assert np.allclose(dsync, 4 * np.ones_like(dsync))
        else:
            assert np.allclose(dsync, data)

        # Min aggregation
        dsync = librosa.util.sync(data, idx, aggregate=np.min, axis=axis)
        if data.ndim == 1 or axis == -1:
            assert np.allclose(dsync, np.zeros_like(dsync))
        else:
            assert np.allclose(dsync, data)

        # Test for dtype propagation
        assert dsync.dtype == data.dtype

    @pytest.mark.xfail(raises=librosa.ParameterError)
    def __test_fail(data, idx):
        librosa.util.sync(data, idx)

    for ndim in [1, 2, 3]:
        shaper = [1] * ndim
        shaper[-1] = -1

        data = np.mod(np.arange(135), 5)
        frames = np.flatnonzero(data[0] == 0)
        slices = [slice(start, stop) for (start, stop) in zip(frames, frames[1:])]
        data = np.reshape(data, shaper)

        for axis in [0, -1]:
            # Test with list of indices
            yield __test_pass, axis, data, list(frames)
            # Test with ndarray of indices
            yield __test_pass, axis, data, frames
            # Test with list of slices
            yield __test_pass, axis, data, slices

    for bad_idx in [['foo', 'bar'], [None], [slice(None), None]]:
        yield __test_fail, data, bad_idx


def test_roll_sparse():
    srand()

    def __test(fmt, shift, axis, X):

        X_sparse = X.asformat(fmt)
        X_dense = X.toarray()

        Xs_roll = librosa.util.roll_sparse(X_sparse, shift, axis=axis)

        assert scipy.sparse.issparse(Xs_roll)
        assert Xs_roll.format == X_sparse.format

        Xd_roll = librosa.util.roll_sparse(X_dense, shift, axis=axis)

        assert np.allclose(Xs_roll.toarray(), Xd_roll), (X_dense, Xs_roll.toarray(), Xd_roll)

        Xd_roll_np = np.roll(X_dense, shift, axis=axis)

        assert np.allclose(Xd_roll, Xd_roll_np)

    X = scipy.sparse.lil_matrix(np.random.randint(0, high=10, size=(16, 16)))

    for fmt in ['csr', 'csc', 'lil', 'dok', 'coo']:
        for shift in [0, 8, -8, 20, -20]:
            for axis in [0, 1, -1]:
                yield __test, fmt, shift, axis, X


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_roll_sparse_bad_axis():

    X = scipy.sparse.eye(5, format='csr')
    librosa.util.roll_sparse(X, 3, axis=2)


def test_softmask():

    def __test(power, split_zeros):
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

    for power in [1, 2, 50, 100, np.inf]:
        for split_zeros in [False, True]:
            yield __test, power, split_zeros


def test_softmask_int():
    X = 2 * np.ones((3, 3), dtype=np.int32)
    X_ref = np.vander(np.arange(3))

    M1 = librosa.util.softmask(X, X_ref, power=1)
    M2 = librosa.util.softmask(X_ref, X, power=1)

    assert np.allclose(M1 + M2, 1)


def test_softmask_fail():

    failure = pytest.mark.xfail(librosa.util.softmask, raises=librosa.ParameterError)
    yield failure, -np.ones(3), np.ones(3), 1, False
    yield failure, np.ones(3), -np.ones(3), 1, False
    yield failure, np.ones(3), np.ones(4), 1, False
    yield failure, np.ones(3), np.ones(3), 0, False
    yield failure, np.ones(3), np.ones(3), -1, False


def test_tiny():

    def __test(x, value):

        assert value == librosa.util.tiny(x)

    for x, value in [(1, np.finfo(np.float32).tiny),
                     (np.ones(3, dtype=int), np.finfo(np.float32).tiny),
                     (np.ones(3, dtype=np.float32), np.finfo(np.float32).tiny),
                     (1.0, np.finfo(np.float64).tiny),
                     (np.ones(3, dtype=np.float64), np.finfo(np.float64).tiny),
                     (1j, np.finfo(np.complex128).tiny),
                     (np.ones(3, dtype=np.complex64), np.finfo(np.complex64).tiny),
                     (np.ones(3, dtype=np.complex128), np.finfo(np.complex128).tiny)]:
        yield __test, x, value


def test_optional_jit():

    @librosa.util.decorators.optional_jit(nopython=True)
    def __func1(x):
        return x**2

    @librosa.util.decorators.optional_jit
    def __func2(x):
        return x**2

    def __test(f):
        y = f(2)
        assert y == 2**2

    yield __test, __func1
    yield __test, __func2

def test_util_fill_off_diagonal_8_8():
    # Case 1: Square matrix (N=M)
    mut_x = np.ones((8, 8))
    librosa.util.fill_off_diagonal(mut_x, 0.25)

    gt_x = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 1]])

    assert np.array_equal(mut_x, gt_x)
    assert np.array_equal(mut_x, gt_x.T)

def test_util_fill_off_diagonal_8_12():
    # Case 2a: N!=M
    mut_x = np.ones((8, 12))
    librosa.util.fill_off_diagonal(mut_x, 0.25)

    gt_x = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

    assert np.array_equal(mut_x, gt_x)

    # Case 2b: (N!=M).T
    mut_x = np.ones((8, 12)).T
    librosa.util.fill_off_diagonal(mut_x, 0.25)

    assert np.array_equal(mut_x, gt_x.T)


@pytest.mark.parametrize('dtype_A', [np.float32, np.float64])
@pytest.mark.parametrize('dtype_B', [np.float32, np.float64])
def test_nnls_vector(dtype_A, dtype_B):
    srand()

    # Make a random basis
    A = np.random.randn(5, 7).astype(dtype_A)

    # Make a random latent vector
    x = np.random.randn(A.shape[1])**2

    B = A.dot(x).astype(dtype_B)

    x_rec = librosa.util.nnls(A, B)

    assert np.all(x_rec >= 0)
    assert np.sqrt(np.mean((B - A.dot(x_rec))**2)) <= 1e-6


@pytest.mark.parametrize('dtype_A', [np.float32, np.float64])
@pytest.mark.parametrize('dtype_B', [np.float32, np.float64])
@pytest.mark.parametrize('x_size', [3, 30])
def test_nnls_matrix(dtype_A, dtype_B, x_size):
    srand()

    # Make a random basis
    A = np.random.randn(5, 7).astype(dtype_A)

    # Make a random latent matrix
    #   when x_size is 3, B is 7x3 (smaller than A)
    x = np.random.randn(A.shape[1], x_size)**2

    B = A.dot(x).astype(dtype_B)

    x_rec = librosa.util.nnls(A, B)

    assert np.all(x_rec >= 0)
    assert np.sqrt(np.mean((B - A.dot(x_rec))**2)) <= 1e-5


@pytest.mark.parametrize('dtype_A', [np.float32, np.float64])
@pytest.mark.parametrize('dtype_B', [np.float32, np.float64])
@pytest.mark.parametrize('x_size', [16, 64, 256])
def test_nnls_multiblock(dtype_A, dtype_B, x_size):
    srand()

    # Make a random basis
    A = np.random.randn(7, 1025).astype(dtype_A)

    # Make a random latent matrix
    #   when x_size is 3, B is 7x3 (smaller than A)
    x = np.random.randn(A.shape[1], x_size)**2

    B = A.dot(x).astype(dtype_B)

    x_rec = librosa.util.nnls(A, B)

    assert np.all(x_rec >= 0)
    assert np.sqrt(np.mean((B - A.dot(x_rec))**2)) <= 1e-4


@pytest.fixture
def psig():

    # [[0, 1, 2, 3, 4]]
    # axis=1 or -1 ==> [-1.5, 1, 1, 1, -1.5]
    # axis=0 ==> [0, 0, 0, 0, 0]
    return np.arange(0, 5, dtype=float)[np.newaxis]


@pytest.mark.parametrize('edge_order', [1, 2])
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_cyclic_gradient(psig, edge_order, axis):


    grad = librosa.util.cyclic_gradient(psig,
                                        edge_order=edge_order,
                                        axis=axis)

    assert grad.shape == psig.shape
    assert grad.dtype == psig.dtype

    # Check the values
    if axis == 0:
        assert np.allclose(grad, 0)
    else:
        assert np.allclose(grad, [-1.5, 1, 1, 1, -1.5])
