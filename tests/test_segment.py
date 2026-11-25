#!/usr/bin/env python
"""Tests for segmentation functions"""

import contextlib

# Disable cache
import os
from typing import TYPE_CHECKING, cast

with contextlib.suppress(KeyError):
    os.environ.pop("LIBROSA_CACHE_DIR")

import warnings

import numpy as np
import pytest
import scipy
from scipy.spatial.distance import cdist, pdist, squareform

import librosa

if TYPE_CHECKING:
    from types import EllipsisType

    from librosa._typing import _SparseArray


@pytest.mark.parametrize("n", [20, 250])
@pytest.mark.parametrize("k", [None, 5])
@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_cross_similarity(n, k, metric, rng):

    # Make a data matrix
    data_ref = rng.standard_normal(size=(3, n))
    data = rng.standard_normal(size=(3, n + 7))

    D = librosa.segment.cross_similarity(data, data_ref, k=k, metric=metric)

    assert D.shape == (data_ref.shape[1], data.shape[1])

    if k is not None:
        real_k = min(k, n)
        assert not np.any(D.sum(axis=0) != real_k)


def test_cross_similarity_sparse(rng):

    data_ref = rng.standard_normal(size=(3, 50))
    data = rng.standard_normal(size=(3, 100))

    D_sparse = librosa.segment.cross_similarity(data, data_ref, sparse=True)
    D_dense = librosa.segment.cross_similarity(data, data_ref, sparse=False)

    assert scipy.sparse.issparse(D_sparse)
    assert isinstance(D_sparse, scipy.sparse.sparray)
    assert not isinstance(D_sparse, scipy.sparse.spmatrix)
    assert np.allclose(D_sparse.toarray(), D_dense)


def test_cross_similarity_distance(rng):

    data_ref = rng.standard_normal(size=(3, 50))
    data = rng.standard_normal(size=(3, 70))
    distance = cdist(data.T, data_ref.T, metric="sqeuclidean").T
    rec = librosa.segment.cross_similarity(data, data_ref, mode="distance", metric="sqeuclidean", sparse=True)

    assert isinstance(rec, scipy.sparse.sparray)
    assert not isinstance(rec, scipy.sparse.spmatrix)
    i, j, vals = scipy.sparse.find(rec)
    assert np.allclose(vals, distance[i, j])


@pytest.mark.parametrize("metric", ["sqeuclidean", "cityblock"])
@pytest.mark.parametrize("bandwidth", [None, 1])
def test_cross_similarity_affinity(metric, bandwidth, rng):

    data_ref = rng.standard_normal(size=(3, 70))
    data = rng.standard_normal(size=(3, 50))
    distance = cdist(data_ref.T, data.T, metric=metric)
    rec = librosa.segment.cross_similarity(
        data, data_ref, mode="affinity", metric=metric, sparse=True, bandwidth=bandwidth
    )

    assert isinstance(rec, scipy.sparse.sparray)
    assert not isinstance(rec, scipy.sparse.spmatrix)
    i, j, vals = scipy.sparse.find(rec)
    logvals = np.log(vals)

    ratio = -logvals / (distance[i, j] + librosa.util.tiny(distance))

    if bandwidth is None:
        assert np.allclose(-logvals, distance[i, j] * np.nanmax(ratio))
    else:
        assert np.allclose(-logvals, distance[i, j] * bandwidth)


def test_cross_similarity_full():
    data = np.eye(10)
    data_ref = np.eye(10)
    rec = librosa.segment.cross_similarity(data, data_ref, mode="distance", full=True)
    assert np.all(rec >= 0)


def test_cross_similarity_badmode(rng):

    data_ref = rng.standard_normal(size=(3, 70))
    data = rng.standard_normal(size=(3, 50))

    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.cross_similarity(data, data_ref, mode="NOT A MODE", metric="sqeuclidean", sparse=True)


def test_cross_similarity_bad_bandwidth(rng):

    data_ref = rng.standard_normal(size=(3, 50))
    data = rng.standard_normal(size=(3, 70))
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.cross_similarity(data, data_ref, bandwidth=-2, mode="affinity")


def test_cross_similarity_fail_mismatch():
    D1 = np.zeros((3, 3))
    D2 = np.zeros((2, 3))
    with pytest.raises(librosa.ParameterShapeError):
        librosa.segment.cross_similarity(D1, D2)


def test_cross_similarity_multi(rng):
    X1 = rng.standard_normal(size=(2, 10, 100))
    X2 = rng.standard_normal(size=(2, 10, 50))

    R = librosa.segment.cross_similarity(X1, X2, mode="affinity")
    # This should give the same output as if we stacked out the leading channel

    X1f = np.concatenate([X1[0], X1[1]], axis=0)
    X2f = np.concatenate([X2[0], X2[1]], axis=0)
    Rf = librosa.segment.cross_similarity(X1f, X2f, mode="affinity")

    assert np.allclose(R, Rf)


@pytest.mark.parametrize("n", [20, 40])
@pytest.mark.parametrize("k", [None, 5])
@pytest.mark.parametrize("mutual", [False, True])
@pytest.mark.parametrize("width", [1, 5])
@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("self", [False, True])
def test_recurrence_matrix(n, k, width, mutual, metric, self, rng):

    # Make a data matrix
    data = rng.standard_normal(size=(3, n))

    D = librosa.segment.recurrence_matrix(data, k=k, width=width, mutual=mutual, axis=-1, metric=metric, self=self)

    # First test for symmetry
    if mutual:
        assert np.allclose(D, D.T)

    # Test for target-axis invariance
    D_trans = librosa.segment.recurrence_matrix(
        data.T, k=k, width=width, mutual=mutual, axis=0, metric=metric, self=self
    )
    assert np.allclose(D, D_trans)

    # If not mutual, test for correct number of links
    if not mutual and k is not None:
        real_k = min(k, n - width)
        if self:
            real_k += 1
        assert not np.any(D.sum(axis=0) != real_k)

    if self:
        assert np.allclose(np.diag(D), True)

    # Make sure the +- width diagonal is hollow
    # It's easier to test if zeroing out the triangles leaves nothing
    idx = np.tril_indices(n, k=width)

    D[idx] = False
    D.T[idx] = False
    assert not np.any(D)


@pytest.mark.parametrize("width", [1, 5])
@pytest.mark.parametrize("k", [None, 5])
def test_recurrence_connectivity_matches_distance(width, k, rng):
    n = 30
    data = rng.standard_normal(size=(3, n))

    rec_conn = librosa.segment.recurrence_matrix(data, k=k, width=width, mode="connectivity")
    rec_dist = librosa.segment.recurrence_matrix(data, k=k, width=width, mode="distance")
    assert np.array_equal(rec_conn, rec_dist > 0)

    k_eff = int(2 * np.ceil(np.sqrt(n - 2 * width + 1))) if k is None else k
    dist_mat = squareform(pdist(data.T, metric="euclidean"))
    idx_i, idx_j = np.indices(dist_mat.shape)
    dist_mat[np.abs(idx_i - idx_j) < width] = np.inf
    expected = np.zeros_like(rec_conn)
    for j in range(n):
        expected[np.argsort(dist_mat[:, j])[:k_eff], j] = True
    assert np.array_equal(rec_conn, expected)


@pytest.mark.parametrize("sym_val", [False, True])
def test_recurrence_matrix_deprecated_sym(sym_val, rng):
    data = rng.standard_normal(size=(3, 25))
    pattern = (
        r"(?s)recurrence_matrix\(\) keyword argument 'sym' has been "
        r"renamed to 'mutual' in version 1\.1\.0.*2\.0\.0"
    )

    with pytest.warns(FutureWarning, match=pattern) as record:
        D_old = librosa.segment.recurrence_matrix(data, sym=sym_val)
    assert len(record) == 1

    D_new = librosa.segment.recurrence_matrix(data, mutual=sym_val)
    assert np.allclose(D_old, D_new)

    if sym_val:
        with pytest.warns(FutureWarning, match=pattern) as record_tuple:
            D_old_tuple = librosa.segment.recurrence_matrix(data, k=(3, 7), sym=True)
        assert len(record_tuple) == 1
        D_new_tuple = librosa.segment.recurrence_matrix(data, k=(3, 7), mutual=True)
        assert np.allclose(D_old_tuple, D_new_tuple)
    else:
        with pytest.warns(FutureWarning, match=pattern), pytest.raises(librosa.ParameterValueError):
            librosa.segment.recurrence_matrix(data, k=(3, 7), sym=False)


@pytest.mark.parametrize("sym_val", [False, True])
def test_recurrence_matrix_sym_mutual_conflict(sym_val, rng):
    data = rng.standard_normal(size=(3, 25))
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, mutual=True, sym=sym_val)


@pytest.mark.parametrize("mode", ["connectivity", "distance", "affinity"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("k", [None, 5, (5, 8), (8, 5), (5, None), (None, 8), (None, None)])
def test_cross_similarity_mutual(mode, sparse, metric, k, rng):
    n = 25
    n_ref = 35
    data = rng.standard_normal(size=(3, n))
    data_ref = rng.standard_normal(size=(3, n_ref))

    xsim = librosa.segment.cross_similarity(data, data_ref, k=k, metric=metric, mutual=True, sparse=sparse, mode=mode)
    xsim_dense = xsim.toarray() if sparse else xsim
    assert xsim_dense.shape == (n_ref, n)

    if isinstance(k, tuple):
        k0, k1 = k
        k_swapped = (k1, k0)
    else:
        k0 = k1 = k
        k_swapped = k

    xsim_rev = librosa.segment.cross_similarity(
        data_ref, data, k=k_swapped, metric=metric, mutual=True, sparse=sparse, mode=mode
    )
    xsim_rev_dense = xsim_rev.toarray() if sparse else xsim_rev

    # Check exact intersection and symmetry
    fwd = librosa.segment.cross_similarity(data, data_ref, k=k0, metric=metric, mutual=False, mode=mode)
    rev = librosa.segment.cross_similarity(data_ref, data, k=k1, metric=metric, mutual=False, mode=mode)
    if mode == "connectivity":
        assert np.allclose(xsim_dense, xsim_rev_dense.T)
        expected = fwd & rev.T
        assert np.array_equal(xsim_dense, expected)

        k0_eff = int(min(n_ref, 2 * np.ceil(np.sqrt(n_ref)))) if k0 is None else k0
        k1_eff = int(min(n, 2 * np.ceil(np.sqrt(n)))) if k1 is None else k1
        assert np.all(xsim_dense.sum(axis=0) <= k0_eff)
        assert np.all(xsim_dense.sum(axis=1) <= k1_eff)
    elif mode == "distance":
        assert np.allclose(xsim_dense, xsim_rev_dense.T)
        expected = np.minimum(fwd, rev.T)
        assert np.allclose(xsim_dense, expected)
    else:
        # med_k_scalar uses data's independent k0 median, so surviving edges match mutual=False
        mask = xsim_dense > 0
        assert np.array_equal(mask, xsim_rev_dense.T > 0)
        assert np.allclose(xsim_dense[mask], fwd[mask])

        # Fixed scalar bandwidth preserves exact transpose symmetry
        xsim_fixed = librosa.segment.cross_similarity(
            data, data_ref, k=k, metric=metric, mutual=True, mode="affinity", bandwidth=1.0
        )
        xsim_fixed_rev = librosa.segment.cross_similarity(
            data_ref, data, k=k_swapped, metric=metric, mutual=True, mode="affinity", bandwidth=1.0
        )
        assert np.allclose(xsim_fixed, xsim_fixed_rev.T)


@pytest.mark.parametrize("n,n_ref", [(10, 25), (25, 10)])
@pytest.mark.parametrize("mutual", [False, True])
@pytest.mark.parametrize("mode", ["distance", "affinity"])
def test_cross_similarity_full_rectangular(n, n_ref, mutual, mode, rng):
    data = rng.standard_normal(size=(3, n))
    data_ref = rng.standard_normal(size=(3, n_ref))

    xsim = librosa.segment.cross_similarity(data, data_ref, mutual=mutual, mode=mode, full=True)
    assert xsim.shape == (n_ref, n)

    if mode == "distance":
        expected = cdist(data_ref.T, data.T, metric="euclidean")
        assert np.allclose(xsim, expected)
    else:
        assert np.all(xsim > 0)
        if mutual:
            xsim_nomut = librosa.segment.cross_similarity(data, data_ref, mutual=False, mode=mode, full=True)
            assert np.allclose(xsim, xsim_nomut)


@pytest.mark.parametrize("bw_mode", ["mean_k", "gmean_k", "mean_k_avg", "gmean_k_avg", "mean_k_avg_and_pair"])
def test_cross_similarity_automatic_bandwidth(bw_mode, rng):
    data = rng.standard_normal(size=(3, 25))
    data_ref = rng.standard_normal(size=(3, 40))
    data_eq = rng.standard_normal(size=(3, 25))

    # Small asymmetric k with mutual=True: bandwidth is computed from unpruned graphs,
    # so nodes with 0 mutual neighbors do not fail and transpose symmetry holds
    xsim = librosa.segment.cross_similarity(data, data_ref, k=(5, 8), mutual=True, mode="affinity", bandwidth=bw_mode)
    xsim_rev = librosa.segment.cross_similarity(
        data_ref, data, k=(8, 5), mutual=True, mode="affinity", bandwidth=bw_mode
    )
    assert np.all(np.isfinite(xsim))
    assert np.allclose(xsim, xsim_rev.T)

    # mutual=False on both equal-length distinct and rectangular inputs (full=False and full=True)
    for ref in (data_eq, data_ref):
        for full in (False, True):
            res = librosa.segment.cross_similarity(
                data, ref, k=5, mutual=False, mode="affinity", bandwidth=bw_mode, full=full
            )
            assert np.all(np.isfinite(res))
            assert np.any(res > 0)

    # Verify mean_k uses data_ref's own k_ref-NN distances (not data's) even when n == n_ref
    if bw_mode == "mean_k":
        res_eq = librosa.segment.cross_similarity(data, data_eq, k=5, mutual=False, mode="affinity", bandwidth=bw_mode)
        dists = cdist(data_eq.T, data.T, metric="euclidean")
        sigma_data = np.sort(dists, axis=0)[4, :]
        sigma_ref = np.sort(dists, axis=1)[:, 4]
        bw_mat = 0.5 * (sigma_data[None, :] + sigma_ref[:, None])
        expected = np.exp(-dists / bw_mat)
        mask = res_eq > 0
        assert np.allclose(res_eq[mask], expected[mask])


@pytest.mark.parametrize("mode", ["connectivity", "distance", "affinity"])
@pytest.mark.parametrize("width", [1, 5])
@pytest.mark.parametrize("self", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("k", [(3, 7), (7, 3), (5, None), (None, 5), (None, None)])
def test_recurrence_matrix_tuple_k(mode, width, self, sparse, k, rng):
    n = 30
    data = rng.standard_normal(size=(3, n))

    rec = librosa.segment.recurrence_matrix(data, k=k, width=width, mutual=True, sparse=sparse, mode=mode, self=self)
    rec_dense = rec.toarray() if sparse else rec
    assert rec_dense.shape == (n, n)

    k0, k1 = k
    rec_swapped = librosa.segment.recurrence_matrix(
        data, k=(k1, k0), width=width, mutual=True, sparse=sparse, mode=mode, self=self
    )
    rec_swapped_dense = rec_swapped.toarray() if sparse else rec_swapped

    if mode in ("connectivity", "distance"):
        assert np.allclose(rec_dense, rec_swapped_dense.T)

    if mode == "connectivity":
        rec_col = librosa.segment.recurrence_matrix(data, k=k0, width=width, mutual=False, mode=mode, self=self)
        rec_row = librosa.segment.recurrence_matrix(data, k=k1, width=width, mutual=False, mode=mode, self=self)
        assert np.array_equal(rec_dense, rec_col & rec_row.T)

        k_default = int(2 * np.ceil(np.sqrt(n - 2 * width + 1)))
        k0_eff = k_default if k0 is None else k0
        k1_eff = k_default if k1 is None else k1
        diag_extra = 1 if self else 0
        assert np.all(rec_dense.sum(axis=0) <= k0_eff + diag_extra)
        assert np.all(rec_dense.sum(axis=1) <= k1_eff + diag_extra)
    elif mode == "affinity":
        rec_nomut = librosa.segment.recurrence_matrix(data, k=k0, width=width, mutual=False, mode=mode, self=self)
        mask = rec_dense > 0
        assert np.array_equal(mask, rec_swapped_dense.T > 0)
        assert np.allclose(rec_dense[mask], rec_nomut[mask])


@pytest.mark.parametrize("bw_mode", ["mean_k", "gmean_k", "mean_k_avg", "gmean_k_avg", "mean_k_avg_and_pair"])
@pytest.mark.parametrize("self", [False, True])
def test_recurrence_matrix_tuple_k_automatic_bandwidth(bw_mode, self, rng):
    data = rng.standard_normal(size=(3, 30))
    rec = librosa.segment.recurrence_matrix(
        data, k=(3, 7), width=1, mutual=True, mode="affinity", bandwidth=bw_mode, self=self
    )
    rec_swapped = librosa.segment.recurrence_matrix(
        data, k=(7, 3), width=1, mutual=True, mode="affinity", bandwidth=bw_mode, self=self
    )
    assert np.all(np.isfinite(rec))
    assert np.allclose(rec, rec_swapped.T)
    if self:
        assert np.allclose(np.diag(rec), 1.0)
    else:
        assert np.allclose(np.diag(rec), 0.0)


def test_cross_similarity_recurrence_parity(rng):
    data = rng.standard_normal(size=(4, 25))
    k0, k1 = 4, 7
    xsim = librosa.segment.cross_similarity(data, data, k=(k0 + 1, k1 + 1), mutual=True, mode="connectivity")
    rec = librosa.segment.recurrence_matrix(data, k=(k0, k1), width=1, mutual=True, self=True, mode="connectivity")
    assert np.array_equal(xsim, rec)


@pytest.mark.parametrize(
    "k,mutual",
    [
        ((5, 5), False),
        ((5,), True),
        ((5, 5, 5), True),
        (0, False),
        (-1, False),
        (2.5, False),
        (True, False),
        ((0, 5), True),
        ((5, -1), True),
        ((5, True), True),
        ((5, 2.5), True),
    ],
)
def test_segment_bad_k(k, mutual, rng):
    data = rng.standard_normal(size=(3, 20))
    data_ref = rng.standard_normal(size=(3, 25))
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.cross_similarity(data, data_ref, k=k, mutual=mutual)
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, k=k, mutual=mutual)


@pytest.mark.parametrize("data", [np.ones((3, 10))])
@pytest.mark.parametrize("width", [-1, 0, 11])
def test_recurrence_badwidth(data, width):
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, width=width)


@pytest.mark.parametrize("self", [False, True])
def test_recurrence_sparse(self, rng):

    data = rng.standard_normal(size=(3, 100))
    D_sparse = librosa.segment.recurrence_matrix(data, sparse=True, self=self)
    D_dense = librosa.segment.recurrence_matrix(data, sparse=False, self=self)

    assert scipy.sparse.issparse(D_sparse)
    assert isinstance(D_sparse, scipy.sparse.sparray)
    assert not isinstance(D_sparse, scipy.sparse.spmatrix)
    assert np.allclose(D_sparse.toarray(), D_dense)

    if self:
        assert np.allclose(D_sparse.diagonal(), True)
    else:
        assert np.allclose(D_sparse.diagonal(), False)


@pytest.mark.parametrize("self", [False, True])
def test_recurrence_distance(self, rng):

    data = rng.standard_normal(size=(3, 100))
    distance = squareform(pdist(data.T, metric="sqeuclidean"))
    rec = librosa.segment.recurrence_matrix(data, mode="distance", metric="sqeuclidean", sparse=True, self=self)

    assert isinstance(rec, scipy.sparse.sparray)
    assert not isinstance(rec, scipy.sparse.spmatrix)
    i, j, vals = scipy.sparse.find(rec)
    assert np.allclose(vals, distance[i, j])
    assert np.allclose(rec.diagonal(), 0.0)


@pytest.mark.parametrize("metric", ["sqeuclidean", "cityblock"])
@pytest.mark.parametrize("bandwidth", [None, 1])
@pytest.mark.parametrize("self", [False, True])
def test_recurrence_affinity(metric, bandwidth, self, rng):

    data = rng.standard_normal(size=(3, 100))
    distance = squareform(pdist(data.T, metric=metric))
    rec = librosa.segment.recurrence_matrix(
        data,
        mode="affinity",
        metric=metric,
        sparse=True,
        bandwidth=bandwidth,
        self=self,
    )

    assert isinstance(rec, scipy.sparse.sparray)
    assert not isinstance(rec, scipy.sparse.spmatrix)
    if self:
        assert np.allclose(rec.diagonal(), 1.0)
    else:
        assert np.allclose(rec.diagonal(), 0.0)

    i, j, vals = scipy.sparse.find(rec)
    logvals = np.log(vals)

    # After log-scaling, affinity will match distance up to a constant factor
    ratio = -logvals / (distance[i, j] + librosa.util.tiny(distance))
    if bandwidth is None:
        # Estimate the global bandwidth using non-zero distances
        assert np.allclose(-logvals, distance[i, j] * np.nanmax(ratio))
    else:
        assert np.allclose(-logvals, distance[i, j] * bandwidth)


def test_recurrence_full():
    data = np.eye(10)
    rec = librosa.segment.recurrence_matrix(data, mode="distance", metric="euclidean", sparse=False, full=True)
    assert np.all(rec >= 0)


def test_big_width(rng):
    data = rng.standard_normal(size=(3, 100))
    width = 55
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, mode="affinity", width=width)


def test_empty_data_recurrence():
    data = np.zeros((10, 10))
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, mode="affinity")


def test_empty_rows_recurrence():
    data = np.zeros((10, 10))
    data[0, 5] = 1
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, mode="affinity", bandwidth="mean_k")


def test_empty_rows_recurrence_okay():
    data = np.zeros((10, 10))
    data[0, 5] = 1
    librosa.segment.recurrence_matrix(data, mode="affinity", bandwidth="med_k_scalar")


def test_recurrence_multi(rng):
    X = rng.standard_normal(size=(2, 10, 100))

    R = librosa.segment.recurrence_matrix(X, mode="affinity")
    # This should give the same output as if we stacked out the leading channel

    Xf = np.concatenate([X[0], X[1]], axis=0)
    Rf = librosa.segment.recurrence_matrix(Xf, mode="affinity")

    assert np.allclose(R, Rf)


def test_recurrence_badmode(rng):

    data = rng.standard_normal(size=(3, 100))

    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, mode="NOT A MODE", metric="sqeuclidean", sparse=True)


@pytest.mark.parametrize(
    "bandwidth",
    [
        -2,
        True,
        np.nan,
        np.complex128(1 + 2j),
        "FAKE",
        np.ones((2, 5)),
        -1 * np.ones((100, 100)),
        np.full((100, 100), np.nan),
        np.ones((100, 100), dtype=complex),
        np.ones((100, 100), dtype=bool),
    ],
)
def test_recurrence_bad_bandwidth(bandwidth, rng):
    data = rng.standard_normal(size=(3, 100))
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.recurrence_matrix(data, bandwidth=bandwidth, mode="affinity")


@pytest.mark.parametrize("self", [False, True])
@pytest.mark.parametrize("mutual", [False, True])
def test_recurrence_array_bandwidth(self, mutual, rng):
    data = rng.standard_normal(size=(3, 30))
    bw = rng.random((30, 30)) + 0.1
    distance = squareform(pdist(data.T, metric="euclidean"))
    rec = librosa.segment.recurrence_matrix(data, k=5, bandwidth=bw, mode="affinity", self=self, mutual=mutual)

    mask = rec > 0
    if self:
        assert np.allclose(np.diag(rec), 1.0)
        np.fill_diagonal(mask, False)
    else:
        assert np.allclose(np.diag(rec), 0.0)

    assert np.allclose(rec[mask], np.exp(-distance[mask] / bw[mask]))


@pytest.mark.parametrize("mutual", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_cross_similarity_array_bandwidth(mutual, sparse, rng):
    data = rng.standard_normal(size=(3, 20))
    data_ref = rng.standard_normal(size=(3, 35))
    bw = rng.random((35, 20)) + 0.1
    distance = cdist(data_ref.T, data.T, metric="euclidean")

    xsim = librosa.segment.cross_similarity(
        data, data_ref, k=5, bandwidth=bw, mode="affinity", mutual=mutual, sparse=sparse
    )
    xsim_dense = xsim.toarray() if sparse else xsim
    assert xsim_dense.shape == (35, 20)

    mask = xsim_dense > 0
    assert np.any(mask)
    assert np.allclose(xsim_dense[mask], np.exp(-distance[mask] / bw[mask]))


@pytest.mark.parametrize(
    "bw,exc",
    [
        (np.ones((20, 35)), librosa.ParameterShapeError),
        (-np.ones((35, 20)), librosa.ParameterValueError),
        (np.full((35, 20), np.nan), librosa.ParameterValueError),
        (np.ones((35, 20), dtype=complex), librosa.ParameterValueError),
        (np.nan, librosa.ParameterValueError),
        (np.complex128(1 + 2j), librosa.ParameterValueError),
        (True, librosa.ParameterValueError),
    ],
)
def test_cross_similarity_bad_array_bandwidth(bw, exc, rng):
    data = rng.standard_normal(size=(3, 20))
    data_ref = rng.standard_normal(size=(3, 35))
    with pytest.raises(exc):
        librosa.segment.cross_similarity(data, data_ref, bandwidth=bw, mode="affinity")


@pytest.mark.parametrize("bw", [np.float32(1.5), np.int64(2)])
def test_segment_numpy_scalar_bandwidth(bw, rng):
    data = rng.standard_normal(size=(3, 20))
    data_ref = rng.standard_normal(size=(3, 25))
    c_np = librosa.segment.cross_similarity(data, data_ref, k=4, mode="affinity", bandwidth=bw)
    c_py = librosa.segment.cross_similarity(data, data_ref, k=4, mode="affinity", bandwidth=float(bw))
    assert np.allclose(c_np, c_py)

    r_np = librosa.segment.recurrence_matrix(data, k=4, mode="affinity", bandwidth=bw)
    r_py = librosa.segment.recurrence_matrix(data, k=4, mode="affinity", bandwidth=float(bw))
    assert np.allclose(r_np, r_py)


@pytest.mark.parametrize("bw_mode", ["mean_k", "gmean_k", "mean_k_avg", "gmean_k_avg", "mean_k_avg_and_pair"])
def test_automatic_bandwidth(bw_mode, rng):
    data = rng.standard_normal(size=(3, 100))
    librosa.segment.recurrence_matrix(data, bandwidth=bw_mode, mode="affinity")


@pytest.mark.parametrize("n", [10, 100, 500])
@pytest.mark.parametrize("pad", [False, True])
def test_recurrence_to_lag(n, pad, rng):

    data = rng.standard_normal(size=(17, n))

    rec = librosa.segment.recurrence_matrix(data)

    lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=-1)
    lag2 = librosa.segment.recurrence_to_lag(rec.T, pad=pad, axis=0)

    assert np.allclose(lag, lag2.T)

    x: EllipsisType | slice = Ellipsis
    if pad:
        x = slice(n)

    for i in range(n):
        assert np.allclose(rec[:, i], np.roll(lag[:, i], i)[x])


@pytest.mark.parametrize("size", [(17,), (17, 34), (17, 17, 17)])
def test_recurrence_to_lag_fail(size):
    with pytest.raises(librosa.ParameterShapeError):
        librosa.segment.recurrence_to_lag(np.zeros(size))


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("rec", [librosa.segment.recurrence_matrix(np.ones((3, 100)), sparse=True)])
@pytest.mark.parametrize("fmt", ["csc", "csr", "lil", "bsr", "dia"])
# This warning is expected when using fmt='dia'
@pytest.mark.filterwarnings("ignore:Constructing a DIA matrix")
def test_recurrence_to_lag_sparse(pad, axis, rec, fmt):

    rec_dense = rec.toarray()

    rec = rec.asformat(fmt)

    lag_sparse = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=axis)
    lag_dense = librosa.segment.recurrence_to_lag(rec_dense, pad=pad, axis=axis)

    assert scipy.sparse.issparse(lag_sparse)
    assert isinstance(lag_sparse, scipy.sparse.sparray)
    assert not isinstance(lag_sparse, scipy.sparse.spmatrix)

    lag_sparse = cast("_SparseArray", lag_sparse)

    assert rec.format == lag_sparse.format
    assert rec.dtype == lag_sparse.dtype
    assert np.allclose(lag_sparse.toarray(), lag_dense)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("pad", [False, True])
def test_lag_to_recurrence(n, pad, rng):

    data = rng.standard_normal(size=(17, n))

    rec = librosa.segment.recurrence_matrix(data)
    lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=-1)
    lag2 = librosa.segment.recurrence_to_lag(rec.T, pad=pad, axis=0).T

    rec2 = librosa.segment.lag_to_recurrence(lag)

    assert np.allclose(rec, rec2)
    assert np.allclose(lag, lag2)


@pytest.mark.parametrize("size", [(17,), (17, 35), (17, 17, 17)])
def test_lag_to_recurrence_badsize(size):
    with pytest.raises(librosa.ParameterShapeError):
        librosa.segment.lag_to_recurrence(np.zeros(size))


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("pad", [False, True])
def test_lag_to_recurrence_sparse(axis, pad, rng):

    data = rng.standard_normal(size=(3, 10))
    rec = librosa.segment.recurrence_matrix(data, sparse=True)
    lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=axis)

    lag_dense = lag.toarray()

    rec_sparse = librosa.segment.lag_to_recurrence(lag, axis=axis)
    rec_dense = librosa.segment.lag_to_recurrence(lag_dense, axis=axis)

    assert scipy.sparse.issparse(rec_sparse)
    assert isinstance(rec_sparse, scipy.sparse.sparray)
    assert not isinstance(rec_sparse, scipy.sparse.spmatrix)
    assert rec_sparse.format == lag.format
    assert rec_sparse.dtype == lag.dtype
    assert np.allclose(rec_sparse.toarray(), rec_dense)


@pytest.mark.parametrize("pad", [False, True])
def test_recurrence_lag_spmatrix_deprecated(pad):
    rec_arr = scipy.sparse.eye_array(8, format="csc")
    rec_mat = scipy.sparse.eye(8, format="csc")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        lag_arr = librosa.segment.recurrence_to_lag(rec_arr, pad=pad)
        librosa.segment.lag_to_recurrence(lag_arr)

    with pytest.warns(FutureWarning, match="spmatrix") as record:
        lag_mat = librosa.segment.recurrence_to_lag(rec_mat, pad=pad)  # type: ignore[type-var]
    assert len(record) == 1

    with pytest.warns(FutureWarning, match="spmatrix") as record2:
        librosa.segment.lag_to_recurrence(lag_mat)  # type: ignore[type-var]
    assert len(record2) == 1


def test_lag_to_recurrence_sparse_badaxis(rng):

    data = rng.standard_normal(size=(3, 100))
    R = librosa.segment.recurrence_matrix(data, sparse=True)
    L = librosa.segment.recurrence_to_lag(R)
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.lag_to_recurrence(L, axis=2)


def test_timelag_filter(rng):
    X = rng.standard_normal(size=(15, 15))
    d_pos0 = librosa.segment.timelag_filter(lambda X: X)
    assert np.allclose(X, d_pos0(X))


def test_timelag_filter_pos1(rng):
    X = rng.standard_normal(size=(15, 15))
    d_pos1 = librosa.segment.timelag_filter(lambda _, X: X, index=1)
    assert np.allclose(X, d_pos1(None, X))


@pytest.fixture(scope="module")
def ysr():
    sr = 22050
    y = librosa.chirp(fmin=100, fmax=2000, sr=sr, duration=5.0)
    return y, sr


@pytest.fixture(scope="module")
def mfcc(ysr):
    y, sr = ysr
    return librosa.feature.mfcc(y=y, sr=sr)


@pytest.fixture(scope="module")
def beats(ysr):
    y, sr = ysr
    _tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return beats


@pytest.mark.parametrize("n_segments", [1, 2, 3, 4, 100])
def test_subsegment(mfcc, beats, n_segments):

    subseg = librosa.segment.subsegment(mfcc, beats, n_segments=n_segments, axis=-1)

    # Make sure that the boundaries are within range
    assert subseg.min() >= 0
    assert subseg.max() <= mfcc.shape[-1]

    # Make sure that all input beats are retained
    for b in beats:
        assert b in subseg

    # Do we have a 0 marker?
    assert 0 in subseg

    # Did we over-segment?  +2 here for 0- and end-padding
    assert len(subseg) <= n_segments * (len(beats) + 2)

    # Verify that running on the transpose gives the same answer
    ss2 = librosa.segment.subsegment(mfcc.T, beats, n_segments=n_segments, axis=0)
    assert np.allclose(subseg, ss2)


@pytest.mark.parametrize("n_segments", [-1, 0])
def test_subsegment_badn(mfcc, beats, n_segments):
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.subsegment(mfcc, beats, n_segments=n_segments, axis=-1)


@pytest.fixture
def R_input(rng):
    X = rng.standard_normal(size=(30, 5))

    return X.dot(X.T)


@pytest.mark.parametrize("window", ["rect", "hann"])
@pytest.mark.parametrize("n", [5, 9])
@pytest.mark.parametrize("max_ratio", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("min_ratio", [None, 1.0])
@pytest.mark.parametrize("n_filters", [1, 2, 5])
@pytest.mark.parametrize("zero_mean", [False, True])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("kwargs", [dict(), dict(mode="reflect")])
def test_path_enhance(R_input, window, n, max_ratio, min_ratio, n_filters, zero_mean, clip, kwargs):

    R_smooth = librosa.segment.path_enhance(
        R_input,
        window=window,
        n=n,
        max_ratio=max_ratio,
        min_ratio=min_ratio,
        n_filters=n_filters,
        zero_mean=zero_mean,
        clip=clip,
        **kwargs,
    )

    assert R_smooth.shape == R_input.shape
    assert np.all(np.isfinite(R_smooth))
    assert R_smooth.dtype == R_input.dtype

    if clip:
        assert np.min(R_smooth) >= 0


def test_path_enhance_badratio(R_input):
    # We can't have min_ratio > max_ratio
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.path_enhance(R_input, n=5, min_ratio=3, max_ratio=2)


def test_path_enhance_multi(rng):

    R = rng.standard_normal(size=(2, 100, 100))

    Rs0 = librosa.segment.path_enhance(R[0], n=5)
    Rs1 = librosa.segment.path_enhance(R[1], n=5)
    Rs = librosa.segment.path_enhance(R, n=5)

    assert np.allclose(Rs0, Rs[0])
    assert np.allclose(Rs1, Rs[1])
    assert not np.allclose(Rs0, Rs1)


@pytest.mark.parametrize("M", [3, 4, 5])
@pytest.mark.parametrize("mode", ["same", "valid", "full"])
@pytest.mark.parametrize("k", [-3, 0, 2])
def test_diagonal_convolve_scipy_parity(rng, M, mode, k):
    import scipy.signal

    R = rng.standard_normal(size=(25, 25))
    K = rng.standard_normal(size=(M, M))

    ref = np.diagonal(
        scipy.signal.convolve2d(R, K, mode=mode, boundary="fill", fillvalue=0.0),
        offset=k,
    )
    est = librosa.segment.diagonal_convolve(R, K, k=k, mode=mode, boundary="fill", fillvalue=0.0)

    assert np.allclose(ref, est, atol=1e-6)


def test_diagonal_convolve_separable(rng):
    R = rng.standard_normal(size=(30, 30))
    u = rng.standard_normal(size=7)
    v = rng.standard_normal(size=7)
    K_2d = np.outer(u, v)

    # 1. 2D kernel
    out_2d = librosa.segment.diagonal_convolve(R, K_2d, mode="same")

    # 2. Tuple of 1D arrays
    out_tuple = librosa.segment.diagonal_convolve(R, (u, v), mode="same")
    assert np.allclose(out_2d, out_tuple, atol=1e-6)

    # 3. 1D array (symmetric kernel)
    K_sym = np.outer(u, u)
    out_sym_2d = librosa.segment.diagonal_convolve(R, K_sym, mode="same")
    out_sym_1d = librosa.segment.diagonal_convolve(R, u, mode="same")
    assert np.allclose(out_sym_2d, out_sym_1d, atol=1e-6)


@pytest.mark.parametrize("boundary", ["fill", "symm", "wrap", "reflect", "edge"])
def test_diagonal_convolve_boundaries(rng, boundary):
    R = rng.standard_normal(size=(20, 20))
    K = rng.standard_normal(size=(5, 5))

    out = librosa.segment.diagonal_convolve(R, K, mode="same", boundary=boundary, fillvalue=1.5)
    assert out.shape == (20,)
    assert np.all(np.isfinite(out))


def test_diagonal_convolve_multichannel(rng):
    R = rng.standard_normal(size=(2, 3, 25, 25))
    K = rng.standard_normal(size=(5, 5))

    out = librosa.segment.diagonal_convolve(R, K, mode="same")
    assert out.shape == (2, 3, 25)

    for i in range(2):
        for j in range(3):
            single = librosa.segment.diagonal_convolve(R[i, j], K, mode="same")
            assert np.allclose(out[i, j], single)


def test_diagonal_convolve_sparse_raises():
    import scipy.sparse

    R_sparse = scipy.sparse.csr_array(np.eye(20))
    K = np.ones((5, 5))

    with pytest.raises(librosa.ParameterTypeError):
        librosa.segment.diagonal_convolve(R_sparse, K)  # type: ignore[arg-type]


def test_diagonal_convolve_bad_inputs():
    R = np.eye(20)

    # Invalid kernel dimensions
    with pytest.raises(librosa.ParameterShapeError):
        librosa.segment.diagonal_convolve(R, np.ones((2, 2, 2)))

    with pytest.raises(librosa.ParameterShapeError):
        librosa.segment.diagonal_convolve(R, (np.ones((2, 2)), np.ones(2)))

    # Empty kernel
    with pytest.raises(librosa.ParameterShapeError):
        librosa.segment.diagonal_convolve(R, np.empty((0, 0)))

    # Invalid mode
    with pytest.raises(librosa.ParameterValueError):
        librosa.segment.diagonal_convolve(
            R,
            np.ones((3, 3)),
            mode="invalid",  # type: ignore[arg-type]
        )


def test_diagonal_convolve_empty_dimensions():
    # mode='valid' with kernel larger than input matrix -> grid_rows/cols <= 0
    out_valid = librosa.segment.diagonal_convolve(np.ones((5, 5)), np.ones((10, 10)), mode="valid")
    assert out_valid.shape == (0,)

    # Out-of-bounds offset k -> length == 0
    out_pos = librosa.segment.diagonal_convolve(np.ones((10, 10)), np.ones((3, 3)), k=15, mode="same")
    assert out_pos.shape == (0,)

    out_neg = librosa.segment.diagonal_convolve(np.ones((10, 10)), np.ones((3, 3)), k=-15, mode="same")
    assert out_neg.shape == (0,)


def test_recurrence_novelty_libfmp_parity(rng):
    # Reference libfmp implementation of compute_novelty_ssm
    def libfmp_novelty_ssm(S, L=5, var=0.5, exclude=False):
        # 1. Kernel
        s = np.arange(-L, L + 1)
        X, Y = np.meshgrid(s, s)
        kernel = (np.sign(X) * np.sign(Y)) * np.exp(-0.5 * (X**2 + Y**2) / (var * L) ** 2)
        kernel /= np.sum(np.abs(kernel))

        # 2. Padded sliding sum
        N = S.shape[0]
        M = 2 * L + 1
        nov = np.zeros(N)
        S_padded = np.pad(S, L, mode="constant")
        for n in range(N):
            nov[n] = np.sum(S_padded[n : n + M, n : n + M] * kernel)

        if exclude:
            nov[0:L] = 0
            nov[N - L : N] = 0
        return nov

    N = 30
    L = 5
    n = 2 * L + 1
    S = rng.uniform(low=0.0, high=1.0, size=(N, N))
    S = 0.5 * (S + S.T)  # symmetric SSM

    ref_nov = libfmp_novelty_ssm(S, L=L, var=0.5, exclude=False)
    # clip=False to match raw libfmp sum
    est_nov = librosa.segment.recurrence_novelty(S, n=n, window="gaussian", var=0.5, norm=1, clip=False, exclude=False)
    assert np.allclose(ref_nov, est_nov, atol=1e-5)

    # Test exclude=True
    ref_ex = libfmp_novelty_ssm(S, L=L, var=0.5, exclude=True)
    est_ex = librosa.segment.recurrence_novelty(S, n=n, window="gaussian", var=0.5, norm=1, clip=False, exclude=True)
    assert np.allclose(ref_ex, est_ex, atol=1e-5)


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("exclude", [False, True])
@pytest.mark.parametrize("mode", ["same", "valid", "full"])
def test_recurrence_novelty_options(rng, clip, exclude, mode):
    R = rng.standard_normal(size=(30, 30))
    nov = librosa.segment.recurrence_novelty(R, n=9, window="boxcar", clip=clip, exclude=exclude, mode=mode)

    if mode == "same":
        assert nov.shape == (30,)
        if exclude:
            assert np.all(nov[:4] == 0.0)
            assert np.all(nov[-4:] == 0.0)
    elif mode == "valid":
        assert nov.shape == (22,)
    elif mode == "full":
        assert nov.shape == (38,)

    if clip:
        assert np.all(nov >= 0.0)


def test_recurrence_novelty_precomputed_kernel(rng):
    R = rng.standard_normal(size=(25, 25))
    K = librosa.filters.checkerboard_kernel(15, window="hann")

    nov = librosa.segment.recurrence_novelty(R, kernel=K)
    assert nov.shape == (25,)
    assert np.all(np.isfinite(nov))


def test_recurrence_novelty_multichannel(rng):
    R = rng.standard_normal(size=(2, 20, 20))
    nov = librosa.segment.recurrence_novelty(R, n=7)
    assert nov.shape == (2, 20)

    nov0 = librosa.segment.recurrence_novelty(R[0], n=7)
    nov1 = librosa.segment.recurrence_novelty(R[1], n=7)
    assert np.allclose(nov[0], nov0)
    assert np.allclose(nov[1], nov1)


def test_recurrence_novelty_exclude_kernel_types():
    R = np.eye(20)

    # 1. Tuple kernel (u, v)
    nov_tuple = librosa.segment.recurrence_novelty(R, kernel=(np.ones(5), np.ones(5)), exclude=True)
    assert np.all(nov_tuple[:2] == 0.0)
    assert np.all(nov_tuple[-2:] == 0.0)

    # 2. 1D kernel array
    nov_1d = librosa.segment.recurrence_novelty(R, kernel=np.ones(5), exclude=True)
    assert np.all(nov_1d[:2] == 0.0)
    assert np.all(nov_1d[-2:] == 0.0)

    # 3. 2D kernel array
    nov_2d = librosa.segment.recurrence_novelty(R, kernel=np.ones((5, 5)), exclude=True)
    assert np.all(nov_2d[:2] == 0.0)
    assert np.all(nov_2d[-2:] == 0.0)

    # 4. List kernel (hits else: k_len = n)
    nov_list = librosa.segment.recurrence_novelty(
        R,
        n=5,
        kernel=[[1, -1], [-1, 1]],  # type: ignore[arg-type]
        exclude=True,
    )
    assert np.all(nov_list[:2] == 0.0)
    assert np.all(nov_list[-2:] == 0.0)
