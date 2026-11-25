#!/usr/bin/env python
"""
librosa.segment provides functions for structural analysis and
segmentation of audio.

This includes recurrence (self-similarity) and cross-similarity matrix
computation, filtering and transformations of similarity matrices, and
temporal segmentation by clustering.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import scipy
from decorator import decorator

from . import util
from ._cache import cache
from .filters import checkerboard_kernel, diagonal_filter
from .util.deprecation import Deprecated, _warn_spmatrix, rename_kw
from .util.exceptions import (
    ParameterShapeError,
    ParameterTypeError,
    ParameterValueError,
)
from .util.utils import _shear

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    import sklearn.cluster
    import sklearn.neighbors

    from ._typing import _Array1D, _FloatLike_co, _SparseArray, _WindowSpec


__all__ = [
    "cross_similarity",
    "recurrence_matrix",
    "recurrence_to_lag",
    "lag_to_recurrence",
    "timelag_filter",
    "agglomerative",
    "subsegment",
    "path_enhance",
    "diagonal_convolve",
    "recurrence_novelty",
]


@overload
def cross_similarity(
    data: np.ndarray,
    data_ref: np.ndarray,
    *,
    k: int | tuple[int | None, int | None] | None = ...,
    metric: str = ...,
    mutual: bool = ...,
    sparse: Literal[False] = ...,
    mode: str = ...,
    bandwidth: np.ndarray | _FloatLike_co | str | None = None,
    full: bool = False,
) -> np.ndarray: ...
@overload
def cross_similarity(
    data: np.ndarray,
    data_ref: np.ndarray,
    *,
    k: int | tuple[int | None, int | None] | None = ...,
    metric: str = ...,
    mutual: bool = ...,
    sparse: Literal[True],
    mode: str = ...,
    bandwidth: np.ndarray | _FloatLike_co | str | None = None,
    full: bool = False,
) -> scipy.sparse.csc_array: ...
@cache(level=30)
def cross_similarity(
    data: np.ndarray,
    data_ref: np.ndarray,
    *,
    k: int | tuple[int | None, int | None] | None = None,
    metric: str = "euclidean",
    mutual: bool = False,
    sparse: bool = False,
    mode: str = "connectivity",
    bandwidth: np.ndarray | _FloatLike_co | str | None = None,
    full: bool = False,
) -> np.ndarray | scipy.sparse.csc_array:
    """Compute cross-similarity from one data sequence to a reference sequence.

    The output is a matrix ``xsim``, where ``xsim[i, j]`` is non-zero
    if ``data_ref[..., i]`` is a k-nearest neighbor of ``data[..., j]``.
    If ``mutual=True``, ``xsim[i, j]`` is non-zero only if ``data_ref[..., i]``
    is a k-nearest neighbor of ``data[..., j]`` and vice versa.

    Parameters
    ----------
    data : np.ndarray [shape=(..., d, n)]
        A feature matrix for the comparison sequence.
        If the data has more than two dimensions (e.g., for multi-channel inputs),
        the leading dimensions are flattened prior to comparison.
        For example, a stereo input with shape `(2, d, n)` is
        automatically reshaped to `(2 * d, n)`.

    data_ref : np.ndarray [shape=(..., d, n_ref)]
        A feature matrix for the reference sequence
        If the data has more than two dimensions (e.g., for multi-channel inputs),
        the leading dimensions are flattened prior to comparison.
        For example, a stereo input with shape `(2, d, n_ref)` is
        automatically reshaped to `(2 * d, n_ref)`.

    k : int > 0, tuple of (int > 0 or None, int > 0 or None), or None
        the number of nearest-neighbors for each sample.

        If a 2-tuple ``(k_data, k_ref)`` is provided (only valid when
        ``mutual=True``), ``k_data`` specifies the number of nearest neighbors
        from ``data_ref`` for each sample in ``data``, and ``k_ref`` specifies
        the number of nearest neighbors from ``data`` for each sample in
        ``data_ref``. Either element may be ``None`` to use its default.

        Default for ``data``: ``k_data = min(n_ref, 2 * ceil(sqrt(n_ref)))``
        Default for ``data_ref`` (when ``mutual=True``):
        ``k_ref = min(n, 2 * ceil(sqrt(n)))``

    metric : str
        Distance metric to use for nearest-neighbor calculation.

        See `sklearn.neighbors.NearestNeighbors` for details.

    mutual : bool [scalar]
        If ``True``, only link mutual nearest neighbors: ``(i, j)`` is linked
        if ``data_ref[..., i]`` is a nearest neighbor of ``data[..., j]``
        and ``data[..., j]`` is a nearest neighbor of ``data_ref[..., i]``.
        Candidate neighborhoods are computed strictly between ``data`` and
        ``data_ref``.

    sparse : bool [scalar]
        if False, returns a dense type (ndarray)
        if True, returns a sparse type (scipy.sparse.csc_array)

    mode : str, {'connectivity', 'distance', 'affinity'}
        If 'connectivity', a binary connectivity matrix is produced.

        If 'distance', then a non-zero entry contains the distance between
        points.

        If 'affinity', then non-zero entries are mapped to
        ``exp( - distance(i, j) / bandwidth)`` where ``bandwidth`` is
        as specified below.

    bandwidth : None, float > 0, ndarray, or str
        str options include
            - 'med_k_scalar'
            - 'mean_k'
            - 'gmean_k'
            - 'mean_k_avg'
            - 'gmean_k_avg'
            - 'mean_k_avg_and_pair'

        If ndarray [shape=(n_ref, n)] is supplied, use ndarray as bandwidth for each (i, j) pair.

        If using ``mode='affinity'``, this can be used to set the
        bandwidth on the affinity kernel.

        If no value is provided or ``None``, default to ``'med_k_scalar'``.

        If ``bandwidth='med_k_scalar'``, bandwidth is set automatically to the median
        distance to the k'th nearest neighbor of each ``data[:, i]``.

        If ``bandwidth='mean_k'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        arithmetic mean between distances to the k-th nearest neighbor for sample i and sample j.

        If ``bandwidth='gmean_k'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        geometric mean between distances to the k-th nearest neighbor for sample i and j [#z]_.

        If ``bandwidth='mean_k_avg'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        arithmetic mean between the average distances to the first k-th nearest neighbors for
        sample i and sample j.
        This is similar to the approach in Wang et al. (2014) [#w]_ but does not include the distance
        between i and j.

        If ``bandwidth='gmean_k_avg'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        geometric mean between the average distances to the first k-th nearest neighbors for
        sample i and sample j.

        If ``bandwidth='mean_k_avg_and_pair'``, bandwidth is estimated for each sample-pair (i, j) by
        taking the arithmetic mean between three terms: the average distances to the first
        k-th nearest neighbors for sample i and sample j respectively, as well as
        the distance between i and j.
        This is similar to the approach in Wang et al. (2014). [#w]_

        .. [#z] Zelnik-Manor, Lihi, and Pietro Perona. (2004).
            "Self-tuning spectral clustering." Advances in neural information processing systems 17.

        .. [#w] Wang, Bo, et al. (2014).
            "Similarity network fusion for aggregating data types on a genomic scale." Nat Methods 11, 333--337.
            https://doi.org/10.1038/nmeth.2810

    full : bool
        If using ``mode='affinity'`` or ``mode='distance'``, this option can be used to compute
        the full affinity or distance matrix as opposed to a sparse matrix with only non-zero terms
        for the first k-neighbors of each sample.
        This option has no effect when using ``mode='connectivity'``.

        When using ``mode='distance'``, setting ``full=True`` will ignore ``k``.
        When using ``mode='affinity'``, setting ``full=True`` will use ``k`` exclusively for
        bandwidth estimation.

    Returns
    -------
    xsim : np.ndarray or scipy.sparse.csc_array, [shape=(n_ref, n)]
        Cross-similarity matrix

    See Also
    --------
    recurrence_matrix
    recurrence_to_lag
    librosa.feature.stack_memory
    sklearn.neighbors.NearestNeighbors
    scipy.spatial.distance.cdist

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Find nearest neighbors in CQT space between two sequences

    >>> hop_length = 1024
    >>> y_ref, sr = librosa.loadx("pistachio")
    >>> y_comp, sr = librosa.loadx("pistachio", offset=10)
    >>> chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sr, hop_length=hop_length)
    >>> chroma_comp = librosa.feature.chroma_cqt(y=y_comp, sr=sr, hop_length=hop_length)
    >>> # Use time-delay embedding to get a cleaner recurrence matrix
    >>> x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    >>> x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)
    >>> xsim = librosa.segment.cross_similarity(x_comp, x_ref)

    Or fix the number of nearest neighbors to 5

    >>> xsim = librosa.segment.cross_similarity(x_comp, x_ref, k=5)

    Require mutual nearest neighbors

    >>> xsim_mutual = librosa.segment.cross_similarity(
    ...     x_comp, x_ref, k=(5, 7), mutual=True
    ... )

    Use cosine similarity instead of Euclidean distance

    >>> xsim = librosa.segment.cross_similarity(x_comp, x_ref, metric="cosine")

    Use an affinity matrix instead of binary connectivity

    >>> xsim_aff = librosa.segment.cross_similarity(
    ...     x_comp, x_ref, metric="cosine", mode="affinity"
    ... )

    Plot the feature and recurrence matrices

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> imgsim = librosa.display.specshow(
    ...     xsim, x_axis="s", y_axis="s", hop_length=hop_length, ax=ax[0]
    ... )
    >>> ax[0].set(title="Binary cross-similarity")
    >>> imgaff = librosa.display.specshow(
    ...     xsim_aff,
    ...     x_axis="s",
    ...     y_axis="s",
    ...     cmap="magma_r",
    ...     hop_length=hop_length,
    ...     ax=ax[1],
    ... )
    >>> ax[1].set(title="Cross-affinity")
    >>> ax[1].label_outer()
    >>> fig.colorbar(imgsim, ax=ax[0], orientation="horizontal", ticks=[0, 1])
    >>> fig.colorbar(imgaff, ax=ax[1], orientation="horizontal")
    """
    data_ref = np.atleast_2d(data_ref)
    data = np.atleast_2d(data)

    if not np.allclose(data_ref.shape[:-1], data.shape[:-1]):
        raise ParameterShapeError(
            f"data_ref.shape={data_ref.shape} and data.shape={data.shape} do not match on leading dimension(s)"
        )

    # Move time axis to front and flatten feature dimensions in C order
    data_ref = np.moveaxis(data_ref, -1, 0)
    n_ref = data_ref.shape[0]
    data_ref = data_ref.reshape((n_ref, -1))

    data = np.moveaxis(data, -1, 0)
    n = data.shape[0]
    data = data.reshape((n, -1))

    if mode not in ["connectivity", "distance", "affinity"]:
        raise ParameterValueError(f"Invalid mode='{mode}'. Must be one of ['connectivity', 'distance', 'affinity']")

    k_data_default = int(min(n_ref, 2 * np.ceil(np.sqrt(n_ref))))
    k_ref_default = int(min(n, 2 * np.ceil(np.sqrt(n))))
    k_data, k_ref = __parse_k(k, mutual=mutual, default_0=k_data_default, default_1=k_ref_default)

    # using k for bandwidth estimation also and decouple k for full mode
    bandwidth_k: tuple[int, int] = (k_data, k_ref)
    full_mode = full and (mode != "connectivity")
    if full_mode:
        k_data = n_ref
        k_ref = n

    # Build the neighbor search object (data -> data_ref)
    xsim_fwd = __cross_knn_graph(data_ref, data, k_data, metric, mode)

    xsim_rev: scipy.sparse.csr_array | None = None
    if mutual:
        xsim_rev = __cross_knn_graph(data, data_ref, k_ref, metric, mode)
        xsim = xsim_fwd.minimum(xsim_rev.T.tocsr())
        xsim.eliminate_zeros()
    else:
        xsim = xsim_fwd
        if mode == "affinity" and isinstance(bandwidth, str) and bandwidth in _PAIRWISE_BANDWIDTH_MODES:
            if full_mode:
                xsim_rev = xsim_fwd.T.tocsr()
            else:
                xsim_rev = __cross_knn_graph(data, data_ref, k_ref, metric, mode)

    if mode == "connectivity":
        xsim = xsim.astype(bool)
    elif mode == "affinity":
        aff_bandwidth = __affinity_bandwidth(xsim, bandwidth, bandwidth_k, rec_fwd=xsim_fwd, rec_rev=xsim_rev)
        xsim.data[:] = np.exp(xsim.data / (-1 * aff_bandwidth))

    # Transpose to n_ref by n and guarantee sparse *array* return type
    xsim_out = scipy.sparse.csc_array(xsim.T)

    if not sparse:
        return xsim_out.toarray()

    return xsim_out


@overload
def recurrence_matrix(
    data: np.ndarray,
    *,
    k: int | tuple[int | None, int | None] | None = ...,
    width: int = ...,
    metric: str = ...,
    mutual: bool = ...,
    sparse: Literal[True],
    mode: str = ...,
    bandwidth: np.ndarray | _FloatLike_co | str | None = ...,
    self: bool = ...,
    axis: int = ...,
    full: bool = False,
    sym: bool | Deprecated = ...,
) -> scipy.sparse.csc_array: ...
@overload
def recurrence_matrix(
    data: np.ndarray,
    *,
    k: int | tuple[int | None, int | None] | None = ...,
    width: int = ...,
    metric: str = ...,
    mutual: bool = ...,
    sparse: Literal[False] = ...,
    mode: str = ...,
    bandwidth: np.ndarray | _FloatLike_co | str | None = ...,
    self: bool = ...,
    axis: int = ...,
    full: bool = False,
    sym: bool | Deprecated = ...,
) -> np.ndarray: ...
@cache(level=30)
def recurrence_matrix(
    data: np.ndarray,
    *,
    k: int | tuple[int | None, int | None] | None = None,
    width: int = 1,
    metric: str = "euclidean",
    mutual: bool = False,
    sparse: bool = False,
    mode: str = "connectivity",
    bandwidth: np.ndarray | _FloatLike_co | str | None = None,
    self: bool = False,
    axis: int = -1,
    full: bool = False,
    sym: bool | Deprecated = Deprecated(),
) -> np.ndarray | scipy.sparse.csc_array:
    """Compute a recurrence matrix from a data matrix.

    ``rec[i, j]`` is non-zero if ``data[..., i]`` is a k-nearest neighbor
    of ``data[..., j]`` and ``|i - j| >= width``

    The specific value of ``rec[i, j]`` can have several forms, governed
    by the ``mode`` parameter below:

        - Connectivity: ``rec[i, j] = 1 or 0`` indicates that frames ``i`` and ``j`` are repetitions

        - Affinity: ``rec[i, j] > 0`` measures how similar frames ``i`` and ``j`` are.  This is also
          known as a (sparse) self-similarity matrix.

        - Distance: ``rec[i, j] > 0`` measures how distant frames ``i`` and ``j`` are.  This is also
          known as a (sparse) self-distance matrix.

    The general term *recurrence matrix* can refer to any of the three forms above.

    Parameters
    ----------
    data : np.ndarray [shape=(..., d, n)]
        A feature matrix.
        If the data has more than two dimensions (e.g., for multi-channel inputs),
        the leading dimensions are flattened prior to comparison.
        For example, a stereo input with shape `(2, d, n)` is
        automatically reshaped to `(2 * d, n)`.

    k : int > 0, tuple of (int > 0 or None, int > 0 or None), or None
        the number of nearest-neighbors for each sample.

        If a 2-tuple ``(k_col, k_row)`` is provided (only valid when
        ``mutual=True``), ``k_col`` specifies the number of nearest neighbors
        for each column and ``k_row`` specifies the number of nearest neighbors
        for each row. Either element may be ``None`` to use the default.

        Default: ``k = 2 * ceil(sqrt(t - 2 * width + 1))``,
        or ``k = 2`` if ``t <= 2 * width + 1``

    width : int >= 1 [scalar]
        only link neighbors ``(data[..., i], data[..., j])``
        if ``|i - j| >= width``

        ``width`` cannot exceed the length of the data.

    metric : str
        Distance metric to use for nearest-neighbor calculation.

        See `sklearn.neighbors.NearestNeighbors` for details.

    mutual : bool [scalar]
        If ``True``, only link mutual nearest neighbors: ``(i, j)`` is linked
        if ``data[..., i]`` is a nearest neighbor of ``data[..., j]`` and
        vice versa. Note that ``mutual=False`` does not guarantee symmetry even
        for scalar ``k``. When ``mutual=True`` and ``k`` is a scalar (or
        ``(k0, k0)``), the resulting link structure is symmetric; when
        ``k=(k_col, k_row)`` with ``k_col != k_row``, each column has at most
        ``k_col`` links and each row has at most ``k_row`` links.

    sparse : bool [scalar]
        if False, returns a dense type (ndarray)
        if True, returns a sparse type (scipy.sparse.csc_array)

    mode : str, {'connectivity', 'distance', 'affinity'}
        If 'connectivity', a binary connectivity matrix is produced.

        If 'distance', then a non-zero entry contains the distance between
        points.

        If 'affinity', then non-zero entries are mapped to
        ``exp( - distance(i, j) / bandwidth)`` where ``bandwidth`` is
        as specified below.

    bandwidth : None, float > 0, ndarray, or str
        str options include:
            - 'med_k_scalar'
            - 'mean_k'
            - 'gmean_k'
            - 'mean_k_avg'
            - 'gmean_k_avg'
            - 'mean_k_avg_and_pair'

        If ndarray [shape=(t, t)] is supplied, use ndarray as bandwidth for each (i, j) pair.

        If using ``mode='affinity'``, the ``bandwidth`` option can be used to set the
        bandwidth on the affinity kernel.

        If no value is provided or ``None``, default to ``'med_k_scalar'``.

        If ``bandwidth='med_k_scalar'``, a scalar bandwidth is set to the median distance
        of the k-th nearest neighbor for all samples.

        If ``bandwidth='mean_k'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        arithmetic mean between distances to the k-th nearest neighbor for sample i and sample j.

        If ``bandwidth='gmean_k'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        geometric mean between distances to the k-th nearest neighbor for sample i and j [#z]_.

        If ``bandwidth='mean_k_avg'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        arithmetic mean between the average distances to the first k-th nearest neighbors for
        sample i and sample j.
        This is similar to the approach in Wang et al. (2014) [#w]_ but does not include the distance
        between i and j.

        If ``bandwidth='gmean_k_avg'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        geometric mean between the average distances to the first k-th nearest neighbors for
        sample i and sample j.

        If ``bandwidth='mean_k_avg_and_pair'``, bandwidth is estimated for each sample-pair (i, j) by
        taking the arithmetic mean between three terms: the average distances to the first
        k-th nearest neighbors for sample i and sample j respectively, as well as
        the distance between i and j.
        This is similar to the approach in Wang et al. (2014). [#w]_

        .. [#z] Zelnik-Manor, Lihi, and Pietro Perona. (2004).
            "Self-tuning spectral clustering." Advances in neural information processing systems 17.

        .. [#w] Wang, Bo, et al. (2014).
            "Similarity network fusion for aggregating data types on a genomic scale." Nat Methods 11, 333--337.
            https://doi.org/10.1038/nmeth.2810

    self : bool
        If ``True``, then the main diagonal is populated with self-links:
        0 if ``mode='distance'``, and 1 otherwise.

        If ``False``, the main diagonal is left empty.

    axis : int
        The axis along which to compute recurrence.
        By default, the last index (-1) is taken.

    full : bool
        If using ``mode='affinity'`` or ``mode='distance'``, this option can be used to compute
        the full affinity or distance matrix as opposed to a sparse matrix with only non-zero terms
        for the first k-neighbors of each sample.
        This option has no effect when using ``mode='connectivity'``.

        When using ``mode='distance'``, setting ``full=True`` will ignore ``k`` and ``width``.
        When using ``mode='affinity'``, setting ``full=True`` will use ``k`` exclusively for
        bandwidth estimation, and ignore ``width``.

    sym : bool [scalar]
        Deprecated alias for ``mutual``.
        An exception is raised if both ``mutual=True`` and ``sym`` are provided.

        .. deprecated:: 1.1.0
            This parameter is deprecated and will be removed in version 2.0.0.
            Use ``mutual`` instead.

    Returns
    -------
    rec : np.ndarray or scipy.sparse.csc_array, [shape=(t, t)]
        Recurrence matrix

    See Also
    --------
    sklearn.neighbors.NearestNeighbors
    scipy.spatial.distance.cdist
    librosa.feature.stack_memory
    recurrence_to_lag

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Find nearest neighbors in CQT space

    >>> y, sr = librosa.loadx("nutcracker")
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> # Use time-delay embedding to get a cleaner recurrence matrix
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> R = librosa.segment.recurrence_matrix(chroma_stack)

    Or fix the number of nearest neighbors to 5

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, k=5)

    Suppress neighbors within +- 7 frames

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, width=7)

    Use cosine similarity instead of Euclidean distance

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, metric="cosine")

    Require mutual nearest neighbors

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, mutual=True)

    Use an affinity matrix instead of binary connectivity

    >>> R_aff = librosa.segment.recurrence_matrix(
    ...     chroma_stack, metric="cosine", mode="affinity"
    ... )

    Plot the feature and recurrence matrices

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> imgsim = librosa.display.specshow(
    ...     R, x_axis="s", y_axis="s", hop_length=hop_length, ax=ax[0]
    ... )
    >>> ax[0].set(title="Binary recurrence (symmetric)")
    >>> imgaff = librosa.display.specshow(
    ...     R_aff,
    ...     x_axis="s",
    ...     y_axis="s",
    ...     hop_length=hop_length,
    ...     cmap="magma_r",
    ...     ax=ax[1],
    ... )
    >>> ax[1].set(title="Affinity recurrence")
    >>> ax[1].label_outer()
    >>> fig.colorbar(imgsim, ax=ax[0], orientation="horizontal", ticks=[0, 1])
    >>> fig.colorbar(imgaff, ax=ax[1], orientation="horizontal")
    """
    if not isinstance(sym, Deprecated):
        if mutual:
            raise ParameterValueError("Both 'sym' and 'mutual' were provided. Please use only the 'mutual' parameter.")

        mutual = rename_kw(
            old_name="sym",
            old_value=sym,
            new_name="mutual",
            new_value=mutual,
            version_deprecated="1.1.0",
            version_removed="2.0.0",
        )

    data = np.atleast_2d(data)

    # Move observations to the first dimension and flatten the rest in C order
    data = np.moveaxis(data, axis, 0)
    t = data.shape[0]
    data = data.reshape((t, -1))

    if width < 1 or width >= (t - 1) // 2:
        raise ParameterValueError(
            f"width={width} must be at least 1 and at most (data.shape[{axis}] - 1) // 2={(t - 1) // 2}"
        )

    if mode not in ["connectivity", "distance", "affinity"]:
        raise ParameterValueError(f"Invalid mode='{mode}'. Must be one of ['connectivity', 'distance', 'affinity']")

    k_default = int(2 * np.ceil(np.sqrt(t - 2 * width + 1)))
    k_col, k_row = __parse_k(k, mutual=mutual, default_0=k_default, default_1=k_default)

    # using k for bandwidth estimation also and decouple k for full mode
    bw_k_col, bw_k_row = k_col, k_row
    full_mode = full and (mode != "connectivity")
    if full_mode:
        k_col = t - 1
        k_row = t - 1

    # Build the neighbor search object
    n_neighbors = t - 1 if full_mode else min(t - 1, max(k_col, k_row) + 2 * width)
    knn = __build_knn(data, n_neighbors, metric)
    dists, indices = knn.kneighbors(return_distance=True)

    eff_width = 0 if full_mode else width
    rec_col = __build_knn_graph(dists, indices, (t, t), k_col, mode, width=eff_width)
    rec_row: scipy.sparse.csr_array | None = None

    if mutual:
        if k_row == k_col:
            rec_row = rec_col
        else:
            rec_row = __build_knn_graph(dists, indices, (t, t), k_row, mode, width=eff_width)
        rec = rec_col.minimum(rec_row.T.tocsr())
        rec.eliminate_zeros()
    else:
        rec = rec_col

    if mode == "affinity":
        rec_rev_bw = rec_row if bw_k_row != bw_k_col else None
        aff_bandwidth = __affinity_bandwidth(rec, bandwidth, (bw_k_col, bw_k_row), rec_fwd=rec_col, rec_rev=rec_rev_bw)
        rec.data[:] = np.exp(rec.data / (-1 * aff_bandwidth))

    if self and mode != "distance":
        rec = (rec + scipy.sparse.eye_array(t, format="csr")).tocsr()

    if mode == "connectivity":
        rec = rec.astype(bool)

    # Transpose to be column-major
    rec_out = scipy.sparse.csc_array(rec.T)

    if not sparse:
        return rec_out.toarray()

    return rec_out


def recurrence_to_lag[ArrayOrSparseArray: np.ndarray | _SparseArray](
    rec: ArrayOrSparseArray, *, pad: bool = True, axis: int = -1
) -> ArrayOrSparseArray:
    """Convert a recurrence matrix into a lag matrix.

        ``lag[i, j] == rec[i+j, j]``

    This transformation turns diagonal structures in the recurrence matrix
    into horizontal structures in the lag matrix.
    These horizontal structures can be used to infer changes in the repetition
    structure of a piece, e.g., the beginning of a new section as done in [#]_.

    .. [#] Serra, J., Müller, M., Grosche, P., & Arcos, J. L. (2014).
           Unsupervised music structure annotation by time series structure
           features and segment similarity.
           IEEE Transactions on Multimedia, 16(5), 1229-1240.

    Parameters
    ----------
    rec : np.ndarray, or scipy.sparse array [shape=(n, n)]
        A (binary) recurrence matrix, as returned by `recurrence_matrix`

        .. deprecated:: 1.1.0
            Support for ``scipy.sparse.spmatrix`` inputs is deprecated and will
            be removed in version 2.0.0. Use ``scipy.sparse`` arrays instead.

    pad : bool
        If False, ``lag`` matrix is square, which is equivalent to
        assuming that the signal repeats itself indefinitely.

        If True, ``lag`` is padded with ``n`` zeros, which eliminates
        the assumption of repetition.

    axis : int
        The axis to keep as the ``time`` axis.
        The alternate axis will be converted to lag coordinates.

    Returns
    -------
    lag : np.ndarray
        The recurrence matrix in (lag, time) (if ``axis=1``)
        or (time, lag) (if ``axis=0``) coordinates

    Raises
    ------
    ParameterShapeError
        If ``rec`` is non-square

    See Also
    --------
    recurrence_matrix
    lag_to_recurrence
    util.shear

    Examples
    --------
    >>> y, sr = librosa.loadx("nutcracker")
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> recurrence = librosa.segment.recurrence_matrix(chroma_stack)
    >>> lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
    >>> lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(
    ...     lag_pad, x_axis="time", y_axis="lag", hop_length=hop_length, ax=ax[0]
    ... )
    >>> ax[0].set(title="Lag (zero-padded)")
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(
    ...     lag_nopad, x_axis="time", y_axis="lag", hop_length=hop_length, ax=ax[1]
    ... )
    >>> ax[1].set(title="Lag (no padding)")
    """
    axis = np.abs(axis)

    if rec.ndim != 2 or rec.shape[0] != rec.shape[1]:
        raise ParameterShapeError(f"non-square recurrence matrix shape: {rec.shape}")

    _warn_spmatrix(rec, stacklevel=3)

    sparse = scipy.sparse.issparse(rec)

    if sparse:
        # suppress type check here, mypy doesn't know about issparse
        fmt = rec.format  # type: ignore

    t = rec.shape[axis]

    if pad:
        if sparse:
            padding = np.asarray([[1, 0]], dtype=rec.dtype).swapaxes(axis, 0)
            rec_fmt: Literal["csr", "csc"]
            if axis == 0:
                rec_fmt = "csr"
            else:
                rec_fmt = "csc"
            rec = scipy.sparse.kron(padding, rec, format=rec_fmt)  # type: ignore
        else:
            padding = np.array([(0, 0), (0, 0)])
            padding[(1 - axis), :] = [0, t]
            # Suppress type check, mypy doesn't know that rec is an ndarray here
            rec = np.pad(rec, padding, mode="constant")  # type: ignore

    # Call private _shear to avoid emitting a duplicate spmatrix deprecation warning;
    # revert to util.shear once spmatrix support is removed in 2.0.0.
    lag: ArrayOrSparseArray = _shear(rec, factor=-1, axis=axis)  # type: ignore[arg-type, assignment]

    if sparse:
        # Suppress type check, mypy doesn't know
        # that lag is sparse here
        lag = lag.asformat(fmt)  # type: ignore

    return lag


def lag_to_recurrence[ArrayOrSparseArray: np.ndarray | _SparseArray](
    lag: ArrayOrSparseArray, *, axis: int = -1
) -> ArrayOrSparseArray:
    """Convert a lag matrix into a recurrence matrix.

    Parameters
    ----------
    lag : np.ndarray or scipy.sparse array
        A lag matrix, as produced by ``recurrence_to_lag``

        .. deprecated:: 1.1.0
            Support for ``scipy.sparse.spmatrix`` inputs is deprecated and will
            be removed in version 2.0.0. Use ``scipy.sparse`` arrays instead.
    axis : int
        The axis corresponding to the time dimension.
        The alternate axis will be interpreted in lag coordinates.

    Returns
    -------
    rec : np.ndarray or scipy.sparse array [shape=(n, n)]
        A recurrence matrix in (time, time) coordinates
        For sparse arrays, format will match that of ``lag``.

    Raises
    ------
    ParameterShapeError
        If ``lag`` does not have the correct shape

    See Also
    --------
    recurrence_to_lag

    Examples
    --------
    >>> y, sr = librosa.loadx("nutcracker")
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> recurrence = librosa.segment.recurrence_matrix(chroma_stack)
    >>> lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
    >>> lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)
    >>> rec_pad = librosa.segment.lag_to_recurrence(lag_pad)
    >>> rec_nopad = librosa.segment.lag_to_recurrence(lag_nopad)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    >>> librosa.display.specshow(
    ...     lag_pad, x_axis="s", y_axis="lag", hop_length=hop_length, ax=ax[0, 0]
    ... )
    >>> ax[0, 0].set(title="Lag (zero-padded)")
    >>> ax[0, 0].label_outer()
    >>> librosa.display.specshow(
    ...     lag_nopad, x_axis="s", y_axis="time", hop_length=hop_length, ax=ax[0, 1]
    ... )
    >>> ax[0, 1].set(title="Lag (no padding)")
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(
    ...     rec_pad, x_axis="s", y_axis="time", hop_length=hop_length, ax=ax[1, 0]
    ... )
    >>> ax[1, 0].set(title="Recurrence (with padding)")
    >>> librosa.display.specshow(
    ...     rec_nopad, x_axis="s", y_axis="time", hop_length=hop_length, ax=ax[1, 1]
    ... )
    >>> ax[1, 1].set(title="Recurrence (without padding)")
    >>> ax[1, 1].label_outer()
    """
    if axis not in [0, 1, -1]:
        raise ParameterValueError(f"Invalid target axis: {axis}")

    axis = np.abs(axis)

    if lag.ndim != 2 or (lag.shape[0] != lag.shape[1] and lag.shape[1 - axis] != 2 * lag.shape[axis]):
        raise ParameterShapeError(f"Invalid lag matrix shape: {lag.shape}")

    _warn_spmatrix(lag, stacklevel=3)

    # Since lag must be 2-dimensional, abs(axis) = axis
    t = lag.shape[axis]

    # Call private _shear to avoid emitting a duplicate spmatrix deprecation warning;
    # revert to util.shear once spmatrix support is removed in 2.0.0.
    rec = _shear(lag, factor=+1, axis=axis)  # type: ignore[arg-type]

    sub_slice = [slice(None)] * rec.ndim
    sub_slice[1 - axis] = slice(t)
    rec_slice: ArrayOrSparseArray = rec[tuple(sub_slice)]  # type: ignore[assignment]
    return rec_slice


def timelag_filter[F: Callable[..., Any]](function: F, pad: bool = True, index: int = 0) -> F:
    """Apply a filter in the time-lag domain.

    This is primarily useful for adapting image filters to operate on
    `recurrence_to_lag` output.

    Using `timelag_filter` is equivalent to the following sequence of
    operations:

    >>> data_tl = librosa.segment.recurrence_to_lag(data)
    >>> data_filtered_tl = function(data_tl)
    >>> data_filtered = librosa.segment.lag_to_recurrence(data_filtered_tl)

    Parameters
    ----------
    function : callable
        The filtering function to wrap, e.g., `scipy.ndimage.median_filter`
    pad : bool
        Whether to zero-pad the structure feature matrix
    index : int >= 0
        If ``function`` accepts input data as a positional argument, it should be
        indexed by ``index``

    Returns
    -------
    wrapped_function : callable
        A new filter function which applies in time-lag space rather than
        time-time space.

    Examples
    --------
    Apply a 31-bin median filter to the diagonal of a recurrence matrix.
    With default, parameters, this corresponds to a time window of about
    0.72 seconds.

    >>> y, sr = librosa.loadx("nutcracker", duration=30)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=3, delay=3)
    >>> rec = librosa.segment.recurrence_matrix(chroma_stack)
    >>> from scipy.ndimage import median_filter
    >>> diagonal_median = librosa.segment.timelag_filter(median_filter)
    >>> rec_filtered = diagonal_median(rec, size=(1, 31), mode="mirror")

    Or with affinity weights

    >>> rec_aff = librosa.segment.recurrence_matrix(chroma_stack, mode="affinity")
    >>> rec_aff_fil = diagonal_median(rec_aff, size=(1, 31), mode="mirror")

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(rec, y_axis="s", x_axis="s", ax=ax[0, 0])
    >>> ax[0, 0].set(title="Raw recurrence matrix")
    >>> ax[0, 0].label_outer()
    >>> librosa.display.specshow(rec_filtered, y_axis="s", x_axis="s", ax=ax[0, 1])
    >>> ax[0, 1].set(title="Filtered recurrence matrix")
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(
    ...     rec_aff, x_axis="s", y_axis="s", cmap="magma_r", ax=ax[1, 0]
    ... )
    >>> ax[1, 0].set(title="Raw affinity matrix")
    >>> librosa.display.specshow(
    ...     rec_aff_fil, x_axis="s", y_axis="s", cmap="magma_r", ax=ax[1, 1]
    ... )
    >>> ax[1, 1].set(title="Filtered affinity matrix")
    >>> ax[1, 1].label_outer()
    """

    def __my_filter(wrapped_f, *args, **kwargs):
        """Wrap the filter with lag conversions"""
        # Map the input data into time-lag space
        args = list(args)

        args[index] = recurrence_to_lag(args[index], pad=pad)

        # Apply the filtering function
        result = wrapped_f(*args, **kwargs)

        # Map back into time-time and return
        return lag_to_recurrence(result)

    return decorator(__my_filter, function)  # type: ignore


@cache(level=30)
def subsegment(data: np.ndarray, frames: np.ndarray, *, n_segments: int = 4, axis: int = -1) -> _Array1D[np.int_]:
    """Sub-divide a segmentation by feature clustering.

    Given a set of frame boundaries (``frames``), and a data matrix (``data``),
    each successive interval defined by ``frames`` is partitioned into
    ``n_segments`` by constrained agglomerative clustering.

    .. note::
        If an interval spans fewer than ``n_segments`` frames, then each
        frame becomes a sub-segment.

    Parameters
    ----------
    data : np.ndarray
        Data matrix to use in clustering
    frames : np.ndarray [shape=(n_boundaries,)], dtype=int, non-negative]
        Array of beat or segment boundaries, as provided by
        `librosa.beat.beat_track`,
        `librosa.onset.onset_detect`,
        or `agglomerative`.
    n_segments : int > 0
        Maximum number of frames to sub-divide each interval.
    axis : int
        Axis along which to apply the segmentation.
        By default, the last index (-1) is taken.

    Returns
    -------
    boundaries : np.ndarray [shape=(n_subboundaries,)]
        List of sub-divided segment boundaries

    See Also
    --------
    agglomerative : Temporal segmentation
    librosa.onset.onset_detect : Onset detection
    librosa.beat.beat_track : Beat tracking

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Load audio, detect beat frames, and subdivide in twos by CQT

    >>> y, sr = librosa.loadx("choice", duration=6.5)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    >>> cqt = np.abs(
    ...     librosa.cqt(y, sr=sr, bins_per_octave=36, n_bins=36 * 7, hop_length=512)
    ... )
    >>> subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
    >>> subseg_t = librosa.frames_to_time(subseg, sr=sr, hop_length=512)

    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.transforms as mpt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(
    ...     cqt,
    ...     vscale="dBFS",
    ...     bins_per_octave=36,
    ...     y_axis="cqt_hz",
    ...     x_axis="time",
    ...     ax=ax,
    ... )
    >>> hl = librosa.display.highlight(ax=ax, alpha=0.75, linewidth=2)
    >>> trans = mpt.blended_transform_factory(ax.transData, ax.transAxes)
    >>> ax.plot(
    ...     beat_times,
    ...     np.zeros_like(beat_times),
    ...     "^",
    ...     zorder=4,
    ...     markerfacecolor="C0",
    ...     color="C0",
    ...     linestyle="",
    ...     clip_on=False,
    ...     markersize=10,
    ...     label="Beats",
    ...     transform=trans,
    ...     path_effects=hl,
    ... )
    >>> ax.vlines(
    ...     beat_times,
    ...     0,
    ...     1,
    ...     color="C0",
    ...     linestyle="-",
    ...     transform=trans,
    ...     linewidth=3,
    ...     alpha=0.9,
    ...     zorder=1.5,
    ...     path_effects=hl,
    ... )
    >>> ax.plot(
    ...     subseg_t,
    ...     np.zeros_like(subseg_t),
    ...     "^",
    ...     zorder=3,
    ...     markerfacecolor="C2",
    ...     color="C2",
    ...     linestyle="",
    ...     clip_on=False,
    ...     markersize=6,
    ...     label="Sub-beats",
    ...     transform=trans,
    ...     path_effects=hl,
    ... )
    >>> ax.vlines(
    ...     subseg_t,
    ...     0,
    ...     1,
    ...     color="C2",
    ...     linestyle="--",
    ...     transform=trans,
    ...     linewidth=1,
    ...     alpha=0.9,
    ...     path_effects=hl,
    ... )
    >>> ax.legend(loc="upper right")
    >>> ax.set(title="CQT + Beat and sub-beat markers")
    """
    frames = util.fix_frames(frames, x_min=0, x_max=data.shape[axis], pad=True)

    if n_segments < 1:
        raise ParameterValueError("n_segments must be a positive integer")

    boundaries: list = []
    idx_slices = [slice(None)] * data.ndim

    for seg_start, seg_end in itertools.pairwise(frames):
        idx_slices[axis] = slice(seg_start, seg_end)
        boundaries.extend(
            seg_start
            + agglomerative(
                data[tuple(idx_slices)],
                cast("np.integer", min(seg_end - seg_start, n_segments)),
                axis=axis,
            )
        )

    return np.array(boundaries)


@cache(level=30)
def agglomerative(
    data: np.ndarray,
    k: int | np.integer,
    *,
    clusterer: sklearn.cluster.AgglomerativeClustering | None = None,
    axis: int = -1,
) -> _Array1D[np.int_]:
    """Bottom-up temporal segmentation.

    Use a temporally-constrained agglomerative clustering routine to partition
    ``data`` into ``k`` contiguous segments.

    Parameters
    ----------
    data : np.ndarray
        data to cluster
    k : int > 0 [scalar]
        number of segments to produce
    clusterer : sklearn.cluster.AgglomerativeClustering, optional
        An optional AgglomerativeClustering object.
        If `None`, a constrained Ward object is instantiated.
    axis : int
        axis along which to cluster.
        By default, the last axis (-1) is chosen.

    Returns
    -------
    boundaries : np.ndarray [shape=(k,)]
        left-boundaries (frame numbers) of detected segments. This
        will always include `0` as the first left-boundary.

    See Also
    --------
    sklearn.cluster.AgglomerativeClustering

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Cluster by chroma similarity, break into 20 segments

    >>> y, sr = librosa.loadx("nutcracker", duration=15)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, threshold=0.5)
    >>> bounds = librosa.segment.agglomerative(chroma, 20)
    >>> bound_times = librosa.frames_to_time(bounds, sr=sr)
    >>> bound_times
    array([ 0.   ,  0.65 ,  1.091,  1.927,  2.438,  2.902,  3.924,
            4.783,  5.294,  5.712,  6.13 ,  7.314,  8.522,  8.916,
            9.66 , 10.844, 11.238, 12.028, 12.492, 14.095])

    Plot the segmentation over the chromagram

    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.transforms as mpt
    >>> fig, ax = plt.subplots()
    >>> trans = mpt.blended_transform_factory(ax.transData, ax.transAxes)
    >>> librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", ax=ax)
    >>> hl = librosa.display.highlight(ax=ax)
    >>> ax.vlines(
    ...     bound_times,
    ...     0,
    ...     1,
    ...     linestyle="--",
    ...     path_effects=hl,
    ...     label="Segment boundaries",
    ...     transform=trans,
    ... )
    >>> ax.legend()
    >>> ax.set(title="Power spectrogram")
    """
    # Make sure we have at least two dimensions
    data = np.atleast_2d(data)

    # Move data index to position 0 and flatten features in C order
    data = np.moveaxis(data, axis, 0)
    n = data.shape[0]
    data = data.reshape((n, -1))

    if clusterer is None:
        import sklearn.cluster
        import sklearn.feature_extraction

        # Connect the temporal connectivity graph
        grid = sklearn.feature_extraction.image.grid_to_graph(n_x=n, n_y=1, n_z=1)

        # Instantiate the clustering object
        clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=k, connectivity=grid, memory=cache.memory)

    # Fit the model
    clusterer.fit(data)

    # Find the change points from the labels
    boundaries = [0]
    boundaries.extend(list(1 + np.nonzero(np.diff(clusterer.labels_))[0].astype(int)))
    return np.asarray(boundaries)


@cache(level=30)
def path_enhance(
    R: np.ndarray,
    n: int,
    *,
    window: _WindowSpec = "hann",
    max_ratio: float = 2.0,
    min_ratio: float | None = None,
    n_filters: int = 7,
    zero_mean: bool = False,
    clip: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Multi-angle path enhancement for self- and cross-similarity matrices.

    This function convolves multiple diagonal smoothing filters with a self-similarity (or
    recurrence) matrix R, and aggregates the result by an element-wise maximum.

    Technically, the output is a matrix R_smooth such that::

        R_smooth[i, j] = max_theta(R * filter_theta)[i, j]

    where `*` denotes 2-dimensional convolution, and ``filter_theta`` is a smoothing filter at
    orientation theta.

    This is intended to provide coherent temporal smoothing of self-similarity matrices
    when there are changes in tempo.

    Smoothing filters are generated at evenly spaced orientations between min_ratio and
    max_ratio.

    This function is inspired by the multi-angle path enhancement of [#]_, but differs by
    modeling tempo differences in the space of similarity matrices rather than re-sampling
    the underlying features prior to generating the self-similarity matrix.

    .. [#] Müller, Meinard and Frank Kurth.
            "Enhancing similarity matrices for music audio analysis."
            2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings.
            Vol. 5. IEEE, 2006.

    .. note:: if using recurrence_matrix to construct the input similarity matrix, be sure to include the main
              diagonal by setting ``self=True``.  Otherwise, the diagonal will be suppressed, and this is likely to
              produce discontinuities which will pollute the smoothing filter response.

    Parameters
    ----------
    R : np.ndarray
        The self- or cross-similarity matrix to be smoothed.
        Note: sparse inputs are not supported.

        If the recurrence matrix is multi-dimensional, e.g. `shape=(c, n, n)`,
        then enhancement is conducted independently for each leading channel.

    n : int > 0
        The length of the smoothing filter

    window : window specification
        The type of smoothing filter to use.  See `filters.get_window` for more information
        on window specification formats.

    max_ratio : float > 0
        The maximum tempo ratio to support

    min_ratio : float > 0
        The minimum tempo ratio to support.
        If not provided, it will default to ``1/max_ratio``

    n_filters : int >= 1
        The number of different smoothing filters to use, evenly spaced
        between ``min_ratio`` and ``max_ratio``.

        If ``min_ratio = 1/max_ratio`` (the default), using an odd number
        of filters will ensure that the main diagonal (ratio=1) is included.

    zero_mean : bool
        By default, the smoothing filters are non-negative and sum to one (i.e. are averaging
        filters).

        If ``zero_mean=True``, then the smoothing filters are made to sum to zero by subtracting
        a constant value from the non-diagonal coordinates of the filter.  This is primarily
        useful for suppressing blocks while enhancing diagonals.

    clip : bool
        If True, the smoothed similarity matrix will be thresholded at 0, and will not contain
        negative entries.

    **kwargs : additional keyword arguments
        Additional arguments to pass to `scipy.ndimage.convolve`

    Returns
    -------
    R_smooth : np.ndarray, shape=R.shape
        The smoothed self- or cross-similarity matrix

    See Also
    --------
    librosa.filters.diagonal_filter
    recurrence_matrix

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Use a 51-frame diagonal smoothing filter to enhance paths in a recurrence matrix

    >>> y, sr = librosa.loadx("nutcracker")
    >>> hop_length = 2048
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> rec = librosa.segment.recurrence_matrix(
    ...     chroma_stack, mode="affinity", self=True
    ... )
    >>> rec_smooth = librosa.segment.path_enhance(rec, 51, window="hann", n_filters=7)

    Plot the recurrence matrix before and after smoothing

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(
    ...     rec, x_axis="s", y_axis="s", hop_length=hop_length, ax=ax[0]
    ... )
    >>> ax[0].set(title="Unfiltered recurrence")
    >>> imgpe = librosa.display.specshow(
    ...     rec_smooth, x_axis="s", y_axis="s", hop_length=hop_length, ax=ax[1]
    ... )
    >>> ax[1].set(title="Multi-angle enhanced recurrence")
    >>> ax[1].label_outer()
    >>> fig.colorbar(img, ax=ax[0], orientation="horizontal")
    >>> fig.colorbar(imgpe, ax=ax[1], orientation="horizontal")
    """
    if min_ratio is None:
        min_ratio = 1.0 / max_ratio
    elif min_ratio > max_ratio:
        raise ParameterValueError(f"min_ratio={min_ratio} cannot exceed max_ratio={max_ratio}")

    R_smooth = None
    for ratio in np.logspace(np.log2(min_ratio), np.log2(max_ratio), num=n_filters, base=2):
        kernel = diagonal_filter(window, n, slope=ratio, zero_mean=zero_mean)

        # Expand leading dimensions to match R
        # This way, if R has shape, eg, [2, 3, n, n]
        # the expanded kernel will have shape [1, 1, m, m]

        kernel = np.expand_dims(kernel, axis=tuple(range(R.ndim - kernel.ndim)))

        import scipy.ndimage

        if R_smooth is None:
            R_smooth = scipy.ndimage.convolve(R, kernel, **kwargs)
        else:
            # Compute the point-wise maximum in-place
            np.maximum(R_smooth, scipy.ndimage.convolve(R, kernel, **kwargs), out=R_smooth)

    if clip:
        # Clip the output in-place
        np.clip(R_smooth, 0, None, out=R_smooth)  # type: ignore

    return np.asanyarray(R_smooth)


def __parse_k(
    k: int | tuple[int | None, int | None] | None,
    *,
    mutual: bool,
    default_0: int,
    default_1: int,
) -> tuple[int, int]:
    """Validate and resolve neighborhood size ``k`` to ``(k_0, k_1)``."""
    if k is None:
        return default_0, default_1

    if isinstance(k, tuple):
        if not mutual:
            raise ParameterValueError("Tuple k is only supported when mutual=True")
        if len(k) != 2:
            raise ParameterValueError(
                f"k must be a positive integer or a 2-tuple of positive integers, got tuple of length {len(k)}"
            )
        resolved: list[int] = []
        for idx, (val, default) in enumerate(zip(k, (default_0, default_1), strict=True)):
            if val is None:
                resolved.append(default)
            elif util.is_positive_int(val) and not isinstance(val, bool):
                resolved.append(int(val))
            else:
                raise ParameterValueError(f"k[{idx}]={val!r} must be a positive integer or None")
        return resolved[0], resolved[1]

    if util.is_positive_int(k) and not isinstance(k, bool):
        k_int = int(k)
        return k_int, k_int

    raise ParameterValueError(f"k={k!r} must be a positive integer, a 2-tuple of positive integers, or None")


def __build_knn(fit_data: np.ndarray, n_neighbors: int, metric: str) -> sklearn.neighbors.NearestNeighbors:
    """Fit a nearest-neighbors index."""
    import sklearn.neighbors

    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
    knn.fit(fit_data)
    return knn


def __build_knn_graph(
    dists: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
    k_limit: int,
    mode: str,
    width: int = 0,
) -> scipy.sparse.csr_array:
    """Build a CSR k-NN graph from ``NearestNeighbors.kneighbors`` outputs."""
    n_rows = shape[0]
    if width > 1:
        row_ids = np.arange(n_rows)[:, None]
        valid = np.abs(indices - row_ids) >= width
    else:
        valid = np.ones_like(indices, dtype=bool)

    rank = np.cumsum(valid, axis=1)
    keep = valid & (rank <= k_limit)

    rows, cols_pos = np.nonzero(keep)
    cols = indices[rows, cols_pos]
    vals = np.ones(len(rows), dtype=float) if mode == "connectivity" else dists[rows, cols_pos]

    graph = scipy.sparse.csr_array((vals, (rows, cols)), shape=shape)
    graph.eliminate_zeros()
    return graph


def __cross_knn_graph(
    fit_data: np.ndarray,
    query_data: np.ndarray,
    k_limit: int,
    metric: str,
    mode: str,
) -> scipy.sparse.csr_array:
    """Build a CSR k-NN graph from ``query_data`` into ``fit_data``."""
    n_fit = fit_data.shape[0]
    n_query = query_data.shape[0]
    knn = __build_knn(fit_data, min(n_fit, k_limit), metric)
    dists, indices = knn.kneighbors(X=query_data, return_distance=True)
    return __build_knn_graph(dists, indices, (n_query, n_fit), k_limit, mode)


def __knn_row_stats(rec: scipy.sparse.csr_array, k: int, bw_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row k-th nearest-neighbor distance and average distance up to k."""
    n_rows = rec.shape[0]
    knn_dists: list[np.ndarray] = []
    for i in range(n_rows):
        start, end = rec.indptr[i], rec.indptr[i + 1]
        row_data = rec.data[start:end]
        if row_data.size == 0:
            # Disconnected vertices are only a problem for point-wise bandwidth estimation
            if bw_mode != "med_k_scalar":
                raise ParameterValueError(f"The sample at time point {i} has no neighbors")
            knn_dists.append(np.array([np.nan]))
        else:
            knn_dists.append(np.sort(row_data)[:k])

    dist_to_k = np.asarray([dists[-1] for dists in knn_dists])
    avg_dist_to_first_ks = np.asarray([np.mean(dists) for dists in knn_dists])
    return dist_to_k, avg_dist_to_first_ks


_PAIRWISE_BANDWIDTH_MODES = (
    "mean_k",
    "gmean_k",
    "mean_k_avg",
    "gmean_k_avg",
    "mean_k_avg_and_pair",
)


def __affinity_bandwidth(
    rec: scipy.sparse.csr_array,
    bw_mode: np.ndarray | _FloatLike_co | str | None,
    k: int | tuple[int, int],
    *,
    rec_fwd: scipy.sparse.csr_array | None = None,
    rec_rev: scipy.sparse.csr_array | None = None,
) -> float | np.ndarray:
    # rec should be a csr_array

    # the api allows users to specify a scalar bandwidth directly, besides the string based options.
    if isinstance(bw_mode, np.ndarray):
        bandwidth = bw_mode
        # rec is transposed prior to returning, so bandwidth matches output shape (rec.shape[1], rec.shape[0])
        expected_shape = (rec.shape[1], rec.shape[0])
        if bandwidth.shape != expected_shape:
            raise ParameterShapeError(
                f"Invalid matrix bandwidth shape: {bandwidth.shape}. Should be {expected_shape}."
            )
        if not (np.issubdtype(bandwidth.dtype, np.floating) or np.issubdtype(bandwidth.dtype, np.integer)):
            raise ParameterValueError(f"Invalid bandwidth dtype={bandwidth.dtype}. Must be real-valued numeric.")
        if not np.all(bandwidth > 0):
            raise ParameterValueError("Invalid bandwidth. All entries must be strictly positive.")
        rows = np.repeat(np.arange(rec.shape[0]), np.diff(rec.indptr))
        cols = rec.indices
        return np.array(bandwidth[cols, rows])

    elif isinstance(bw_mode, (int, float, np.integer, np.floating)) and not isinstance(bw_mode, bool):
        scalar_bandwidth = float(bw_mode)
        if not (scalar_bandwidth > 0):
            raise ParameterValueError(f"Invalid scalar bandwidth={scalar_bandwidth}. Must be strictly positive.")
        return scalar_bandwidth

    if bw_mode is None:
        bw_mode = "med_k_scalar"

    if not isinstance(bw_mode, str) or bw_mode not in ("med_k_scalar", *_PAIRWISE_BANDWIDTH_MODES):
        raise ParameterValueError(
            f"Invalid bandwidth='{bw_mode}'. Must be either a positive scalar or one of "
            "['med_k_scalar', 'mean_k', 'gmean_k', 'mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair']"
        )

    k_0, k_1 = k if isinstance(k, tuple) else (k, k)
    src_fwd = rec if rec_fwd is None else rec_fwd
    row_dist_to_k, row_avg_dist = __knn_row_stats(src_fwd, k_0, bw_mode)

    if bw_mode == "med_k_scalar":
        if not np.any(np.isfinite(row_dist_to_k)):
            raise ParameterValueError("Cannot estimate bandwidth from an empty graph")
        return float(np.nanmedian(row_dist_to_k))

    if rec_rev is not None:
        col_dist_to_k, col_avg_dist = __knn_row_stats(rec_rev, k_1, bw_mode)
    else:
        col_dist_to_k, col_avg_dist = row_dist_to_k, row_avg_dist

    t = rec.shape[0]
    sigma_i_data = np.empty_like(rec.data)
    sigma_j_data = np.empty_like(rec.data)

    if bw_mode in ["mean_k", "gmean_k"]:
        # building bandwidth components (sigma) using sparse matrix structures and indices
        for row in range(t):
            sigma_i_data[rec.indptr[row] : rec.indptr[row + 1]] = row_dist_to_k[row]
            col_idx = rec.indices[rec.indptr[row] : rec.indptr[row + 1]]
            sigma_j_data[rec.indptr[row] : rec.indptr[row + 1]] = col_dist_to_k[col_idx]

        if bw_mode == "mean_k":
            out = np.array((sigma_i_data + sigma_j_data) / 2)
        elif bw_mode == "gmean_k":
            out = np.array((sigma_i_data * sigma_j_data) ** 0.5)

    if bw_mode in ["mean_k_avg", "gmean_k_avg", "mean_k_avg_and_pair"]:
        # building bandwidth components (sigma) using sparse matrix structures and indices
        for row in range(t):
            sigma_i_data[rec.indptr[row] : rec.indptr[row + 1]] = row_avg_dist[row]
            col_idx = rec.indices[rec.indptr[row] : rec.indptr[row + 1]]
            sigma_j_data[rec.indptr[row] : rec.indptr[row + 1]] = col_avg_dist[col_idx]

        if bw_mode == "mean_k_avg":
            out = np.array((sigma_i_data + sigma_j_data) / 2)
        elif bw_mode == "gmean_k_avg":
            out = np.array((sigma_i_data * sigma_j_data) ** 0.5)
        elif bw_mode == "mean_k_avg_and_pair":
            out = np.array((sigma_i_data + sigma_j_data + rec.data) / 3)

    return out


def diagonal_convolve(
    data: np.ndarray,
    kernel: np.ndarray | tuple[np.ndarray, np.ndarray],
    *,
    k: int = 0,
    mode: Literal["full", "valid", "same"] = "same",
    boundary: str = "fill",
    fillvalue: float = 0.0,
) -> np.ndarray:
    """Efficient 2D convolution along a diagonal.

    Computes the 2-dimensional convolution of `data` and `kernel` evaluated
    only along the diagonal specified by `k`, without allocating or calculating
    the full 2D convolution output [#]_.

    Parameters
    ----------
    data : np.ndarray [shape=(..., n_rows, n_cols)]
        Input 2D matrix (or multi-channel batch of matrices), such as a recurrence
        or self-similarity matrix.
        Sparse matrices are not supported.
    kernel : np.ndarray [shape=(m, m)] or tuple of np.ndarray [shape=((m1,), (m2,))]
        Filter kernel. Can be:

        - A 2D array of shape ``(m1, m2)``
        - A 1D array ``v`` of shape ``(m,)`` representing a symmetric separable kernel ``v @ v.T``
        - A tuple of two 1D arrays ``(u, v)`` representing a separable kernel ``u @ v.T``

    k : int, default=0
        Diagonal offset. ``k=0`` is the main diagonal. ``k > 0`` is above the main
        diagonal, ``k < 0`` is below. Matches `numpy.diagonal(..., offset=k)`.
    mode : {'same', 'valid', 'full'}, default='same'
        Convolution mode, matching `scipy.signal.convolve2d`:

        - ``'same'``: Output has the same length as the diagonal of `data`, centered.
        - ``'valid'``: Output contains only positions where `kernel` completely overlaps `data`.
        - ``'full'``: Output contains all positions where `kernel` overlaps `data` by at least one element.

    boundary : str, default='fill'
        Boundary condition matching `scipy.signal.convolve2d`:

        - ``'fill'``: pad with `fillvalue` (default 0.0).
        - ``'symm'``: symmetric reflection padding.
        - ``'wrap'``: circular padding.
        - Standard `numpy.pad` mode strings (e.g. ``'constant'``, ``'edge'``, ``'reflect'``) are also accepted.

    fillvalue : float, default=0.0
        Value to fill past boundaries when `boundary='fill'` or `'constant'`.

    Returns
    -------
    diag_conv : np.ndarray [shape=(..., n_diag)]
        The 2D convolution evaluated along diagonal `k`.

    Raises
    ------
    ParameterTypeError
        If `data` is sparse.
    ParameterShapeError
        If `kernel` shape is invalid.
    ParameterValueError
        If `mode` is unsupported.

    See Also
    --------
    recurrence_novelty
    librosa.filters.checkerboard_kernel

    References
    ----------
    .. [#] Foote, Jonathan.
        "Automatic audio segmentation using a measure of audio novelty."
        In 2000 IEEE International Conference on Multimedia and Expo (ICME),
        vol. 1, pp. 452-455. IEEE, 2000.

    Examples
    --------
    Compute the main diagonal of a 2D convolution with a 17x17 checkerboard kernel
    on a short excerpt:

    >>> y, sr = librosa.loadx("nutcracker", duration=30)
    >>> hop_length = 512
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> rec = librosa.segment.recurrence_matrix(chroma, mode="affinity", self=True)
    >>> kernel = librosa.filters.checkerboard_kernel(17)
    >>> nov = librosa.segment.diagonal_convolve(rec, kernel)

    Plot the recurrence matrix and the resulting novelty trace with proportional heights:

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(nov, sr=sr, hop_length=hop_length)
    >>> fig, ax = plt.subplots(
    ...     nrows=2, sharex=True, height_ratios=[3, 1], layout="compressed"
    ... )
    >>> librosa.display.specshow(
    ...     rec, x_axis="time", y_axis="time", sr=sr, hop_length=hop_length, ax=ax[0]
    ... )
    >>> ax[0].set(title="Recurrence Matrix")
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, nov, label="Novelty")
    >>> ax[1].set(title="Diagonal 2D Convolution (Novelty)", xlabel="Time (s)")
    >>> ax[1].legend()
    """
    if scipy.sparse.issparse(data):
        raise ParameterTypeError(
            "Sparse matrices are not supported by diagonal_convolve. "
            "Convert to a dense array (e.g. data.toarray()) first."
        )

    data = np.asanyarray(data)

    is_separable = False
    u: np.ndarray
    v: np.ndarray
    K: np.ndarray
    M1: int
    M2: int

    if isinstance(kernel, tuple) and len(kernel) == 2:
        u = np.asanyarray(kernel[0])
        v = np.asanyarray(kernel[1])
        if u.ndim == 1 and v.ndim == 1:
            is_separable = True
            M1, M2 = len(u), len(v)
        else:
            raise ParameterShapeError("Tuple kernel must contain two 1D arrays")
    elif isinstance(kernel, np.ndarray) and kernel.ndim == 1:
        u = v = kernel
        is_separable = True
        M1, M2 = len(u), len(v)
    else:
        K = np.asanyarray(kernel)
        if K.ndim != 2:
            raise ParameterShapeError(f"kernel must be 2D, 1D, or a tuple of two 1D arrays; shape={K.shape}")
        M1, M2 = K.shape

    if M1 <= 0 or M2 <= 0:
        raise ParameterShapeError(f"Kernel dimensions must be positive, got ({M1}, {M2})")

    N1, N2 = data.shape[-2], data.shape[-1]

    if mode == "full":
        shift1, shift2 = 0, 0
        grid_rows, grid_cols = N1 + M1 - 1, N2 + M2 - 1
    elif mode == "same":
        shift1, shift2 = (M1 - 1) // 2, (M2 - 1) // 2
        grid_rows, grid_cols = N1, N2
    elif mode == "valid":
        shift1, shift2 = M1 - 1, M2 - 1
        grid_rows, grid_cols = max(0, N1 - M1 + 1), max(0, N2 - M2 + 1)
    else:
        raise ParameterValueError(f"Unsupported mode='{mode}'. Must be one of ('same', 'valid', 'full')")

    if grid_rows <= 0 or grid_cols <= 0:
        return np.empty((*data.shape[:-2], 0), dtype=data.dtype)

    if k >= 0:
        length = max(0, min(grid_rows, grid_cols - k))
        row_start_mode = 0
        col_start_mode = k
    else:
        length = max(0, min(grid_rows + k, grid_cols))
        row_start_mode = -k
        col_start_mode = 0

    if length == 0:
        return np.empty((*data.shape[:-2], 0), dtype=data.dtype)

    boundary_map = {
        "fill": "constant",
        "symm": "symmetric",
        "wrap": "wrap",
    }
    np_pad_mode: Any = boundary_map.get(boundary, boundary)

    pad_width = [(0, 0)] * (data.ndim - 2) + [(M1 - 1, M1 - 1), (M2 - 1, M2 - 1)]

    data_pad: np.ndarray
    if np_pad_mode == "constant":
        data_pad = np.pad(data, pad_width, mode=np_pad_mode, constant_values=fillvalue)
    else:
        data_pad = np.pad(data, pad_width, mode=np_pad_mode)

    r_start = shift1 + row_start_mode
    c_start = shift2 + col_start_mode

    leading_shape = data.shape[:-2]
    leading_strides = data_pad.strides[:-2]
    s0, s1 = data_pad.strides[-2], data_pad.strides[-1]

    sub = data_pad[..., r_start : r_start + length + M1 - 1, c_start : c_start + length + M2 - 1]
    out_shape = (*leading_shape, length, M1, M2)
    out_strides = (*leading_strides, s0 + s1, s0, s1)
    patches = np.lib.stride_tricks.as_strided(sub, shape=out_shape, strides=out_strides)

    if is_separable:
        return np.asanyarray((patches @ v[::-1]) @ u[::-1])
    else:
        return np.asanyarray(np.tensordot(patches, K[::-1, ::-1], axes=([-2, -1], [0, 1])))


@cache(level=30)
def recurrence_novelty(
    rec: np.ndarray,
    *,
    n: int = 33,
    window: _WindowSpec | None = None,
    var: float = 0.5,
    norm: bool | float | None = 1,
    kernel: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    clip: bool = True,
    exclude: bool = False,
    mode: Literal["full", "valid", "same"] = "same",
    boundary: str = "fill",
    fillvalue: float = 0.0,
) -> np.ndarray:
    """Compute Foote's novelty curve from a recurrence or self-similarity matrix.

    Slides a checkerboard kernel along the main diagonal of `rec` to detect
    structural transitions [#]_. Homogeneous musical sections appear as high-similarity
    blocks along the diagonal of `rec`. When two contrasting sections meet, the
    kernel aligns with the block boundary, producing a local peak in the novelty curve.

    Parameters
    ----------
    rec : np.ndarray [shape=(..., n_frames, n_frames)]
        Self-similarity or recurrence matrix.
        Note: Sparse matrices are not supported.
    n : int > 0, default=33
        Length of the checkerboard kernel if `kernel` is not provided.
    window : _WindowSpec or None, default=None
        Window type for the checkerboard kernel. Defaults to ``None`` (unwindowed).
        See `librosa.filters.checkerboard_kernel` for details.
    var : float > 0, default=0.5
        Gaussian variance parameter for the checkerboard kernel when ``window='gaussian'``.
    norm : bool, float, or None, default=1
        Normalization for the checkerboard kernel.
    kernel : np.ndarray or tuple, optional
        Pre-computed filter kernel. If provided, `n`, `window`, `var`, `norm` are ignored.
    clip : bool, default=True
        If True, half-wave rectifies the output (negative novelty scores clipped to 0).
    exclude : bool, default=False
        If True and ``mode='same'``, zeroes out the first and last ``n // 2`` values to
        suppress boundary artifacts (as in libfmp [#]_).
    mode : {'same', 'valid', 'full'}, default='same'
        Convolution mode passed to `diagonal_convolve`.
    boundary : str, default='fill'
        Boundary padding mode passed to `diagonal_convolve`.
    fillvalue : float, default=0.0
        Fill value for boundary padding.

    Returns
    -------
    novelty : np.ndarray [shape=(..., n_out)]
        1D novelty curve.

    See Also
    --------
    diagonal_convolve
    librosa.filters.checkerboard_kernel
    librosa.segment.recurrence_matrix

    Notes
    -----
    This function caches at level 30.

    References
    ----------
    .. [#] Foote, Jonathan.
        "Automatic audio segmentation using a measure of audio novelty."
        In 2000 IEEE International Conference on Multimedia and Expo (ICME),
        vol. 1, pp. 452-455. IEEE, 2000.
    .. [#] Müller, Meinard.
        "Fundamentals of music processing: Audio, analysis, algorithms, applications."
        Springer, 2015.

    Examples
    --------
    Compute and plot Foote's novelty curve on a short excerpt using a 17-frame boxcar
    checkerboard kernel:

    >>> y, sr = librosa.loadx("nutcracker", duration=30)
    >>> hop_length = 512
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> rec = librosa.segment.recurrence_matrix(
    ...     chroma_stack, mode="affinity", self=True
    ... )
    >>> nov = librosa.segment.recurrence_novelty(rec, n=17)

    Compare unwindowed (boxcar) and Gaussian-tapered novelty curves:

    >>> nov_gauss = librosa.segment.recurrence_novelty(
    ...     rec, n=17, window="gaussian", var=0.5
    ... )

    Plot the recurrence matrix and both novelty curves with proportional heights:

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(nov, sr=sr, hop_length=hop_length)
    >>> fig, ax = plt.subplots(
    ...     nrows=2, sharex=True, height_ratios=[3, 1], layout="compressed"
    ... )
    >>> librosa.display.specshow(
    ...     rec, x_axis="time", y_axis="time", sr=sr, hop_length=hop_length, ax=ax[0]
    ... )
    >>> ax[0].set(title="Recurrence Matrix")
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, nov, label="Boxcar (default)")
    >>> ax[1].plot(times, nov_gauss, label="Gaussian (var=0.5)", linestyle="--")
    >>> ax[1].set(title="Foote Novelty Curve", xlabel="Time (s)")
    >>> ax[1].legend()
    """
    if kernel is None:
        kernel = checkerboard_kernel(n, window=window, var=var, norm=norm)

    nov = diagonal_convolve(rec, kernel, k=0, mode=mode, boundary=boundary, fillvalue=fillvalue)

    if clip:
        np.clip(nov, 0, None, out=nov)

    if exclude and mode == "same":
        k_len: int
        if isinstance(kernel, tuple) and len(kernel) == 2:
            k_len = max(len(kernel[0]), len(kernel[1]))
        elif isinstance(kernel, np.ndarray) and kernel.ndim == 1:
            k_len = len(kernel)
        elif isinstance(kernel, np.ndarray):
            k_len = max(kernel.shape[-2:])
        else:
            k_len = n

        L = k_len // 2
        if L > 0:
            nov[..., :L] = 0
            nov[..., -L:] = 0

    return nov
