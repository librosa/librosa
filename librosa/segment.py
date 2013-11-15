#!/usr/bin/env python
"""Temporal segmentation"""

import numpy as np
import scipy
import scipy.signal

import sklearn
import sklearn.cluster
import sklearn.feature_extraction

def stack_memory(X, m=2, delay=1):
    """Short-term history embedding.

    Each column ``X[:, i]`` is mapped to

    ``X[:,i] ->  [   X[:, i].T, X[:, i - delay].T ...  X[:, i - (m-1)*delay].T ].T``


    :parameters:
      - X : np.ndarray
          feature matrix (d-by-t)
      - m : int > 0
          embedding dimension
      - delay : int > 0
          the number of columns to step

    :returns:
      - Xhat : np.ndarray, shape=(d*m, t)
          X augmented with lagged copies of itself.
          
      .. note:: zeros are padded for the initial columns

    """

    d, t = X.shape

    # Pad the end with zeros, which will roll to the front below
    X = np.hstack([X, np.zeros((d, m * delay))])

    Xhat = X

    for i in range(1, m):
        Xhat = np.vstack([Xhat, np.roll(X, i * delay, axis=1)])

    return Xhat[:, :t]

def recurrence_matrix(X, k=5, width=1, metric='sqeuclidean', sym=True):
    '''Compute the binary recurrence matrix from a time-series.

    R[i,j] == True <=> (X[:,i], X[:,j]) are k-nearest-neighbors and ||i-j|| >= width

    :parameters:
      - X : np.ndarray
          feature matrix (d-by-t)
      - k : int > 0, float in (0, 1)
          if integer, the number of nearest-neighbors.
          if floating point (eg, 0.05), neighbors = ceil(k * t)
      - width : int > 0
          no not link columns within `width` of each-other
      - metric : see scipy.spatial.distance.pdist()
          distance metric to use for nearest-neighbor calculation
      - sym : bool
          if using mode='knn', should we symmetrize the linkages?

    :returns:
      - R : np.ndarray, shape=(t,t), dtype=bool
          Binary recurrence matrix
    :raises:
      - ValueError
          if k is a float outside the range (0,1)
          or if mode is not one of {'knn', 'gaussian'}
    '''

    t = X.shape[1]

    if isinstance(k, float):
        if 0 < k < 1:
            k = np.ceil(k * t)
        else:
            raise ValueError('Valid values of k are strictly between 0 and 1.')

    def _band_infinite():
        '''Suppress the diagonal+- of a distance matrix'''
        A       = np.empty( (t, t) )
        A[:]    = np.inf
        A[np.triu_indices_from(A, width)] = 0
        A[np.tril_indices_from(A, -width)] = 0

        return A

    # Build the distance matrix
    D = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(X.T, metric=metric))

    # Max out the diagonal band
    D = D + _band_infinite()

    # build the recurrence plot

    R = np.zeros( (t, t), dtype=bool)

    # get the k nearest neighbors for each point
    for i in range(t):
        for j in np.argsort(D[i])[:k]:
            R[i, j] = True

    # symmetrize
    if sym:
        R = R * R.T

    return R 

def structure_feature(R, pad=True):
    '''Compute the structure feature from a recurrence matrix.

    The i'th column of the recurrence matrix is shifted up by i.
    The resulting matrix is indexed horizontally by time,
    and vertically by lag.

    :parameters:
      - R   : np.ndarray, shape=(t,t)
          recurrence matrix (see `librosa.segment.recurrence_matrix`)
      
      - pad : boolean

    :returns:
      - L : np.ndarray
          ``L[i, t]`` = the recurrence at time ``t`` with lag ``i``.

      .. note:: negative lag values are supported by wrapping to the end of the array.

    :raises:
      - ValueError
          if R is not square
    '''

    t = R.shape[0]
    if t != R.shape[1]:
        raise ValueError('R must be a square matrix')

    if pad:
        L = np.vstack( ( R, np.zeros_like(R) ) )
    else:
        L = R.copy()

    for i in range(1, t):
        L[:, i] = np.roll(L[:, i], -i, axis=-1)

    return L

def agglomerative(data, k):
    """Bottom-up temporal segmentation

    :parameters:
      - data     : np.ndarray    
          feature matrix (d-by-t)

      - k        : int > 0
          number of segments to produce

    :returns:
      - boundaries : np.ndarray, shape=(k,1)  
          left-boundaries (frame numbers) of detected segments

    """

    # Connect the temporal connectivity graph
    grid = sklearn.feature_extraction.image.grid_to_graph(  n_x=data.shape[1], 
                                                            n_y=1, 
                                                            n_z=1)

    # Instantiate the clustering object
    ward = sklearn.cluster.Ward(n_clusters=k, connectivity=grid)

    # Fit the model
    ward.fit(data.T)

    # Find the change points from the labels
    boundaries = [0]
    boundaries.extend(
        list(1 + np.nonzero(np.diff(ward.labels_))[0].astype(int)))
    return boundaries

