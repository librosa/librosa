#!/usr/bin/env python
"""Temporal segmentation"""

import numpy as np
import scipy
import scipy.signal

import sklearn
import sklearn.cluster
import sklearn.feature_extraction

import librosa.core

def stack_memory(X, m=2, delay=1):
    """Short-term history embedding.

    Each column `xi = X[:, i]` is mapped to
    ```
    X[:,i] =>   [   X[:, i]
                    X[:, i - delay]
                    ...
                    X[:, i - (m-1)*delay]
                ]```

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


def segment(data, k):
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

