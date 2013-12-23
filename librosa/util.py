#!/usr/bin/env python
"""Utility functions"""

import numpy as np

import os, glob

def axis_sort(X, axis=-1, value=np.argmax): 
    '''Sort an array along its rows or columns.
    
    :usage:
        >>> # Visualize NMF output for a spectrogram S
        >>> # Sort the columns of W by peak frequency bin
        >>> W, H = librosa.decompose.decompose(S)
        >>> W_sort = librosa.util.axis_sort(W)

        >>> # Or sort by the lowest frequency bin
        >>> W_sort = librosa.util.axis_sort(W, value=np.argmin)

        >>> # Or sort the rows instead of the columns
        >>> W_sort_rows = librosa.util.axis_sort(W, axis=0)

    :parameters:
      - X : np.ndarray, ndim=2
        Matrix to sort
        
      - axis : int
        The axis along which to sort.  
        
        - `axis=0` to sort rows by peak column index
        - `axis=1` to sort columns by peak row index
        
      - value : function
        function to return the index corresponding to the sort order.
        Default: `np.argmax`.

    :raises:
      - ValueError
        If `X` does not have 2 dimensions.
    '''
    
    if X.ndim != 2:
        raise ValueError('axis_sort is only defined for 2-dimensional arrays.')
        
    bin_idx = value(X, axis=np.mod(1-axis, X.ndim))
    idx     = np.argsort(bin_idx)
    
    if axis == 0:
        return X[idx,:]
    
    return X[:, idx]



