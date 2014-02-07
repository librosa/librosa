#!/usr/bin/env python
"""Utility functions"""

import numpy as np
import os
import glob

from numpy.lib.stride_tricks import as_strided

def frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.
    
    This implementation uses low-level stride manipulation to avoid
    redundant copies of the time series data.

    :usage:
        >>> # Load a file
        >>> y, sr = librosa.load('file.mp3')
        >>> # Extract 2048-sample frames from y with a hop of 64
        >>> y_frames = librosa.util.frame(y, frame_length=2048, hop_length=64)

    :parameters:
      - y : np.ndarray, ndim=1
        Time series to frame

      - frame_length : int > 0
        Length of the frame in samples

      - hop_length : int > 0
        Number of samples to hop between frames

    :returns:
      - y_frames : np.ndarray, shape=(frame_length, N_FRAMES)
        An array of frames sampled from ``y``:
        ``y_frames[i, j] == y[j * hop_length + i]``
    '''

    assert(hop_length > 0)

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int( (len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is ``hop_length`` samples
    y_frames = as_strided(  y, 
                            shape=(frame_length, n_frames), 
                            strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames

def pad_center(data, size, **kwargs):
    '''Wrapper for np.pad to automatically center a vector prior to padding.
    This is analogous to ``str.center()``

    :usage:
        >>> # Generate a window vector
        >>> window = scipy.signal.hann(256)
        >>> # Center and pad it out to length 1024
        >>> window = librosa.util.pad_center(window, 1024, mode='constant')

    :parameters:
        - data : np.ndarray, ndim=1
          Vector to be padded and centered 

        - size : int >= len(data)
          Length to pad ``data``

        - kwargs
          Additional keyword arguments passed to ``numpy.pad()``
    
    :returns:
        - data_padded : np.ndarray, ndim=1
          ``data`` centered and padded to length ``size``
    '''

    kwargs.setdefault('mode', 'constant')
    lpad = (size - len(data))/2
    return np.pad( data, (lpad, size - len(data) - lpad), **kwargs) 

def axis_sort(S, axis=-1, index=False, value=None): 
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

        >>> # Get the sorting index also, and use it to permute the rows of H
        >>> W_sort, idx = librosa.util.axis_sort(W, index=True)
        >>> H_sort = H[index, :]
        >>> # np.dot(W_sort, H_sort) == np.dot(W, H)

    :parameters:
      - S : np.ndarray, ndim=2
        Matrix to sort
        
      - axis : int
        The axis along which to sort.  
        
        - ``axis=0`` to sort rows by peak column index
        - ``axis=1`` to sort columns by peak row index

      - index : boolean    
        If true, returns the index array as well as the permuted data.

      - value : function
        function to return the index corresponding to the sort order.
        Default: ``np.argmax``.

    :returns:
      - S_sort : np.ndarray
        ``S`` with the columns or rows permuted in sorting order
     
      - idx : np.ndarray (optional)
        If ``index == True``, the sorting index used to permute ``S``.

    :raises:
      - ValueError
        If ``S`` does not have 2 dimensions.
    '''
    
    if value is None:
        value = np.argmax

    if S.ndim != 2:
        raise ValueError('axis_sort is only defined for 2-dimensional arrays.')
        
    bin_idx = value(S, axis=np.mod(1-axis, S.ndim))
    idx     = np.argsort(bin_idx)
    
    if axis == 0:
        if index:
            return S[idx, :], idx
        else:
            return S[idx, :]
    else:
        if index:
            return S[:, idx], idx
        else:
            return S[:, idx]

def normalize(S, norm=np.inf, axis=0):
    '''Normalize the columns or rows of a matrix
    
    :parameters:
      - S : np.ndarray
        The matrix to normalize
      
      - norm : {inf, -inf, 0, float > 0}
        - ``inf``  : maximum absolute value
        - ``-inf`` : mininum absolute value
        - ``0``    : number of non-zeros
        - float  : corresponding l_p norm. See ``scipy.linalg.norm`` for details.
    
      - axis : int
        Axis along which to compute the norm.
        ``axis=0`` will normalize columns, ``axis=1`` will normalize rows.
      
    :returns: 
      - S_norm : np.ndarray
        Normalized matrix
      
    .. note::
        Columns/rows with length 0 will be left as zeros.
    '''
    
    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S)
    
    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)
        
    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)
        
    elif norm == 0:
        length = np.sum(mag > 0, axis=axis, keepdims=True)
        
    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True)**(1./norm)
    else:
        raise ValueError('Unsupported norm value: ' + repr(norm))
        
    # Avoid div-by-zero
    length[length == 0] = 1.0
    
    return S / length

def find_files(directory, ext=None, recurse=True, case_sensitive=False, limit=None, offset=0):
    '''Get a sorted list of (audio) files in a directory or directory sub-tree.
    
    :usage:
       >>> # Get all audio files in a directory sub-tree
       >>> files = librosa.util.find_files('~/Music')
       
       >>> # Look only within a specific directory, not the sub-tree
       >>> files = librosa.util.find_files('~/Music', recurse=False)
       
       >>> # Only look for mp3 files
       >>> files = librosa.util.find_files('~/Music', ext='mp3')
       
       >>> # Or just mp3 and ogg
       >>> files = librosa.util.find_files('~/Music', ext=['mp3', 'ogg'])
       
       >>> # Only get the first 10 files
       >>> files = librosa.util.find_files('~/Music', limit=10)
       
       >>> # Or last 10 files
       >>> files = librosa.util.find_files('~/Music', offset=-10)
       
    :parameters:
      - directory : str
        Path to look for files
        
      - ext : str or list of str
        A file extension or list of file extensions to include in the search.
        Default: ``['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']``
        
      - recurse : boolean
        If ``True``, then all subfolders of ``directory`` will be searched.
        Otherwise, only `directory` will be searched.
        
      - case_sensitive : boolean
        If ``False``, files matching upper-case version of extensions will be included.
        
      - limit : int >0 or None
        Return at most ``limit`` files. If ``None``, all files are returned.
        
      - offset : int
        Return files starting at ``offset`` within the list.
        Use negative values to offset from the end of the list.
        
    :returns:
      - files, list of str
        The list of audio files.
    '''
    
    def _get_files(D, extensions):
        '''Helper function to get files in a single directory'''

        # Expand out the directory
        D = os.path.abspath(os.path.expanduser(D))
        
        myfiles = []
        for sub_ext in extensions:
            globstr = os.path.join(D, '*' + os.path.extsep + sub_ext)
            myfiles.extend(glob.glob(globstr))
        
        return myfiles
            
    if ext is None:
        ext = ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']
        
    elif isinstance(ext, str):
        if not case_sensitive:
            ext = ext.lower()
        ext = [ext]
        
    # Generate upper-case versions
    if not case_sensitive:
        for i in range(len(ext)):
            ext.append(ext[i].upper())
    
    files = []
    
    if recurse:
        for walk in os.walk(directory):
            files.extend(_get_files(walk[0], ext))
    else:
        files = _get_files(directory, ext)
    
    files.sort()
    files = files[offset:]
    if limit is not None:
        files = files[:limit]
    
    return files
