#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""

import numpy as np
import os
import glob
import pkg_resources

from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin

import librosa.core
from . import cache

EXAMPLE_AUDIO = 'example_data/Kevin_MacLeod_-_Vibe_Ace.mp3'


def example_audio_file():
    '''Get the path to an included audio example file.

    :usage:
        >>> # Load the waveform from the example track
        >>> y, sr = librosa.load(librosa.util.example_audio_file())

    :parameters:
        - None

    :returns:
        - filename : str
            Path to the audio example file included with librosa

    .. raw:: html

        <div xmlns:cc="http://creativecommons.org/ns#"
             xmlns:dct="http://purl.org/dc/terms/"
             about="http://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace_1278">
             <span property="dct:title">Vibe Ace</span>
             (<a rel="cc:attributionURL" property="cc:attributionName"
                 href="http://freemusicarchive.org/music/Kevin_MacLeod/">Kevin MacLeod</a>)
             / <a rel="license" href="http://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>
        </div>
    '''

    return pkg_resources.resource_filename(__name__, EXAMPLE_AUDIO)


def frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.

    This implementation uses low-level stride manipulation to avoid
    redundant copies of the time series data.

    :usage:
        >>> # Load a file
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> # Extract 2048-sample frames from y with a hop of 64
        >>> librosa.util.frame(y, frame_length=2048, hop_length=64)
        array([[  0.000e+00,   0.000e+00, ...,   1.526e-05,   0.000e+00],
               [  0.000e+00,   0.000e+00, ...,   1.526e-05,   0.000e+00],
               ...,
               [ -2.674e-04,   5.065e-03, ...,   0.000e+00,   0.000e+00],
               [  2.684e-03,   4.817e-03, ...,   0.000e+00,   0.000e+00]],
              dtype=float32)

    :parameters:
      - y : np.ndarray [shape=(n,)]
          Time series to frame. Must be one-dimensional and contiguous
          in memory.

      - frame_length : int > 0 [scalar]
          Length of the frame in samples

      - hop_length : int > 0 [scalar]
          Number of samples to hop between frames

    :returns:
      - y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]
          An array of frames sampled from ``y``:
          ``y_frames[i, j] == y[j * hop_length + i]``

    :raises:
      - ValueError
          If ``y`` is not contiguous in memory, framing is invalid.
          See ``np.ascontiguous()`` for details.

          If ``hop_length < 1``, frames cannot advance.
    '''

    if hop_length < 1:
        raise ValueError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ValueError('Input buffer must be contiguous.')

    valid_audio(y)

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    if n_frames < 1:
        raise ValueError('Buffer is too short '
                         '(n={:d}) for frame_length={:d}'.format(len(y),
                                                                 frame_length))
    # Vertical stride is one sample
    # Horizontal stride is ``hop_length`` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


@cache
def valid_audio(y, mono=True):
    '''Validate whether a variable contains valid, mono audio data.

    :usage:
        >>> # Only allow monophonic signals
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> valid_audio(y)

        >>> # If we want to allow stereo signals
        >>> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
        >>> valid_audio(y, mono=False)

    :parameters:
        - y
            The input data to validate

        - mono : bool
            Whether or not to force monophonic audio

    :raises:
        - ValueError
            If `y` fails to meet the following criteria:
              - `typebryce(y)` is `np.ndarray`
              - `mono == True` and `y.ndim` is not 1
              - `mono == False` and `y.ndim` is not 1 or 2
              - `np.isfinite(y).all()` is not True
    '''

    if not isinstance(y, np.ndarray):
        raise ValueError('Data is not a numpy ndarray audio buffer.')

    if mono and y.ndim != 1:
        raise ValueError('Invalid shape for monophonic audio: '
                         'ndim={:d}, shape={:s}'.format(y.ndim, y.shape))
    elif y.ndim > 2:
        raise ValueError('Invalid shape for audio: '
                         'ndim={:d}, shape={:s}'.format(y.ndim, y.shape))

    if not np.isfinite(y).all():
        raise ValueError('Audio buffer is not finite everywhere.')


@cache
def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to ``str.center()``

    :usage:
        >>> # Generate a vector
        >>> data = np.ones(5)
        >>> librosa.util.pad_center(data, 10, mode='constant')
        array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

        >>> # Pad a matrix along its first dimension
        >>> data = np.ones((3, 5))
        >>> librosa.util.pad_center(data, 7, axis=0)
        array([[ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.]])
        >>> # Or its second dimension
        >>> librosa.util.pad_center(data, 7, axis=1)
        array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
               [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
               [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    :parameters:
        - data : np.ndarray
            Vector to be padded and centered

        - size : int >= len(data) [scalar]
            Length to pad ``data``

        - axis : int
            Axis along which to pad and center the data

        - *kwargs*
            Additional keyword arguments passed to ``np.pad()``

    :returns:
        - data_padded : np.ndarray
            ``data`` centered and padded to length ``size`` along the
            specified axis

    :raises:
        - ValueError
            If ``size < data.shape[axis]``
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) / 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, size - n - lpad)

    if lpad < 0:
        raise ValueError('Target size {:d} < input size {:d}'.format(size, n))

    return np.pad(data, lengths, **kwargs)


@cache
def fix_length(data, size, axis=-1, **kwargs):
    '''Fix the length an array ``data`` to exactly ``size``.

    If ``data.shape[axis] < n``, pad according to the provided kwargs.
    By default, ``data`` is padded with trailing zeros.

    :usage:
        >>> y = np.arange(7)
        >>> # Default: pad with zeros
        >>> librosa.util.fix_length(y, 10)
        array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
        >>> # Trim to a desired length
        >>> librosa.util.fix_length(y, 5)
        array([0, 1, 2, 3, 4])
        >>> # Use edge-padding instead of zeros
        >>> librosa.util.fix_length(y, 10, mode='edge')
        array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    :parameters:
      - data : np.ndarray
          array to be length-adjusted

      - size : int >= 0 [scalar]
          desired length of the array

      - axis : int, <= data.ndim
          axis along which to fix length

      - *kwargs*
          Additional keyword arguments.  See ``np.pad()``

    :returns:
      - data_fixed : np.ndarray [shape=data.shape]
          ``data`` either trimmed or padded to length ``size``
          along the specified axis.
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [Ellipsis] * data.ndim
        slices[axis] = slice(0, size)
        return data[slices]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data


@cache
def fix_frames(frames, x_min=0, x_max=None, pad=True):
    '''Fix a list of frames to lie within [x_min, x_max]

    :usage:
        >>> # Generate a list of frame indices
        >>> frames = np.arange(0, 1000.0, 50)
        >>> frames
        array([   0.,   50.,  100.,  150.,  200.,  250.,  300.,  350.,  400.,
                450.,  500.,  550.,  600.,  650.,  700.,  750.,  800.,  850.,
                900.,  950.])
        >>> # Clip to span at most 250
        >>> librosa.util.fix_frames(frames, x_max=250)
        array([  0,  50, 100, 150, 200, 250])
        >>> # Or pad to span up to 2500
        >>> librosa.util.fix_frames(frames, x_max=2500)
        array([   0,   50,  100,  150,  200,  250,  300,  350,  400,  450,
                500,  550,  600,  650,  700,  750,  800,  850,  900,  950,
               2500])
        >>> librosa.util.fix_frames(frames, x_max=2500, pad=False)
        array([  0,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
               600, 650, 700, 750, 800, 850, 900, 950])

        >>> # Or starting away from zero
        >>> frames = np.arange(200, 500, 33)
        >>> frames
        array([200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
        >>> librosa.util.fix_frames(frames)
        array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
        >>> librosa.util.fix_frames(frames, x_max=500)
        array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497, 500])

    :parameters:
        - frames : np.ndarray [shape=(n_frames,)]
            List of non-negative frame indices

        - x_min : int >= 0 or None
            Minimum allowed frame index

        - x_max : int >= 0 or None
            Maximum allowed frame index

        - pad : boolean
            If `True`, then `frames` is expanded to span the full range
            `[x_min, x_max]`

    :returns:
        - fixed_frames : np.ndarray [shape=(n_fixed_frames,), dtype=int]
            Fixed frame indices, flattened and sorted

    :raises:
        - ValueError
            If `frames` contains negative values
    '''

    if np.any(frames < 0):
        raise ValueError('Negative frame index detected')

    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)

    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((pad_data, frames))

    if x_min is not None:
        frames = frames[frames >= x_min]

    if x_max is not None:
        frames = frames[frames <= x_max]

    return np.unique(frames).astype(int)


@cache
def axis_sort(S, axis=-1, index=False, value=None):
    '''Sort an array along its rows or columns.

    :usage:
        >>> # Visualize NMF output for a spectrogram S
        >>> # Sort the columns of W by peak frequency bin
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> S = np.abs(librosa.stft(y))
        >>> W, H = librosa.decompose.decompose(S)
        >>> W_sort = librosa.util.axis_sort(W)
        >>> # Or sort by the lowest frequency bin
        >>> W_sort = librosa.util.axis_sort(W, value=np.argmin)
        >>> # Or sort the rows instead of the columns
        >>> W_sort_rows = librosa.util.axis_sort(W, axis=0)
        >>> # Get the sorting index also, and use it to permute the rows of H
        >>> W_sort, idx = librosa.util.axis_sort(W, index=True)
        >>> H_sort = H[index, :]

    :parameters:
      - S : np.ndarray [shape=(d, n)]
          Array to be sorted

      - axis : int [scalar]
          The axis along which to sort.

          - ``axis=0`` to sort rows by peak column index
          - ``axis=1`` to sort columns by peak row index

      - index : boolean [scalar]
          If true, returns the index array as well as the permuted data.

      - value : function
          function to return the index corresponding to the sort order.
          Default: ``np.argmax``.

    :returns:
      - S_sort : np.ndarray [shape=(d, n)]
          ``S`` with the columns or rows permuted in sorting order

      - idx : np.ndarray (optional) [shape=(d,) or (n,)]
        If ``index == True``, the sorting index used to permute ``S``.
        Length of ``idx`` corresponds to the selected ``axis``.

    :raises:
      - ValueError
          If ``S`` does not have exactly 2 dimensions.
    '''

    if value is None:
        value = np.argmax

    if S.ndim != 2:
        raise ValueError('axis_sort is only defined for 2-dimensional arrays.')

    bin_idx = value(S, axis=np.mod(1-axis, S.ndim))
    idx = np.argsort(bin_idx)

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


@cache
def normalize(S, norm=np.inf, axis=0):
    '''Normalize the columns or rows of a matrix

    :usage:
        >>> # Construct an example matrix
        >>> S = np.vander(np.arange(-2.0, 2.0))
        >>> S
        array([[-8.,  4., -2.,  1.],
               [-1.,  1., -1.,  1.],
               [ 0.,  0.,  0.,  1.],
               [ 1.,  1.,  1.,  1.]])
        >>> # Max (l-infinity)-normalize the columns
        >>> librosa.util.normalize(S)
        array([[-1.   ,  1.   , -1.   ,  1.   ],
               [-0.125,  0.25 , -0.5  ,  1.   ],
               [ 0.   ,  0.   ,  0.   ,  1.   ],
               [ 0.125,  0.25 ,  0.5  ,  1.   ]])
        >>> # Max (l-infinity)-normalize the rows
        >>> librosa.util.normalize(S, axis=1)
        array([[-1.   ,  0.5  , -0.25 ,  0.125],
               [-1.   ,  1.   , -1.   ,  1.   ],
               [ 0.   ,  0.   ,  0.   ,  1.   ],
               [ 1.   ,  1.   ,  1.   ,  1.   ]])
        >>> # l1-normalize the columns
        >>> librosa.util.normalize(S, norm=1)
        array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
               [-0.1  ,  0.167, -0.25 ,  0.25 ],
               [ 0.   ,  0.   ,  0.   ,  0.25 ],
               [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
        >>> # l2-normalize the columns
        >>> librosa.util.normalize(S, norm=2)
        array([[-0.985,  0.943, -0.816,  0.5  ],
               [-0.123,  0.236, -0.408,  0.5  ],
               [ 0.   ,  0.   ,  0.   ,  0.5  ],
               [ 0.123,  0.236,  0.408,  0.5  ]])

    :parameters:
      - S : np.ndarray [shape=(d, n)]
          The matrix to normalize

      - norm : {inf, -inf, 0, float > 0}
          - ``inf``  : maximum absolute value
          - ``-inf`` : mininum absolute value
          - ``0``    : number of non-zeros
          - float  : corresponding l_p norm.
            See ``scipy.linalg.norm`` for details.

      - axis : int [scalar]
          Axis along which to compute the norm.
          ``axis=0`` will normalize columns, ``axis=1`` will normalize rows.
          ''axis=None'' will normalize according to the entire matrix.

    :returns:
      - S_norm : np.ndarray [shape=S.shape]
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
    length[length < librosa.core.SMALL_FLOAT] = 1.0

    return S / length


@cache
def match_intervals(intervals_from, intervals_to):
    '''Match one set of time intervals to another.

    This can be useful for tasks such as mapping beat timings
    to segments.

    .. note:: A target interval may be matched to multiple source
      intervals.

    :parameters:
      - intervals_from : np.ndarray [shape=(n, 2)]
          The time range for source intervals.
          The ``i`` th interval spans time ``intervals_from[i, 0]``
          to ``intervals_from[i, 1]``.
          ``intervals_from[0, 0]`` should be 0, ``intervals_from[-1, 1]``
          should be the track duration.

      - intervals_to : np.ndarray [shape=(m, 2)]
          Analogous to ``intervals_from``.

    :returns:
      - interval_mapping : np.ndarray [shape=(n,)]
          For each interval in ``intervals_from``, the
          corresponding interval in ``intervals_to``.

    .. seealso:: :func:`librosa.util.match_events`
    '''

    # The overlap score of a beat with a segment is defined as
    #   max(0, min(beat_end, segment_end) - max(beat_start, segment_start))
    output = np.empty(len(intervals_from), dtype=np.int)

    n_rows = int(librosa.core.MAX_MEM_BLOCK / (len(intervals_to)
                                               * intervals_to.itemsize))

    n_rows = max(1, n_rows)

    for bl_s in range(0, len(intervals_from), n_rows):
        bl_t = min(bl_s + n_rows, len(intervals_from))
        tmp_from = intervals_from[bl_s:bl_t]

        starts = np.maximum.outer(tmp_from[:, 0], intervals_to[:, 0])
        ends = np.minimum.outer(tmp_from[:, 1], intervals_to[:, 1])
        score = np.maximum(0, ends - starts)

        output[bl_s:bl_t] = np.argmax(score, axis=-1)

    return output


@cache
def match_events(events_from, events_to):
    '''Match one set of events to another.

    This is useful for tasks such as matching beats to the nearest
    detected onsets, or frame-aligned events to the nearest zero-crossing.

    .. note:: A target event may be matched to multiple source events.

    :usage:
        >>> # Sources are multiples of 7
        >>> s_from = np.arange(0, 100, 7)
        >>> s_from
        array([ 0,  7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98])
        >>> # Targets are multiples of 10
        >>> s_to = np.arange(0, 100, 10)
        >>> s_to
        array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        >>> # Find the matching
        >>> idx = librosa.util.match_events(s_from, s_to)
        >>> idx
        array([0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9])
        >>> # Print each source value to its matching target
        >>> zip(s_from, s_to[idx])
        [(0, 0), (7, 10), (14, 10), (21, 20), (28, 30), (35, 30),
         (42, 40), (49, 50), (56, 60), (63, 60), (70, 70), (77, 80),
         (84, 80), (91, 90), (98, 90)]

    :parameters:
      - events_from : ndarray [shape=(n,)]
          Array of events (eg, times, sample or frame indices) to match from.

      - events_to : ndarray [shape=(m,)]
          Array of events (eg, times, sample or frame indices) to
          match against.

    :returns:
      - event_mapping : np.ndarray [shape=(n,)]
          For each event in ``events_from``, the corresponding event
          index in ``events_to``.

          ``event_mapping[i] == arg min |events_from[i] - events_to[:]|``

    .. seealso:: :func:`librosa.util.match_intervals`
    '''
    output = np.empty_like(events_from, dtype=np.int)

    n_rows = int(librosa.core.MAX_MEM_BLOCK / (np.prod(output.shape[1:])
                                               * len(events_to)
                                               * events_from.itemsize))

    # Make sure we can at least make some progress
    n_rows = max(1, n_rows)

    # Iterate over blocks of the data
    for bl_s in range(0, len(events_from), n_rows):
        bl_t = min(bl_s + n_rows, len(events_from))

        event_block = events_from[bl_s:bl_t]
        output[bl_s:bl_t] = np.argmin(np.abs(np.subtract.outer(event_block,
                                                               events_to)),
                                      axis=-1)

    return output


def find_files(directory, ext=None, recurse=True, case_sensitive=False,
               limit=None, offset=0):
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

          Otherwise, only ``directory`` will be searched.

      - case_sensitive : boolean
          If ``False``, files matching upper-case version of
          extensions will be included.

      - limit : int > 0 or None
          Return at most ``limit`` files. If ``None``, all files are returned.

      - offset : int
          Return files starting at ``offset`` within the list.

          Use negative values to offset from the end of the list.

    :returns:
      - files : list of str
          The list of audio files.
    '''

    def _get_files(dir_name, extensions):
        '''Helper function to get files in a single directory'''

        # Expand out the directory
        dir_name = os.path.abspath(os.path.expanduser(dir_name))

        myfiles = []
        for sub_ext in extensions:
            globstr = os.path.join(dir_name, '*' + os.path.extsep + sub_ext)
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


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Sci-kit learn wrapper class for feature extraction methods.

    This class acts as a bridge between feature extraction functions
    and scikit-learn pipelines.

    :usage:
        >>> import librosa
        >>> import sklearn.pipeline
        >>>
        >>> # Build a mel-spectrogram extractor
        >>> MS = librosa.util.FeatureExtractor(librosa.feature.melspectrogram,
                                               sr=22050, n_fft=2048,
                                               n_mels=128, fmax=8000)
        >>>
        >>> # And a log-amplitude extractor
        >>> LA = librosa.util.FeatureExtractor(librosa.logamplitude,
                                               ref_power=np.max)
        >>>
        >>> # Chain them into a pipeline
        >>> Features = sklearn.pipeline.Pipeline([('MelSpectrogram', MS),
                                                  ('LogAmplitude', LA)])
        >>>
        >>> # Load an audio file
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>>
        >>> # Apply the transformation to y
        >>> F = Features.transform([y])

    :parameters:
      - function : function
          The feature extraction function to wrap.

          Example: ``librosa.feature.melspectrogram``

      - target : str or None
          If ``None``, then ``function`` is called with the input
          data as the first positional argument.

          If ``str``, then ``function`` is called with the input
          data as a keyword argument with key ``target``.

      - iterate : bool
          If ``True``, then ``function`` is applied iteratively to each
          item of the input.

          If ``False``, then ``function`` is applied to the entire data
          stream simultaneously.  This is useful for things like aggregation
          and stacking.

      - *kwargs*
          Parameters to be passed through to ``function``
    """

    def __init__(self, function, target=None, iterate=True, **kwargs):
        self.function = function
        self.target = target
        self.iterate = iterate
        self.kwargs = {}

        self.set_params(**kwargs)

    # Clobber _get_param_names here for transparency
    def _get_param_names(self):
        """Returns the parameters of the feature extractor as a dictionary."""
        temp_params = {'function': self.function, 'target': self.target}

        temp_params.update(self.kwargs)

        return temp_params

    # Wrap set_params to catch updates
    def set_params(self, **kwargs):
        """Update the parameters of the feature extractor."""

        # We don't want non-functional arguments polluting kwargs
        params = kwargs.copy()
        for k in ['function', 'target']:
            params.pop(k, None)

        self.kwargs.update(params)
        BaseEstimator.set_params(self, **kwargs)

    # We keep these arguments for compatibility, but don't use them.
    def fit(self, *args, **kwargs):  # pylint: disable=unused-argument
        """This function does nothing, and is provided for interface compatibility.

        .. note:: Since most ``TransformerMixin`` classes implement some
            statistical modeling (e.g., PCA), the ``fit()`` method is
            required.

            For the ``FeatureExtraction`` class, all parameters are fixed
            ahead of time, and no statistical estimation takes place.
        """

        return self

    # Variable name 'X' is for consistency with sklearn
    def transform(self, X):  # pylint: disable=invalid-name
        """Applies the feature transformation to an array of input data.

        :parameters:
          - X : iterable
              Array or list of input data

        :returns:
          - X_transform : list
              In positional argument mode (target=None), then
              ``X_transform[i] = function(X[i], [feature parameters])``

              If the ``target`` parameter was given, then
              ``X_transform[i] = function(target=X[i], [feature parameters])``
        """

        if self.target is not None:
            # If we have a target, each element of X takes the keyword argument
            if self.iterate:
                return [self.function(**dict(list(self.kwargs.items())
                                             + list({self.target: i}.items())))
                        for i in X]
            else:
                return self.function(**dict(list(self.kwargs.items())
                                            + list({self.target: X}.items())))
        else:
            # Each element of X takes first position in function()
            if self.iterate:
                return [self.function(i, **self.kwargs) for i in X]
            else:
                return self.function(X, **self.kwargs)


def buf_to_int(x, n_bytes=2):
    """Convert a floating point buffer into integer values.
    This is primarily useful as an intermediate step in wav output.

    .. seealso:: :func:`librosa.util.buf_to_float`

    :parameters:
        - x : np.ndarray [dtype=float]
            Floating point data buffer

        - n_bytes : int [1, 2, 4]
            Number of bytes per output sample

    :returns:
        - x_int : np.ndarray [dtype=int]
            The original buffer cast to integer type.
    """

    # What is the scale of the input data?
    scale = float(1 << ((8 * n_bytes) - 1))

    # Construct a format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and cast the data
    return (x * scale).astype(fmt)


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    .. seealso:: :func:`librosa.util.buf_to_float`

    :parameters:
        - x : np.ndarray [dtype=int]
            The integer-valued data buffer

        - n_bytes : int [1, 2, 4]
            The number of bytes per sample in ``x``

        - dtype : numeric type
            The target output type (default: 32-bit float)

    :return:
        - x_float : np.ndarray [dtype=float]
            The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
