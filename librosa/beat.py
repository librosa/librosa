#!/usr/bin/env python
"""Onsets, tempo, beats and segmentation."""

import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import sklearn
import sklearn.cluster
import sklearn.feature_extraction

import librosa.core
import librosa.feature

def beat_track(y=None, sr=22050, onsets=None, hop_length=128, 
               start_bpm=120.0, n_fft=256, tightness=400, trim=True):
    """Ellis-style beat tracker

    :parameters:
      - y          : np.ndarray 
          audio time series

      - sr         : int
          audio sample rate

      - onsets     : np.ndarray 
          (optional) pre-computed onset envelope

      - hop_length : int
          hop length (in frames)

      - start_bpm  : float
          initial guess for BPM estimator

      - n_fft      : int
          window size (centers beat times).
          Set ``n_fft=None`` to disable frame centering.

      - tightness  : float
          tightness of beat distribution around tempo

      - trim       : bool
          trim leading/trailing beats with weak onsets?

      .. note:: One of either ``onsets`` or ``y`` must be provided.


    :returns: 
      - bpm : float
          estimated global tempo

      - beats : np.ndarray
          estimated frame numbers of beats

    :raises:
      - ValueError  
          if neither y nor onsets are provided

    .. note::
      If no onset strength could be detected, beat_tracker estimates 0 BPM and returns
      an empty list.

    .. note:: 

      - http://labrosa.ee.columbia.edu/projects/beattrack/
      - D. Ellis (2007)
        Beat Tracking by Dynamic Programming
        Journal of New Music Research
        Special Issue on Beat and Tempo Extraction
        vol. 36 no. 1, March 2007, pp. 51-60. 

    """

    # First, get the frame->beat strength profile if we don't already have one
    if onsets is None:
        if y is None:
            raise ValueError('Either "y" or "onsets" must be provided')

        onsets  = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Do we have any onsets to grab?
    if not onsets.any():
        return (0, np.array([], dtype=int))

    fft_res = float(sr) / hop_length

    # Then, estimate bpm
    bpm     = onset_estimate_bpm(onsets, start_bpm, fft_res)
    
    # Then, run the tracker: tightness = 400

    beats   = __beat_tracker(onsets, bpm, fft_res, tightness, trim)

    # Framing correction
    if n_fft is None:
        n_fft = hop_length
    
    beats = beats + n_fft / (hop_length)

    return (bpm, beats)


def __beat_tracker(onsets, bpm, fft_res, tightness, trim):
    """Internal function that does beat tracking from a given onset profile.

    :parameters:
      - onsets   : np.ndarray
          onset envelope
      - bpm      : float
          tempo estimate
      - fft_res  : float
          resolution of the fft (sr / hop_length)
      - tightness: float
          how closely do we adhere to bpm?
      - trim     : boolean
          trim leading/trailing beats with weak onsets?

    :returns:
      - beats    : np.ndarray
          frame numbers of beat events

    """

    #--- First, some helper functions ---#
    def rbf(points):
        """Makes a smoothing filter for onsets"""
        
        return np.exp(-0.5 * (points**2))

    def beat_track_dp(localscore):  
        """Core dynamic program for beat tracking"""

        backlink    = np.zeros_like(localscore, dtype=int)
        cumscore    = np.zeros_like(localscore)

        # Search range for previous beat
        window      = np.arange(-2*period, -np.round(period/2) + 1, dtype=int)

        # Make a score window, which begins biased toward start_bpm and skewed 
        txwt        = - tightness * np.log(-window /period)**2

        # Are we on the first beat?
        first_beat  = True
        for i in xrange(len(localscore)):

            # Are we reaching back before time 0?
            z_pad = np.maximum(0, min(- window[0], len(window)))

            # Search over all possible predecessors 
            candidates          = txwt.copy()
            candidates[z_pad:]  = candidates[z_pad:] + cumscore[window[z_pad:]]

            # Find the best preceding beat
            beat_location       = np.argmax(candidates)

            # Add the local score
            cumscore[i]         = localscore[i] + candidates[beat_location]

            # Special case the first onset.  Stop if the localscore is small
            if first_beat and localscore[i] < 0.01 * localscore.max():
                backlink[i]     = -1
            else:
                backlink[i]     = window[beat_location]
                first_beat      = False

            # Update the time range
            window  = window + 1

        return (backlink, cumscore)

    def get_last_beat(cumscore):
        """Get the last beat from the cumulative score array"""

        maxes       = librosa.core.localmax(cumscore)
        med_score   = np.median(cumscore[np.argwhere(maxes)])

        # The last of these is the last beat (since score generally increases)
        return np.argwhere((cumscore * maxes * 2 > med_score)).max()

    def smooth_beats(beats):
        """Final post-processing: throw out spurious leading/trailing beats"""
        
        smooth_boe  = scipy.signal.convolve(localscore[beats], 
                                            scipy.signal.hann(5), 'same')

        threshold   = 0.5 * ((smooth_boe**2).mean()**0.5)
        valid       = np.argwhere(smooth_boe > threshold)

        return  beats[valid.min():valid.max()]
    #--- End of helper functions ---#

    # convert bpm to a sample period for searching
    period      = round(60.0 * fft_res / bpm)

    # localscore is a smoothed version of AGC'd onset envelope
    localscore  = scipy.signal.convolve(
                        onsets / onsets.std(ddof=1), 
                        rbf(np.arange(-period, period+1)*32.0/period), 
                        'same')

    ### run the DP
    (backlink, cumscore) = beat_track_dp(localscore)

    ### get the position of the last beat
    beats   = [get_last_beat(cumscore)]

    ### Reconstruct the beat path from backlinks
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])

    ### Put the beats in ascending order
    beats.reverse()

    ### Convert into an array of frame numbers
    beats = np.array(beats, dtype=int)

    ### Discard spurious trailing beats
    if trim:
        beats = smooth_beats(beats)

    return beats


def onset_estimate_bpm(onsets, start_bpm, fft_res):
    """Estimate the BPM from an onset envelope

    :parameters:
      - onsets    : np.ndarray   
          time-series of onset strengths
      - start_bpm : float
          initial guess of the BPM
      - fft_res   : float
          resolution of FFT (sample rate / hop length)

    :returns:
      - bpm      : float
          estimated BPM

    """

    ac_size     = 4.0
    duration    = 90.0
    end_time    = 90.0
    bpm_std     = 1.0

    # Chop onsets to X[(upper_limit - duration):upper_limit]
    # or as much as will fit
    maxcol      = min(len(onsets)-1, np.round(end_time * fft_res))
    mincol      = max(0,    maxcol - np.round(duration * fft_res))

    # Use auto-correlation out of 4 seconds (empirically set??)
    ac_window   = np.round(ac_size * fft_res)

    # Compute the autocorrelation
    x_corr      = librosa.core.autocorrelate(onsets[mincol:maxcol], ac_window)


    #   FIXME:  2013-01-25 08:55:40 by Brian McFee <brm2132@columbia.edu>
    #   this fails if ac_window > length of song   
    # re-weight the autocorrelation by log-normal prior
    bpms    = 60.0 * fft_res / (np.arange(1, ac_window+1))

    # Smooth the autocorrelation by a log-normal distribution
    x_corr  = x_corr * np.exp(-0.5 * ((np.log2(bpms / start_bpm)) / bpm_std)**2)

    # Get the local maximum of weighted correlation
    x_peaks = librosa.core.localmax(x_corr)

    # Zero out all peaks before the first negative
    x_peaks[:np.argmax(x_corr < 0)] = False


    # Choose the best peak out of .33, .5, 2, 3 * start_period
    candidates      = np.multiply(  np.argmax(x_peaks * x_corr), 
                                    [1.0/3, 1.0/2, 1.0, 2.0, 3.0])

    candidates      = candidates.astype(int)
    candidates      = candidates[candidates < ac_window]

    best_period     = np.argmax(x_corr[candidates])

    return 60.0 * fft_res / candidates[best_period]


def onset_strength(y=None, sr=22050, S=None, **kwargs):
    """Extract onsets from an audio time series or spectrogram

    :parametrs:
      - y        : np.ndarray
          audio time-series
      - sr       : int
          audio sampling rate of y
      - S        : np.ndarray 
          pre-computed spectrogram

      - kwargs  
          Parameters to mel spectrogram, if S is not provided.

          See librosa.feature.melspectrogram() for details

    .. note:: if S is provided, then (y, sr) are optional.

    :returns:
      - onsets   : np.ndarray 
          vector of onset strength

    :raises:
      - ValueError 
          if neither (y, sr) nor S are provided

    """

    # First, compute mel spectrogram
    if S is None:
        if y is None:
            raise ValueError('One of "S" or "y" must be provided.')

        S   = librosa.feature.melspectrogram(y, sr = sr, **kwargs)

        # Convert to dBs
        S   = librosa.core.logamplitude(S)


    ### Compute first difference
    onsets  = np.diff(S, n=1, axis=1)

    ### Discard negatives (decreasing amplitude)
    #   falling edges could also be useful segmentation cues
    #   to catch falling edges, replace max(0,D) with abs(D)
    onsets  = np.maximum(0.0, onsets)

    ### Average over mel bands
    onsets  = onsets.mean(axis=0)

    ### remove the DC component
    onsets  = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onsets)

    return onsets 


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

