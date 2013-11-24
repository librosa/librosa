#!/usr/bin/env python
"""Onsets, tempo, beats and segmentation."""

import numpy as np
import scipy
import scipy.signal

import librosa.core
import librosa.feature

def onset_detect(y=None, sr=22050, onset_envelope=None, 
                            hop_length=64, **kwargs):
    """Basic onset detector
        
    :parameters:
      - y          : np.ndarray
          audio time series
    
      - sr         : int
          audio sample rate
    
      - onset_envelope     : np.ndarray
          (optional) pre-computed onset envelope
    
      - hop_length : int
          hop length (in frames)
      
      - kwargs  
          Parameters for peak picking
          
          See librosa.core.peak_pick() for details
 
      .. note:: One of either ``onset_envelope`` or ``y`` must be provided.
    
    
    :returns:
    
      - onsets : np.ndarray
          estimated frame numbers of onsets
    
    :raises:
      - ValueError
          if neither y nor onsets are provided
    
    .. note::
      If no onset strength could be detected, onset_detect returns an empty list.
                
    """
    
    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ValueError('Either "y" or "onsets" must be provided')
        onset_envelope  = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return np.array([], dtype=np.int)
    
    # Remove mean and normalize by std. dev.
    # (a common normalization step to make the threshold more consistent)
    onset_envelope -= onset_envelope.mean()
    onset_envelope /= onset_envelope.std()
    
    # Default values for peak picking
    # Taken from "MAXIMUM FILTER VIBRATO SUPPRESSION FOR ONSET DETECTION"
    kwargs['pre_max']  = int( kwargs.get( 'pre_max', .03*sr/hop_length ) )
    kwargs['post_max'] = int( kwargs.get( 'post_max', .03*sr/hop_length ) )
    kwargs['pre_avg']  = int( kwargs.get( 'pre_avg', .1*sr/hop_length ) )
    kwargs['post_avg'] = int( kwargs.get( 'post_avg', .07*sr/hop_length ) )
    kwargs['delta']    = kwargs.get( 'delta', 2 )
    kwargs['wait']     = int( kwargs.get( 'wait', .03*sr/hop_length ) )
    
    # Peak pick the onset envelope
    return librosa.core.peak_pick( onset_envelope, **kwargs )

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

        S   = librosa.feature.melspectrogram(y=y, sr=sr, **kwargs)

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

