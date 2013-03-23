#!/usr/bin/env python
'''
CREATED:2012-11-05 14:38:03 by Brian McFee <brm2132@columbia.edu>

All things rhythmic go here

- Onset detection
- Tempo estimation
- Beat tracking
- Segmentation

'''

import librosa
import numpy as np
import scipy, scipy.signal, scipy.ndimage
import sklearn, sklearn.cluster, sklearn.feature_extraction

def beat_track(onsets=None, y=None, sr=22050, hop_length=64, start_bpm=120.0):
    '''
    Ellis-style beat tracker

    Input:
        onsets:         pre-computed onset envelope                 | default: None
        y:              time-series data                            | default: None
        sr:             sample rate of y                            | default: 22050
        hop_length:     hop length (in frames) for onset detection  | default: 64
        start_bpm:      initial guess for BPM estimator             | default: 120.0

        Either onsets or y must be provided.

    Output:
        bpm:            estimated global tempo
        beats:          array of estimated beats by frame number
    '''

    # First, get the frame->beat strength profile if we don't already have one
    if onsets is None:
        if y is None:
            raise ValueError('Either "y" or "onsets" must be provided')

        onsets  = onset_strength(y=y, sr=sr, hop_length=hop_length)

    fft_res = float(sr) / hop_length

    # Then, estimate bpm
    bpm     = onset_estimate_bpm(onsets, start_bpm, fft_res)
    
    # Then, run the tracker: tightness = 400
    beats   = __beat_tracker(onsets, bpm, fft_res, 400)

    return (bpm, beats)



def __beat_tracker(onsets, bpm, fft_res, tightness):
    '''
        Internal function that does beat tracking from a given onset profile.

        Input:
            onsets:     the onset envelope
            bpm:        the tempo estimate
            fft_res:    resolution of the fft (sr / hop_length)
            tightness:  tight

        Output:
            frame numbers of beat events
    '''

    #--- First, some helper functions ---#
    def rbf(points):
        '''
        Makes a smoothing filter for onsets
        '''
        return np.exp(-0.5 * (points**2))

    def beat_track_dp(localscore):  
        '''
        Core dynamic program for beat tracking
        '''
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
        '''
        Get the last beat from the cumulative score array
        '''
        maxes       = librosa.localmax(cumscore)
        med_score   = np.median(cumscore[np.argwhere(maxes)])

        # The last of these is the last beat (since score generally increases)
        return np.argwhere((cumscore * maxes * 2 > med_score)).max()

    def smooth_beats(beats):
        '''
        Final post-processing: throw out spurious leading/trailing beats
        '''
        
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
    beats = smooth_beats(beats)

    # Add one to account for differencing offset
    return 1 + beats

def onset_estimate_bpm(onsets, start_bpm, fft_res):
    '''
    Estimate the BPM from an onset envelope.

    Input:
        onsets:         time-series of onset strengths
        start_bpm:      initial guess of the BPM
        fft_res:        resolution of FFT (sample rate / hop length)

    Output:
        estimated BPM
    '''
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
    x_corr      = librosa.autocorrelate(onsets[mincol:maxcol], ac_window)


    #   FIXME:  2013-01-25 08:55:40 by Brian McFee <brm2132@columbia.edu>
    #   this fails if ac_window > length of song   
    # re-weight the autocorrelation by log-normal prior
    bpms    = 60.0 * fft_res / (np.arange(1, ac_window+1))

    # Smooth the autocorrelation by a log-normal distribution
    x_corr  = x_corr * np.exp(-0.5 * ((np.log2(bpms / start_bpm)) / bpm_std)**2)

    # Get the local maximum of weighted correlation
    x_peaks = librosa.localmax(x_corr)

    # Zero out all peaks before the first negative
    x_peaks[:np.argmax(x_corr < 0)] = False


    # Choose the best peak out of .33, .5, 2, 3 * start_period
    candidates      = np.multiply(  np.argmax(x_peaks * x_corr), 
                                    [1.0/3, 1.0/2, 1.0, 2.0, 3.0])

    candidates      = candidates.astype(int)
    candidates      = candidates[candidates < ac_window]

    best_period     = np.argmax(x_corr[candidates])

    return 60.0 * fft_res / candidates[best_period]


def onset_strength(S=None, y=None, sr=22050, **kwargs):
    '''
    Adapted from McVicar, adapted from Ellis, etc...
    
    Extract onsets

    INPUT:
        S               = pre-computed spectrogram              | default: None
        y               = time-series waveform (t-by-1 vector)  | default: None
        sr              = sampling rate of the input signal     | default: 22050

        Either S or y,sr must be provided.

        **kwargs        = Parameters to mel spectrogram, if S is not provided
                          See librosa.feature.melspectrogram() for details

    OUTPUT:
        onset_envelope
    '''

    # First, compute mel spectrogram
    if S is None:
        if y is None:
            raise ValueError('One of "S" or "y" must be provided.')

        S   = librosa.feature.melspectrogram(y, sr = sr, **kwargs)

        # Convert to dBs
        S   = librosa.logamplitude(S)


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
    '''
        Perform bottom-up temporal segmentation

        Input:
            data:   d-by-t  spectrogram (t frames)
            k:      number of segments to produce

        Output:
            s:          segment boundaries (frame numbers)
            centroid:   d-by-k  centroids (ordered temporall)
            variance:   d-by-k  variance (mean distortion) for each segment

    '''

    # Connect the temporal connectivity graph
    grid = sklearn.feature_extraction.image.grid_to_graph(  n_x=data.shape[1], 
                                                            n_y=1, 
                                                            n_z=1)

    # Instantiate the clustering object
    ward = sklearn.cluster.Ward(n_clusters=k, connectivity=grid)

    # Fit the model
    ward.fit(data.T)

    # Instantiate output objects
    centers     = np.empty( (data.shape[0], k) )
    variances   = np.empty( (data.shape[0], k) )
    starts      = np.empty(k, dtype=int)

    # Find the change points from the labels
    deltas  = list(1 + np.nonzero(np.diff(ward.labels_))[0].astype(int))

    # tack on the last frame as a change point
    deltas.append(data.shape[1])

    start = 0
    for (i, end) in enumerate(deltas):
        starts[i]       = start
        centers[:, i]   = np.mean(data[:, start:end], axis=1)
        variances[:, i] = np.var( data[:, start:end], axis=1)
        start           = end

    return (starts, centers, variances)

