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
import numpy, scipy, scipy.signal
import sklearn, sklearn.cluster, sklearn.feature_extraction

def beat_track(y, sr=22050, hop_length=64, start_bpm=120.0, tightness=400):
    '''
    Ellis-style beat tracker

    Input:
        y:              time-series data
        sr:             sample rate of y                            | default: 22050
        hop_length:     hop length (in frames) for onset detection  | default: 64 ~= 2.9ms
        start_bpm:      initial guess for BPM estimator             | default: 120.0
        tightness:      tightness parameter for tracker             | default: 400

    Output:
        bpm:            estimated global tempo
        beats:          array of estimated beats by frame number
    '''

    # First, get the frame->beat strength profile
    onsets  = onset_strength(y, sr)

    # Then, estimate bpm
    bpm     = onset_estimate_bpm(onsets, start_bpm, sr, hop_length)
    
    # Then, run the tracker
    beats   = _beat_tracker(onsets, start_bpm, sr, hop_length, tightness)

    return (bpm, beats)



def _beat_tracker(onsets, start_bpm, sampling_rate, hop_length, tightness, alpha=0.0):

    fft_resolution  = numpy.float(sampling_rate) / hop_length
    start_period    = int(round(60.0 * fft_resolution / start_bpm))
    period          = start_period

    # Smooth beat events with a gaussian window
    template        = numpy.exp(-0.5 * (numpy.arange(-period,(period+1)) / (period / hop_length))**2)

    # Convolve 
    localscore      = scipy.signal.convolve(onsets, template, 'same')
    max_localscore  = numpy.max(localscore)

    ### Initialize DP

    backlink        = numpy.zeros_like(localscore, dtype=int)
    cumscore        = numpy.zeros_like(localscore)


    # Search range for previous beat: number of samples forward/backward to look
    search_window   = numpy.arange(-2 * period, -numpy.round(period/2) + 1, dtype=int)

    # Make a score window, which begins biased toward start_bpm and skewed 
    txwt    = - tightness * numpy.abs(numpy.log(-search_window) - numpy.log(period))**2

    # Are we on the first beat?
    first_beat      = True

    time_range      = search_window
    # Forward step
    for i in xrange(len(localscore)):

        # Are we reaching back before time 0?
        z_pad               = numpy.maximum(0, numpy.minimum(- time_range[0], len(search_window)))

        # Search over all possible predecessors and apply transition weighting
        score_candidates                = txwt.copy()
        score_candidates[z_pad:]        = score_candidates[z_pad:] + cumscore[time_range[z_pad:]]

        # Find the best predecessor beat
        beat_location       = numpy.argmax(score_candidates)
        current_score       = score_candidates[beat_location]

        # Add the local score
        cumscore[i]         = current_score + localscore[i] - alpha

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and localscore[i] < 0.01 * max_localscore:
            backlink[i]     = -1
        else:
            backlink[i]     = time_range[beat_location]
            first_beat      = False
            pass

        # Update the time range
        time_range          = time_range + 1
        pass

    ### Get the last beat
    maxes                   = librosa.localmax(cumscore)
    max_indices             = numpy.nonzero(maxes)[0]
    peak_scores             = cumscore[max_indices]

    median_score            = numpy.median(peak_scores)
    bestendposs             = numpy.nonzero(cumscore * maxes > 0.5 * median_score)[0]

    # The last of these is the last beat (since score generally increases)
    bestendx                = numpy.max(bestendposs)

    b                       = [int(bestendx)]

    while backlink[b[-1]] >= 0:
        b.append(backlink[b[-1]])
        pass

    b.reverse()
    return numpy.array(b)

def onset_estimate_bpm(onsets, start_bpm, sampling_rate, hop_length):

    auto_correlation_size   = 4.0
    sample_duration         = 90.0
    sample_end_time         = 90.0
    bpm_std                 = 0.7

    fft_resolution          = numpy.float(sampling_rate) / hop_length

    # Chop onsets to X[(upper_limit - duration):upper_limit], or as much as will fit
    maxcol                  = min(numpy.round(sample_end_time * fft_resolution), len(onsets)-1)
    mincol                  = max(0, maxcol - numpy.round(sample_duration * fft_resolution))

    # Use auto-correlation out of 4 seconds (empirically set??)
    ac_window               = int(numpy.round(auto_correlation_size * fft_resolution))

    # Compute the autocorrelation
    x_corr                  = librosa.autocorrelate(onsets[mincol:maxcol], ac_window)

    # re-weight the autocorrelation by log-normal prior
    #   FIXME:  2013-01-25 08:55:40 by Brian McFee <brm2132@columbia.edu>
    #   this fails if ac_window > length of song   

    bpms                    = 60.0 * fft_resolution / (numpy.arange(1, ac_window+1))
    x_corr_weighting        = numpy.exp(-0.5 * ((numpy.log2(bpms) - numpy.log2(start_bpm)) / bpm_std)**2)

    # Compute the weighted autocorrelation
    x_corr                  = x_corr * x_corr_weighting

    # Get the local maximum of weighted correlation
    x_peaks                 = librosa.localmax(x_corr)

    # Zero out all peaks before the first negative
    x_peaks[:numpy.argmax(x_corr < 0)] = False

    # Find the largest (local) max
    start_period            = numpy.argmax(x_peaks * x_corr)

    # Choose the best peak out of .33, .5, 2, 3 * start_period
    candidate_periods       = numpy.multiply(start_period, [1.0/3, 1.0/2, 1.0, 2.0, 3.0]).astype(int)
    candidate_periods       = candidate_periods[candidate_periods < ac_window]

    best_period             = numpy.argmax(x_corr[candidate_periods])

    start_bpm               = 60.0 * fft_resolution / candidate_periods[best_period]

    return start_bpm


def onset_strength(y, sampling_rate=22050, window_length=512, hop_length=64, mel_channels=40, rising=True, htk=False):
    '''
    Adapted from McVicar, adapted from Ellis, etc...
    
    Extract onsets

    INPUT:
        y               = time-series waveform (t-by-1 vector)
        sampling_rate   = sampling rate of the input signal     | default: 8000
        window_length   = number of samples per frame           | default: 256      | = 32ms @ 8KHz
        hop_length      = offset between frames                 | default: 32       | = 40us @ 8KHz
        mel_channels    = number of Mel bins to use             | default: 40
        rising          = detect only rising edges of beats     | default: True
        htk             = use HTK mels instead of Slaney        | default: False


    OUTPUT:
        onset_envelope
    '''

    gain_threshold  = 80.0

    # First, compute mel spectrogram
    S   = librosa.melspectrogram(y, sampling_rate, window_length, hop_length, mel_channels, htk)

    # Convert to dBs
    S   = librosa.logamplitude(S)

    ### Only look at top 80 dB
    onsets  = numpy.maximum(S, S.max() - gain_threshold)

    ### Compute first difference
    onsets  = numpy.diff(onsets, n=1, axis=1)

    ### Discard negatives (decreasing amplitude)
    #   falling edges could also be useful segmentation cues
    #   to catch falling edges, replace max(0,D) with abs(D)
    if rising:
        onsets  = numpy.maximum(0.0, onsets)
    else:
        onsets  = onsets**2
        pass

    ### Average over mel bands
    onsets      = numpy.mean(onsets, axis=0)

    ### Filter with a difference operator
    onsets      = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onsets)

    ### Threshold at zero
    onsets   = numpy.maximum(0.0, onsets)

    ### Normalize by the maximum onset strength
    Onorm = numpy.max(onsets)
    if Onorm == 0:
        Onorm = 1.0
        pass

    return onsets / Onorm

def segment(X, k):
    '''
        Perform bottom-up temporal segmentation

        Input:
            X:  d-by-t  spectrogram (t frames)
            k:          number of segments to produce

        Output:
            C:  d-by-k  centroids (ordered temporall)
            N:          number of frames used by each centroid
            V:  d-by-k  variance (mean distortion) for each segment

    '''

    # Connect the temporal connectivity graph
    G = sklearn.feature_extraction.image.grid_to_graph(n_x=X.shape[1], n_y=1, n_z=1)

    # Instantiate the clustering object
    W = sklearn.cluster.Ward(n_clusters=k, connectivity=G)

    # Fit the model
    W.fit(X.T)

    # Instantiate output objects
    C = numpy.zeros( (X.shape[0], k) )
    V = numpy.zeros( (X.shape[0], k) )
    N = numpy.zeros(k, dtype=int)

    # Find the change points from the labels
    d = list(1 + numpy.nonzero(numpy.diff(W.labels_))[0].astype(int))

    # tack on the last frame as a change point
    d.append(X.shape[1])

    s = 0
    for (i, t) in enumerate(d):
        N[i]    = t - s
        C[:,i]  = numpy.mean(X[:,s:t], axis=1)
        V[:,i]  = numpy.var(X[:,s:t], axis=1)
        s       = t
        pass

    return (C, N, V)

