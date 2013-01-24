#!/usr/bin/env python
'''
CREATED:2012-11-05 14:38:03 by Brian McFee <brm2132@columbia.edu>

All things rhythmic go here

'''

import librosa
import numpy, scipy, scipy.signal
import sklearn, sklearn.cluster, sklearn.feature_extraction

def beat_track(y, input_rate=8000, start_bpm=120, tightness=0.9):

    # Zeroeth, resample the signal
    sampling_rate = 8000

    # Resample the audio
    if sampling_rate != input_rate:
        y   = scipy.signal.resample(y, len(y) * float(sampling_rate) / input_rate)
        pass

    # First, get the frame->beat strength profile
    onset_strength  = _beat_strength(y, sampling_rate)

    # Then, estimate bpm
    bpm = _beat_estimate_bpm(onset_strength, start_bpm=start_bpm)
    
    # Then, run the tracker
    beats = _beat_tracker(onset_strength, start_bpm=bpm)

    return (bpm, beats)



def _beat_tracker(onset_strength, start_bpm=120.0, sampling_rate=8000, hop_length=32, tightness=0.9, alpha=0.9):

    fft_resolution  = numpy.float(sampling_rate) / hop_length
    start_period    = int(round(60.0 * fft_resolution / start_bpm))
    period          = start_period

    # Smooth beat events with a gaussian window
    template        = numpy.exp(-0.5 * (numpy.arange(-period,(period+1)) / (period / hop_length))**2)

    # Convolve 
    localscore      = scipy.signal.convolve(onset_strength, template, 'same')
    max_localscore  = numpy.max(localscore)

    ### Initialize DP

    backlink        = numpy.zeros_like(localscore)
    cumscore        = numpy.zeros_like(localscore)


    # Search range for previous beat: number of samples forward/backward to look
    prev_beat_range = numpy.round(numpy.arange(-2 * period, -period/2 + 1))

    # TODO:   2012-11-07 15:25:30 by Brian McFee <brm2132@columbia.edu>
    #  make a better variable name for this

    # Make a score window, which begins biased toward start_bpm and skewed 
    txwt            = numpy.exp(-0.5 * (tightness * (numpy.log(-prev_beat_range) - numpy.log(period)))**2)

    # Are we on the first beat?
    first_beat      = True

    # Forward step
    for i in xrange(len(localscore)):
        time_range          = i + prev_beat_range

        # Are we reaching back before time 0?
        z_pad               = numpy.maximum(0, numpy.minimum(1 - time_range[0], len(prev_beat_range)))

        # Search over all possible predecessors and apply transition weighting
        score_candidates                = numpy.zeros_like(txwt)
        score_candidates[z_pad:]        = txwt[z_pad:] * cumscore[time_range[z_pad:]]

        # Find the best predecessor beat
        beat_location       = numpy.argmax(score_candidates)
        current_score       = score_candidates[beat_location]

        # Add the local score
        cumscore[i]         = alpha * current_score + (1-alpha) * localscore[i]

        # Special case the first onset.  Stop if the localscore is small

        if first_beat and localscore[i] < 0.1 * max_localscore:
            backlink[i]     = None
        else:
            backlink[i]     = time_range[beat_location]
            first_beat      = False
            pass
        pass

    #     TODO:   2012-11-07 16:30:23 by Brian McFee <brm2132@columbia.edu>
    # pick up here 

    ### Get the last beat
    maxes                   = librosa.localmax(cumscore)
    max_indices             = numpy.nonzero(maxes)[0]
    peak_scores             = cumscore[max_indices]
    # TODO:   2012-11-07 16:29:29 by Brian McFee <brm2132@columbia.edu>
    #   what is this sorcery?
    median_score            = numpy.median(peak_scores)
    bestendposs             = numpy.nonzero(cumscore * librosa.localmax(cumscore) > 0.5 * median_score)[0]

    # The last of these is the last beat (since score generally increases)
    bestendx                = numpy.max(bestendposs)

    b                       = [int(bestendx)]
    return []
    pass

def _beat_estimate_bpm(onset_strength, sampling_rate=8000, hop_length=32, start_bpm=120):

    auto_correlation_size   = 4.0
    sample_duration         = 90.0
    sample_end_time         = 90.0
    bpm_std                 = 0.7

    fft_resolution          = numpy.float(sampling_rate) / hop_length

    # Chop onsets to X[(upper_limit - duration):upper_limit], or as much as will fit
    maxcol                  = min(numpy.round(sample_end_time * fft_resolution), len(onset_strength)-1)
    mincol                  = max(0, maxcol - numpy.round(sample_duration * fft_resolution))

    # Use auto-correlation out of 4 seconds (empirically set??)
    ac_window               = int(numpy.round(auto_correlation_size * fft_resolution))

    # Compute the autocorrelation
    x_corr                  = librosa.autocorrelate(onset_strength[mincol:maxcol], ac_window)

    # re-weight the autocorrelation by log-normal prior
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


def _beat_strength(y, sampling_rate=8000, window_length=256, hop_length=32, mel_channels=40, rising=True, htk=False):
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
        spectrogram
    '''

    gain_threshold  = 80.0

    # First, compute mel spectrogram
    S   = librosa.melspectrogram(y, sampling_rate, window_length, hop_length, mel_channels, htk)

    # Convert to dBs
    S   = librosa.logamplitude(S)

    ### Only look at top 80 dB
    O   = numpy.maximum(S, S.max() - gain_threshold)

    ### Compute first difference
    O   = numpy.diff(O, n=1, axis=1)

    ### Discard negatives (decreasing amplitude)
    #   falling edges could also be useful segmentation cues
    #   to catch falling edges, replace max(0,D) with abs(D)
    if rising:
        O   = numpy.maximum(0.0, O)
    else:
        O = O**2
        pass

    ### Average over mel bands
    O   = numpy.mean(O, axis=0)

    ### Filter with a difference operator
    O   = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], O)

    ### Threshold at zero
    O   = numpy.maximum(0.0, O)

    ### Normalize by the maximum onset strength
    Onorm = numpy.max(O)
    if Onorm == 0:
        Onorm = 1.0
        pass
    return (O / Onorm, S)

def _recursive_beat_decomposition(onset, t_min=16, sigma=16):

    n           = len(onset)

    if n <= 3 * t_min:
        return numpy.array([], dtype=int)

    score       = onset - 1.0 / (2 * sigma**2) * (numpy.arange(len(onset)) - len(onset) / 2.0)**2

    # Forbid beats less than t_min long
    score[:t_min]   = -numpy.inf
    score[-t_min:]  = -numpy.inf

    # Find the middle beat
    mid_beat    = numpy.argmax(score)
    left_beats  = _recursive_beat_decomposition(onset[:(mid_beat-1)], t_min, sigma)
    right_beats = mid_beat + _recursive_beat_decomposition(onset[mid_beat:], t_min, sigma)

    return numpy.concatenate((left_beats, numpy.array([mid_beat], dtype=int), right_beats), axis=0)

def segment(X, k, variance=False):
    '''
        Perform bottom-up temporal segmentation

        Input:
            X:  d-by-t  spectrogram (t frames)
            k:          number of segments to produce

        Output:
            C:  d-by-k  centroids (ordered temporall)
            N:          number of frames used by each centroid

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

    if variance:
        return (C, N, V)
    
    return (C, N)

def segments_to_onsets(C, N):
    '''
    Input:
        C:      d-by-n  set of segment centroids
        N:      1-by-n  list of frame counts per segment
    Output:
        O:      1-by-t  indicator vector of onsets

    An onset is defined as a boundary between segments (t, t+1)
    where segment C(t+1) is louder (greater in magnitude) than C(t).

    '''
    # FIXME:  2013-01-09 14:03:46 by Brian McFee <brm2132@columbia.edu>
    # probably needs some smoothing     


    # Convert frame counts into raw frame numbers
    NT  = numpy.cumsum(N)
    O   = numpy.zeros(NT[-1], dtype=bool)

    # Shift segment centroids up to 0 baseline
    C   = C - C.min()

    # Compute RMSE per segment
    v = numpy.sum(C**2, axis=0)

    for (i, t) in enumerate(NT[1:-1], 1):
        if v[i] > v[i-1] and v[i] > v[i+1]:
            O[t]    = True
            pass
        pass
    return (O, v)



def frames_to_time(frames, sr=8000, hop_length=32):

    return frames * float(hop_length) / float(sr)
