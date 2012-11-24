#!/usr/bin/env python
'''
CREATED:2012-11-05 14:38:03 by Brian McFee <brm2132@columbia.edu>

All things rhythmic go here

'''

import librosa
import numpy, scipy, scipy.signal

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
    duration_time           = 90.0
    upper_time_zone         = 90.0      # Find a better name for this
    bpm_std                 = 0.7

    fft_resolution          = numpy.float(sampling_rate) / hop_length

    # TODO:   2012-11-07 09:16:26 by Brian McFee <brm2132@columbia.edu>
    #  profile and optimize this

    # Get lower and upper bounds on beat strength vector?
    # not sure what these are for yet
    maxcol                  = min(numpy.round(upper_time_zone * fft_resolution), len(onset_strength)-1)
    mincol                  = max(0, maxcol - numpy.round(duration_time * fft_resolution))

    # Use auto-correlation out of 4 seconds (empirically set??)
    ac_max                  = int(numpy.round(auto_correlation_size * fft_resolution))

    # Find local maximum in global auto-correlation
    x_corr                  = librosa.autocorrelate(onset_strength[mincol:maxcol+1], ac_max)

    # re-weight the autocorrelation by log-normal prior
    bpms                    = 60.0 * fft_resolution / (numpy.arange(1, ac_max+1))
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
    candidate_periods       = numpy.multiply(start_period, [1.0/3, 1.0/2, 2.0, 3.0]).astype(int)
    candidate_periods       = candidate_periods[candidate_periods < ac_max]

    best_period             = numpy.argmax(x_corr[candidate_periods])

    start_bpm               = 60.0 * fft_resolution / numpy.minimum(start_period, candidate_periods[best_period])

    return start_bpm


def _beat_strength(y, sampling_rate=8000, window_length=256, hop_length=32, mel_channels=40, rising=True):
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


    OUTPUT:
        onset_envelope
    '''

    gain_threshold  = 80.0

    # STFT
    D   = librosa.stft(y, window_length, window_length, hop_length)

    ### Convert D to mel scale, discard phase
    M   = librosa.melfb(sampling_rate, window_length/2 + 1, mel_channels)
    D   = numpy.dot(M, numpy.abs(D))

    ### Convert to dB (log-amplitude, suppress zeros/infinitessimals)
    D   = 20.0 * numpy.log10(numpy.maximum(1e-10, D))

    ### Only look at top 80 dB
    D   = numpy.maximum(D, D.max() - gain_threshold)

    ### Compute first difference
    D   = numpy.diff(D, 1, 1)

    ### Discard negatives (decreasing amplitude)
    #   falling edges could also be useful segmentation cues
    #   to catch falling edges, replace max(0,D) with abs(D)
    if rising:
        D   = numpy.maximum(0.0, D)
    else:
        D   = numpy.abs(D)
        pass

    ### Average over mel bands
    D   = numpy.mean(D, 0)

    ### Filter with a difference operator
    D   = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], D)

    ### Normalize by the maximum onset strength
    return D / numpy.max(D)

def _recursive_beat_decomposition(onset, t_min=16, sigma=16):

    n           = len(onset)

    if n <= 3 * t_min:
        return numpy.array([])

    score       = onset - 1.0 / (2 * sigma**2) * (numpy.arange(len(onset)) - len(onset) / 2.0)**2

    # Forbid beats less than t_min long
    score[:t_min]   = -numpy.inf
    score[-t_min:]  = -numpy.inf

    # Find the middle beat
    mid_beat    = numpy.argmax(score)
    left_beats  = _recursive_beat_decomposition(onset[:(mid_beat-1)], t_min, sigma)
    right_beats = mid_beat + _recursive_beat_decomposition(onset[mid_beat:], t_min, sigma)

    return numpy.concatenate((left_beats, numpy.array([mid_beat]), right_beats), axis=0)

def frames_to_time(frames, sr=8000, hop_length=32):

    return frames * float(hop_length) / float(sr)
