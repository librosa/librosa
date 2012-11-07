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
    return

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

    #     TODO:   2012-11-07 11:54:12 by Brian McFee <brm2132@columbia.edu>
    # why the + 0.1 here?  .. stability on 0th point
    bpms                    = 60.0 * fft_resolution / (numpy.arange(ac_max) + 0.1)
    x_corr_weighting        = numpy.exp(-0.5 * ((numpy.log2(bpms) - numpy.log2(start_bpm)) / bpm_std)**2)

    # Compute the weighted autocorrelation
    x_corr                  = x_corr * x_corr_weighting

    # Get the local max
    x_peaks                 = librosa.localmax(x_corr)

    # Throw out any peaks in the initial down slope
    x_peaks[:numpy.min(numpy.nonzero(x_corr < 0))] = False

    # Find the largest local max
    start_period            = numpy.argmax(x_peaks * x_corr)

    # Choose the best peak out of .33, .5, 2, 3 * start_period
    candidate_periods       = numpy.multiply(start_period, [1.0/3, 1.0/2, 2.0, 3.0]).astype(int)
    candidate_periods       = candidate_periods[candidate_periods < ac_max]

    best_period             = numpy.argmax(x_corr[candidate_periods])
    start_period2           = candidate_periods[best_period]

    start_bpm               = 60.0 * fft_resolution / numpy.minimum(start_period, start_period2)
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

    return D
