#!/usr/bin/env python
'''
CREATED:2012-11-05 14:38:03 by Brian McFee <brm2132@columbia.edu>

All things rhythmic go here

'''

import librosa
import numpy, scipy, scipy.signal

def beat_track(y, sampling_rate=8000, start_bpm=120, tightness=0.9):

    # First, get the frame->beat strength profile
    [beat_strength, downsample, hop_length] = _beat_strength(y, sampling_rate)

    # Then, estimate bpm

    # Then, run the tracker

    return _beat_estimate(beat_strength, downsample, hop_length, start_bpm, tightness)

def _beat_estimate(beat_strength, sampling_rate, hop_length, start_bpm, tightness):

    auto_correlation_size   = 4.0
    duration_time           = 90.0
    upper_time_zone         = 90.0
    bpm_std                 = 0.7
    alpha                   = 0.8

    fft_resolution  = numpy.true_divide(sampling_rate, hop_length)

    # TODO:   2012-11-07 09:16:26 by Brian McFee <brm2132@columbia.edu>
    #  profile and optimize this

    # Get lower and upper bounds on beat strength vector?
    # not sure what these are for yet
    maxcol          = min(numpy.round(upper_time_zone * fft_resolution), length(beat_strength)-1)
    mincol          = max(0, maxcol - numpy.round(duration_time * fft_resolution))

    # Use auto-correlation out of 4 seconds (empirically set??)
    ac_max          = int(numpy.round(auto_correlation_size * fft_resolution))
    desired_max     = ac_max * 2 + 1

    cross_corr      = numpy.correlate(beat_strength[mincol:maxcol+1], beat_strength[mincol:maxcol+1], 'full')
    npad            = int((desired_max - cross_corr.shape[0])/2.0)
    cross_corr      = numpy.hstack([numpy.zeros(npad), cross_corr, numpy.zeros(npad)]) 

    # Find local maximum in global auto-correlation
    raw_cross_cor   = cross_corr[ac_max:(2*ax_max+1)]

    bpms            = 60.0 * numpy.true_divide(fft_resolution, numpy.arange(ac_max+1) + 0.1)
    pass

def _beat_strength(y, input_rate, window_length=256, hop_length=32, mel_channels=40, rising=True):

    sampling_rate   = 8000
    gain_threshold  = 80.0

    # Resample the audio
    l   = len(y) * numpy.true_divide(sampling_rate, input_rate)
    y   = scipy.signal.resample(y, l)

    # STFT
    D   = librosa.stft(y, window_length, window_length, hop_length)

    ### Convert D to mel scale, discard phase, suppress zeros/infinitessimals
    M   = librosa.melfb(sampling_rate, window_length/2 + 1, mel_channels)

    D   = numpy.maximum(1e-10, numpy.dot(M, numpy.abs(D)))

    ### Convert to dB
    D   = 20.0 * numpy.log10(D)

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

    return (D, sampling_rate, hop_length)
