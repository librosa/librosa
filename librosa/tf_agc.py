#!/usr/bin/env python
'''
CREATED:2012-10-20 12:20:33 by Brian McFee <brm2132@columbia.edu>

Time-frequency automatic gain control

Ported from tf_agc.m by DPWE
    http://www.ee.columbia.edu/~dpwe/resources/matlab/tf_agc/ 
'''

import numpy
import scipy
import _mfcc

def tf_agc(frame_iterator, sample_rate=22050, **kwargs):
    '''
    frame_iterator              iterates over audio frames, duhr
    sample_rate                 sampling rate of the  audio stream

    Optional arguments:
        frequency_scale     (f_scale from dpwe)     (default: 1.0)
        alpha                                       (default: 0.95 ) 
        gaussian_smoothing  (type)                  (default: False)
                                                    Use time-symmetric, non-causal Gaussian window smoothing        XXX: not supported
                                                    Otherwise, defaults to infinite-attack, exponential release
    '''

    ##
    # Parse arguments
    frequency_scale = 1.0
    if 'frequency_scale' in kwargs:
        frequency_scale = kwargs['frequency_scale']
        if not isinstance(frequency_scale, float):
            raise TypeError('Argument frequency_scale must be of type float')
        if frequency_scale <= 0.0:
            raise ValueError('Argument frequency_scale must be a non-negative float')
        pass
    

    alpha      = 0.95
    if 'alpha'     in kwargs:
        alpha      = kwargs['alpha']
        if not isinstance(alpha, float):
            raise TypeError('Argument alpha must be of type float')
        if not (0.0 < alpha < 1.0):
            raise ValueError('Argument alpha must be between 0.0 and 1.0')
        pass

    gaussian_smoothing = False
    if 'gaussian_smoothing' in kwargs:
        gaussian_smoothing = kwargs['gaussian_smoothing']
        if not isinstance(gaussian_smoothing, bool):
            raise TypeError('Argument gaussian_smoothing must be of type bool')
        if gaussian_smoothing:
            raise ValueError('Gaussian smoothing currently unsupported')
        pass


    # Setup

    # How many frequency bands do we need?
    num_frequency_bands = max(10, int(20 / frequency_scale))
    mel_filter_width    = int(frequency_scale  * num_frequency_bands / 10);

    # initialize the mel filterbank to None; 
    # do the real initialization after grabbing the first frame
    f2a = None

    if gaussian_smoothing:
        pass
    else:
        # Else, use infinite-attack exp-release


        # Iterate over frames
        for frame in frame_iterator:
            frame = numpy.frombuffer(frame, 'h')

            if f2a is None: 
                # initialize the mel filter bank after grabbing the first frame

                f2a = _mfcc.melfb(sample_rate, len(frame), num_frequency_bands, mel_filter_width)
#                 f2a = f2a[:,:(round(len(frame)/2) + 1)]
                

                #% map back to FFT grid, flatten bark loop gain
                #sf2a = sum(f2a);

                normalize_f2a                       = numpy.sum(f2a, axis=0)
                normalize_f2a[normalize_f2a == 0]   = 1.0
                normalize_f2a                       = 1.0 / normalize_f2a

                # initialze the state vector
                state   = numpy.zeros( (num_frequency_bands, 1) )[0]

                pass

            # FFT each frame
            D = scipy.fft(frame)

            # multiply by f2a
            if D.shape[0] != f2a.shape[1]:
                yield 0.0
                break
            audiogram = numpy.dot(f2a, numpy.abs(D))

            ## DPWE
            #             state = max([alpha*state,audgram(:,i)],[],2);
            #             fbg(:,i) = state;
            # ...
            #
            state = numpy.maximum(alpha * state, audiogram)

            #E = diag(1./(sf2a+(sf2a==0))) * f2a' * fbg;
            E   = normalize_f2a * numpy.dot(f2a.T, state);

            #% Remove any zeros in E (shouldn't be any, but who knows?)
            #E(E(:)<=0) = min(E(E(:)>0));
            E[E<=0] = min(E[E>0.0])

            #% invert back to waveform
            #y = istft(D./E);
            y = numpy.real(scipy.ifft(D/E))

            yield y
        pass

    pass
