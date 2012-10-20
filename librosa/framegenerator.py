#!/usr/bin/env python
'''
CREATED:2012-10-20 17:09:43 by Brian McFee <brm2132@columbia.edu>

Well-behaved wrapper to audioread

'''

import numpy
import audioread

def frames_timeseries(audio_blob, blocksize=512, zero_pad=True):
    '''
        audio_blob  = object returned from audioread.audio_open(...)
        blocksize   = size of the frames to return (default 512)
        zero_pad    = if true, last frame is padded out with 0s to blocksize
                        else, last frame is dropped

        iterates over frames and returns 
    '''

    if not isinstance(blocksize, int):
        raise TypeError('blocksize must be a positive integer')
    if blocksize < 0:
        raise ValueError('blocksize must be a positive integer')

    for frame in audio_blob.read_data(blocksize):

        # convert and renormalize buffer from PCM To real-valued
        x = numpy.frombuffer(frame, 'h') / 32768.0

        # is it a fractional frame?
        if len(x) < blocksize and zero_pad:
            y = numpy.zeros((blocksize,))
            y[:len(x)] = x
            yield y
        else:
            # otherwise, just return the data
            yield x
        pass
    pass
