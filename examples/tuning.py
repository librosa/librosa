#!/usr/bin/env python
'''
CREATED:2013-12-09 00:02:54 by Brian McFee <brm2132@columbia.edu>
 
Estimate the tuning (deviation from A440) of a recording.

Usage: ./tuning.py [-h] input_file
'''

import sys, argparse
import numpy as np

import librosa

def estimate_tuning(input_file):
    print 'Loading ', input_file
    y, sr = librosa.load(input_file)

    # Get the instantaneous-frequency pitch track
    print 'Tracking pitches... ',
    pitches, magnitudes, D = librosa.feature.ifptrack(  y, sr )

    print 'Estimating tuning ... ',
    # Just track the pitches associated with high magnitude
    pitches = pitches[magnitudes > np.median(magnitudes)]

    tuning = librosa.feature.estimate_tuning(pitches)
    print '%+0.2f cents' % (100 * tuning)


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='librosa tuning estimation example')

    parser.add_argument(    'input_file',
                            action      =   'store',
                            help        =   'path to the input file (wav, mp3, etc)')

    return vars(parser.parse_args(sys.argv[1:]))
   

if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Run the beat tracker
    estimate_tuning(parameters['input_file'])
