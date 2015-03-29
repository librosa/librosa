#!/usr/bin/env python
'''CREATED:2013-12-09 00:02:54 by Brian McFee <brm2132@columbia.edu>

Estimate the tuning (deviation from A440) of a recording.

Usage: ./tuning.py [-h] input_file
'''
from __future__ import print_function

import argparse
import sys
import librosa


def estimate_tuning(input_file):
    '''Load an audio file and estimate tuning (in cents)'''

    print('Loading ', input_file)
    y, sr = librosa.load(input_file)

    print('Separating harmonic component ... ')
    y_harm = librosa.effects.harmonic(y)

    print('Estimating tuning ... ')
    # Just track the pitches associated with high magnitude
    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)

    print('{:+0.2f} cents'.format(100 * tuning))


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='Tuning estimation example')

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments(sys.argv[1:])

    # Run the beat tracker
    estimate_tuning(parameters['input_file'])
