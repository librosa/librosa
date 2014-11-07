#!/usr/bin/env python
'''CREATED:2013-12-08 14:28:34 by Brian McFee <brm2132@columbia.edu>

Demonstration of harmonic-percussive source separation
'''
from __future__ import print_function

import argparse
import sys
import librosa


def hpss_demo(input_file, output_harmonic, output_percussive):
    '''HPSS demo function.

    :parameters:
      - input_file : str
          path to input audio
      - output_harmonic : str
          path to save output harmonic (wav)
      - output_percussive : str
          path to save output harmonic (wav)
    '''

    # 1. Load the wav file, resample
    print('Loading ', input_file)

    y, sr = librosa.load(input_file)

    # Separate components with the effects module
    print('Separating harmonics and percussives... ')
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # 5. Save the results
    print('Saving harmonic audio to: ', output_harmonic)
    librosa.output.write_wav(output_harmonic, y_harmonic, sr)

    print('Saving percussive audio to: ', output_percussive)
    librosa.output.write_wav(output_percussive, y_percussive, sr)


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='harmonic-percussive example')

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')

    parser.add_argument('output_harmonic',
                        action='store',
                        help='path to the harmonic output (wav)')

    parser.add_argument('output_percussive',
                        action='store',
                        help='path to the percussive output (wav)')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments(sys.argv[1:])

    # Run the HPSS code
    hpss_demo(parameters['input_file'],
              parameters['output_harmonic'],
              parameters['output_percussive'])
