#!/usr/bin/env python
'''CREATED:2013-12-08 14:28:34 by Brian McFee <brm2132@columbia.edu>

Demonstration of phase vocoder time stretching.
'''
from __future__ import print_function

import argparse
import sys
import librosa


def stretch_demo(input_file, output_file, speed):
    '''Phase-vocoder time stretch demo function.

    :parameters:
      - input_file : str
          path to input audio
      - output_file : str
          path to save output (wav)
      - speed : float > 0
          speed up by this factor
    '''

    # 1. Load the wav file, resample
    print('Loading ', input_file)

    y, sr = librosa.load(input_file)

    # 2. Time-stretch through effects module
    print('Playing back at {:3.0f}% speed'.format(speed * 100))

    y_stretch = librosa.effects.time_stretch(y, speed)

    print('Saving stretched audio to: ', output_file)
    librosa.output.write_wav(output_file, y_stretch, sr)


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='Time stretching example')

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')

    parser.add_argument('output_file',
                        action='store',
                        help='path to the stretched output (wav)')

    parser.add_argument('-s', '--speed',
                        action='store',
                        type=float,
                        default=2.0,
                        required=False,
                        help='speed')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments(sys.argv[1:])

    # Run the HPSS code
    stretch_demo(parameters['input_file'],
                 parameters['output_file'],
                 parameters['speed'])
