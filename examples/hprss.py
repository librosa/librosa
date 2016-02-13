#!/usr/bin/env python
'''CREATED:2016-02-12 20:14:12 by CJ Carr <emperorcj@gmail.com>

Demonstration of harmonic-percussive-residual source separation
'''
from __future__ import print_function

import argparse
import sys
import librosa


def hprss_demo(input_file, output_harmonic, output_percussive, output_residual):
    '''HPRSS demo function.

    :parameters:
      - input_file : str
          path to input audio
      - output_harmonic : str
          path to save output harmonic (wav)
      - output_percussive : str
          path to save output harmonic (wav)
      - output_residual: str
          path to save output harmonic (wav)
    '''

    print('Loading ', input_file)
    y, sr = librosa.load(input_file)
    
    print ('Computing STFT... ')
    D = librosa.stft(y)
    
    print('Separating harmonics and percussives... ')
    H, P = librosa.decompose.hpss(D, margin=(2.0,3.0))

    print('Subtracting harmonic and percussive from STFT to get residual... ')
    R = D - (H + P)

    print('Computing ISTFTs... ')
    y_harmonic = librosa.core.istft(H)
    y_percussive = librosa.core.istft(P)
    y_residual = librosa.core.istft(R)

    print('Saving harmonic audio to: ', output_harmonic)
    librosa.output.write_wav(output_harmonic, y_harmonic, sr)

    print('Saving percussive audio to: ', output_percussive)
    librosa.output.write_wav(output_percussive, y_percussive, sr)

    print('Saving residual audio to: ', output_residual)
    librosa.output.write_wav(output_residual, y_residual, sr)


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='harmonic-percussive-residual example')

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')

    parser.add_argument('output_harmonic',
                        action='store',
                        help='path to the harmonic output (wav)')

    parser.add_argument('output_percussive',
                        action='store',
                        help='path to the percussive output (wav)')

    parser.add_argument('output_residual',
                        action='store',
                        help='path to the residual output (wav)')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments(sys.argv[1:])

    # Run the HPSS code
    hprss_demo(parameters['input_file'],
              parameters['output_harmonic'],
              parameters['output_percussive'],
              parameters['output_residual'])
