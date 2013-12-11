#!/usr/bin/env python
'''
CREATED:2013-12-08 14:28:34 by Brian McFee <brm2132@columbia.edu>
 
Demonstration of phase vocoder time stretching 

Usage: ./time_stretch.py [-h] [-s STRETCH_FACTOR] input_file.mp3  output_harmonic.wav

'''

import sys, argparse
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

    N_FFT       = 2048
    HOP_LENGTH  = N_FFT /4

    # 1. Load the wav file, resample
    print 'Loading ', input_file

    y, sr = librosa.load(input_file)

    # 2. generate STFT @ 2048 samples
    print 'Computing short-time fourier transform... '
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)

    print 'Playing back at %3.f%% speed' % (speed * 100)
    D_stretch = librosa.phase_vocoder(D, speed, hop_length=HOP_LENGTH)

    y_stretch = librosa.istft(D_stretch, hop_length=HOP_LENGTH)

    print 'Saving stretched audio to: ', output_file
    librosa.output.write_wav(output_file, y_stretch, sr)

def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='librosa phase vocoder time stretching')

    parser.add_argument(    'input_file',
                            action      =   'store',
                            help        =   'path to the input file (wav, mp3, etc)')

    parser.add_argument(    'output_file',
                            action      =   'store',
                            help        =   'path to the stretched output (wav)')

    parser.add_argument(    '-s',
                            '--speed',
                            action      =   'store',
                            type        =   float,
                            default     =   2.0,
                            required    =   False,
                            help        =   'speed')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments()

    # Run the HPSS code
    stretch_demo(   parameters['input_file'], 
                    parameters['output_file'],
                    parameters['speed'])
