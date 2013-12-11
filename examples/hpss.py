#!/usr/bin/env python
'''
CREATED:2013-12-08 14:28:34 by Brian McFee <brm2132@columbia.edu>
 
Demonstration of harmonic-percussive source separation

Usage:

./hpss.py  input_file.mp3  output_harmonic.wav  output_percussive.wav

'''

import sys, argparse
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

    N_FFT       = 2048
    HOP_LENGTH  = N_FFT /4

    # 1. Load the wav file, resample
    print 'Loading ', input_file

    y, sr = librosa.load(input_file)

    # 2. generate STFT @ 2048 samples
    print 'Computing short-time fourier transform... '
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 3. HPSS.  The default kernel size isn't necessarily optimal, but works okay enough
    print 'Separating harmonics and percussives... '
    harmonic, percussive = librosa.decompose.hpss(D)

    # 4. Invert STFT
    print 'Inverting harmonics and percussives... '
    y_harmonic   = librosa.istft(harmonic, hop_length=HOP_LENGTH)
    y_percussive = librosa.istft(percussive, hop_length=HOP_LENGTH)

    # 5. Save the results
    print 'Saving harmonic audio to: ', output_harmonic
    librosa.output.write_wav(output_harmonic, y_harmonic, sr)

    print 'Saving percussive audio to: ', output_percussive
    librosa.output.write_wav(output_percussive, y_percussive, sr)

def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='librosa harmonic-percussive source separation example')

    parser.add_argument(    'input_file',
                            action      =   'store',
                            help        =   'path to the input file (wav, mp3, etc)')

    parser.add_argument(    'output_harmonic',
                            action      =   'store',
                            help        =   'path to the harmonic output (wav)')

    parser.add_argument(    'output_percussive',
                            action      =   'store',
                            help        =   'path to the percussive output (wav)')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':
    # get the parameters
    parameters = process_arguments()

    # Run the HPSS code
    hpss_demo(  parameters['input_file'], 
                parameters['output_harmonic'],
                parameters['output_percussive'])
