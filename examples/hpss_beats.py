#!/usr/bin/env python
'''
CREATED:2013-02-12 16:33:40 by Brian McFee <brm2132@columbia.edu>

Beat tracking with HPSS filtering

Usage: ./hpss_beats.py [-h] input_audio.mp3 output_beats.csv
'''
from __future__ import print_function

import argparse
import numpy as np
import sys
import librosa

# Some magic number defaults, FFT window and hop length
N_FFT = 2048

# We use a hop of 512 here so that the HPSS spectrogram input
# matches the default beat tracker parameters
HOP_LENGTH = 512


def hpss_beats(input_file, output_csv):
    '''HPSS beat tracking

    :parameters:
      - input_file : str
          Path to input audio file (wav, mp3, m4a, flac, etc.)

      - output_file : str
          Path to save beat event timestamps as a CSV file
    '''

    # Load the file
    print('Loading  ', input_file)
    y, sr = librosa.load(input_file)

    # Do HPSS
    print('Harmonic-percussive separation ... ')
    y = librosa.effects.percussive(y)

    # Construct onset envelope from percussive component
    print('Tracking beats on percussive component')
    onset_env = librosa.onset.onset_strength(y=y,
                                             sr=sr,
                                             hop_length=HOP_LENGTH,
                                             n_fft=N_FFT,
                                             aggregate=np.median)

    # Track the beats
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
                                           sr=sr,
                                           hop_length=HOP_LENGTH)

    beat_times = librosa.frames_to_time(beats,
                                        sr=sr,
                                        hop_length=HOP_LENGTH)

    # Save the output
    print('Saving beats to ', output_csv)
    librosa.output.times_csv(output_csv, beat_times)


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='HPSS beat-tracking example')

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')

    parser.add_argument('output_file',
                        action='store',
                        help='path to the output file (csv of beat times)')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments(sys.argv[1:])

    # Run the beat tracker
    hpss_beats(parameters['input_file'], parameters['output_file'])
