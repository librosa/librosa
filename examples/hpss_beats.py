#!/usr/bin/env python
'''
CREATED:2013-02-12 16:33:40 by Brian McFee <brm2132@columbia.edu>

Beat tracking with HPSS filtering

Usage: ./hpss_beats.py [-h] input_audio.mp3 output_beats.csv
'''

import sys, argparse
import librosa

# Some magic number defaults, FFT window and hop length
N_FFT       = 2048

# We use a hop of 64 here so that the HPSS spectrogram input
# matches the default beat tracker parameters
HOP_LENGTH  = 64

def hpss_beats(input_file, output_csv):
    '''HPSS beat tracking
    
    :parameters:
      - input_file : str
          Path to input audio file (wav, mp3, m4a, flac, etc.)

      - output_file : str
          Path to save beat event timestamps as a CSV file
    '''

    # Load the file
    print 'Loading  ', input_file
    y, sr = librosa.load(input_file)

    # Get the spectrogram
    print 'Generating STFT ... '
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Do HPSS
    print 'Harmonic-percussive separation ... '
    harmonic, percussive = librosa.decompose.hpss(D)

    # Construct onset envelope from percussive component
    print 'Tracking beats on percussive spectrogram',

    S = librosa.feature.melspectrogram(S=percussive, sr=sr, n_mels=128)

    onsets = librosa.onset.onset_strength(S=librosa.logamplitude(S))

    # Track the beats
    tempo, beats = librosa.beat.beat_track( onsets=onsets, 
                                            sr=sr, 
                                            hop_length=HOP_LENGTH, 
                                            n_fft=N_FFT)

    # Save the output
    print 'Saving beats to ', output_csv
    librosa.output.frames_csv(output_csv, beats, sr=sr, hop_length=HOP_LENGTH)


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='librosa HPSS beat-tracking example')

    parser.add_argument(    'input_file',
                            action      =   'store',
                            help        =   'path to the input file (wav, mp3, etc)')

    parser.add_argument(    'output_file',
                            action      =   'store',
                            help        =   'path to the output file (csv of beat times)')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Run the beat tracker
    hpss_beats(parameters['input_file'], parameters['output_file'])
