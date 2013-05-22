#!/usr/bin/env python
'''Audio content analysis script

CREATED:2013-05-22 12:18:53 by Brian McFee <brm2132@columbia.edu>

Usage:

./analyzer.py song.mp3 analysis_output.json

'''

import sys
import ujson as json

import numpy as np
import librosa

HOP = 64
SR  = 22050

def analyze_file(infile):
    '''Analyze an input audio file

    Arguments
    ---------
    infile  -- (str) path to input file


    Returns
    -------
    analysis -- (dict) of various useful things
    '''

    A = {}
    
    A['filename'] = infile

    y, sr = librosa.load(infile, sr=SR)
    
    # First, get the track duration
    A['duration'] = len(y) / sr

    # Then, get the beats
    tempo, beats = librosa.beat.beat_track(y, sr, hop_length=HOP)
    A['tempo'] = tempo
    A['beats'] = librosa.frames_to_time(beats, sr, hop_length=HOP).tolist()

    # Let's make some beat-synchronous mfccs
    S = librosa.feature.melspectrogram(y, sr, hop_length=HOP)
    S = librosa.feature.mfcc(librosa.logamplitude(S))
    A['timbres'] = librosa.feature.sync(S, beats).T.tolist()

    # And some chroma
    S = np.abs(librosa.stft(y, hop_length=HOP))
    A['pitches'] = librosa.feature.sync(librosa.feature.chromagram(S, sr),
                                        beats,
                                        aggregate=np.median).T.tolist()

    return A

if __name__ == '__main__':
    A = analyze_file(sys.argv[1])
    with open(sys.argv[2], 'w') as f:
        json.dump(A, f)
        pass
