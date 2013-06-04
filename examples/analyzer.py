#!/usr/bin/env python
'''Audio content analysis script

CREATED:2013-05-22 12:18:53 by Brian McFee <brm2132@columbia.edu>

Usage:

./analyzer.py song.mp3 analysis_output.json

'''

import sys
import ujson as json

import numpy as np
import scipy, scipy.signal
import librosa

HOP = 128
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

    
    S = librosa.feature.melspectrogram(y, sr,   n_fft=2048, 
                                                hop_length=HOP, 
                                                n_mels=64, 
                                                fmin=37, 
                                                fmax=8000)
    S = S / S.max()

    A['spectrogram'] = librosa.feature.sync(librosa.logamplitude(S**2), beats).T.tolist()

    # Let's make some beat-synchronous mfccs
    S = librosa.feature.melspectrogram(y, sr, hop_length=HOP)
    S = librosa.feature.mfcc(librosa.logamplitude(S))
    A['timbres'] = librosa.feature.sync(S, beats).T.tolist()

    # And some chroma
    S = np.abs(librosa.stft(y, hop_length=HOP))

    # Grab the harmonic component
    H = librosa.hpss.hpss_median(S, win_P=31, win_H=31, p=1.0)[0]
    A['chroma'] = librosa.feature.sync(librosa.feature.chromagram(H, sr),
                                        beats,
                                        aggregate=np.median).T.tolist()

    # Harmonicity: ratio of H::S averaged per frame
    A['harmonicity'] = librosa.feature.sync(np.mean(H / (S + (S==0)), axis=0, keepdims=True),
                                            beats,
                                            aggregate=np.max).flatten().tolist()


    # Relative loudness
    S = S / S.max()
    S = S**2

    A['loudness'] = librosa.feature.sync(np.max(librosa.logamplitude(S), 
                                                axis=0,
                                                keepdims=True), 
                                         beats, aggregate=np.max).flatten().tolist()

    # Subsample the signal for vis purposes
    A['signal'] = scipy.signal.decimate(y, len(y) / 1024, ftype='fir').tolist()

    return A

if __name__ == '__main__':
    A = analyze_file(sys.argv[1])
    with open(sys.argv[2], 'w') as f:
        json.dump(A, f)
        pass
