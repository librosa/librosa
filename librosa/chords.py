#!/usr/bin/env python
"""Chord estimation"""

# Add paths
import cPickle as pickle
from sklearn.hmm import GaussianHMM
import os
import librosa
import numpy as np

def load_model(model):
    r'''Loads an HMM-based chord estimation model from file

    :usage:
        >>> Model = librosa.chords.load_model( model )

    :parameters:
      - model          : string
          full path to .p extension HMM model, which is a dictionary
          named 'Chord_Parameters' with the following keys/values:

            'chord_names' - list of string chord names of length n_states
            'Trans'       - (n_states, n_states) np.ndarray chord transition matrix
            'Init'        - (n_states,) tuple of initial chord distribution
            'Mu'          - (n_states, 12) np.ndarray of mean pitch class energy for 
                            each chord (n_states) and pitch class (12)
            'Sigma'       - (n_states, 12, 12) np.ndarray covariance matrix for each 
                            chord 

          all probabilities should be in normal (non-log) form                                   

    :returns: 
      - Model : dictionary
          Model parameters as above
    '''	

    # Un-pickle the saved parameters
    with open(model, 'r') as f:
        Chord_Parameters = pickle.load(f)

    # Set up in sklearn's framework
    Chord_symbols = Chord_Parameters[ 'chord_names' ]

    # number of chords = number of symbols
    n_states = len( Chord_symbols )

    # assume full covariance
    HMM = GaussianHMM(n_components     = n_states,
	                  covariance_type  = 'full')

    # set trans, init, mu, sigma
    HMM.transmat_  = Chord_Parameters['Trans']
    HMM.startprob_ = Chord_Parameters['Init']
    HMM.means_     = Chord_Parameters['Mu']
    HMM.covars_    = Chord_Parameters['Sigma']   

    HMM.state_labels = Chord_symbols

    return HMM 

def predict_chords( chromagram, HMM ):   
    r'''Predicts chords from a chromagram and HMM

    :usage:
        >>> chords, likelihood = librosa.chords.predict_chords( chromagram, model )

    :parameters:
      - chromagram          : np.ndarray
          (12, tmax) chromagram representing pitch salience
          for each time

      - states              : list
          list of chord names 

      - HMM                 : sklearn.GaussianHMM
          HMM model with the following model parameters:
          HMM.transmat_
          HMM.startprob_
          HMM.means_
          HMM.covars_
          HMM.state_labels 

          for a description of the format for the HMM parameters, see:

          http://scikit-learn.org/stable/modules/generated/sklearn.hmm.GaussianHMM.html               

          HMM.state_labels should be a list of chord names (strings), in the order
          used internall by HMM.transmat_ etc. 

    :returns: 
      - chords              : list
        chord symbol list of length t_max

      - likelihood          : maximum chord likelihood  
    '''	 

    # Run HMM decoder (sklearn likes the chroma to be transposed)
    likelihood, Raw_Sequence = HMM.decode( chromagram.T, algorithm='viterbi')

    # Convert back to chord symbols
    chords = [ HMM.state_labels[ s ] for s in Raw_Sequence ]

    return chords, likelihood 

def write_chords( chords, start_times, end_times, outfile ):
    r'''Writes chord prediction to file

    :usage:
        >>> librosa.chords.write_chords( chords, times, method='beat' )

    :parameters:
      - chords              : list
        chord symbol list of length t_max
      - start_times         : np.ndarray or list
          array of chord start times in seconds
      - end_times         : np.ndarray or list
          array of chord end times in seconds
      - outfile             : string
          path of file to write to (including extension)

    :returns: 
      - None
    '''	 

    # only print when the chord changes
    current_chord = chords[ 0 ]
    t = 0.0

    with open(outfile, 'w') as f:
        for chord, start, end in zip( chords, start_times, end_times ):
            if chord != current_chord:
                # chord has changed
                f.write( ' '.join( [ str( t ), str ( end ), str( current_chord ), '\n' ]) )
                current_chord = chord
                t = end
        # write last chord
        f.write( ' '.join( [ str( t ), str ( end ), str( current_chord )]) )

def train_model( audio_dir, GT_dir, output_feature_dir, output_model_dir ):

  # Get filenames, checking for MacOSX BS
  audio_files = os.listdir( audio_dir )
  audio_files = [f for f in audio_files if f != '.DS_Store' ]

  GT_files = os.listdir( GT_dir )
  GT_files = [ f for f in GT_files if f != '.DS_Store' ]

  # check for consistencey
  n_audio = len( audio_files )
  n_GT = len( GT_files )

  if n_audio != n_GT:

    raise ValueError( 'different number of audio (' + str(n_audio) + ')' + ' and ground truth (' + str(n_GT) + ') files.')
  
  # Loop through zipped files
  for f, gt in zip( audio_files, GT_files ):

    # extract training chroma
    full_audio_path = os.path.join( audio_dir, f )

    print '  Extracting chroma for ' + full_audio_path

    extract_training_chroma( full_audio_path )

  # For each audio/GT pair:
  #   extract beats
  #   extract chroma
  #   assign a chord label to each chroma frame
  # Train model
  # pickle output

  return None

def extract_training_chroma( audio_file, beat_nfft=4096, beat_hop=64, chroma_nfft=8192, chroma_hop=2048 ):

  # load audio
  y, sr = librosa.load( audio_file )

  # track beats
  bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, n_fft=beat_nfft, 
                          hop_length=beat_hop, trim=False)

  # beat_frames -> beat_times -> chroma_frames
  beat_times = librosa.core.frames_to_time(beat_frames, sr=sr, hop_length=beat_hop)
  chroma_beat_frames = librosa.core.time_to_frames(beat_times, sr=sr, hop_length=chroma_hop)  

  # compute spectrum
  S = librosa.stft(y, n_fft=chroma_nfft, hop_length=chroma_hop)

  # HPSS
  Harmonic, Percussive = librosa.decompose.hpss(S)

  # invert
  y_harmonic = librosa.istft(Harmonic)

  # Logspectrum
  Raw_chroma = librosa.feature.logfsgram( y_harmonic, sr, 
             n_fft=chroma_nfft, hop_length=chroma_hop, 
              normalise_D=False, tuning=0.0)

  # beat-synch
  # if last chroma_beat_frame is the length of Raw_chroma, 
  # librosa.feature.sync will not append a 'last' frame, meaning
  # that there won't be n+1 frames for n beats.
  #
  # To counteract this, append a single empty chroma frame
  if chroma_beat_frames[ - 1 ] == Raw_chroma.shape[ 1 ]:

    Raw_chroma = np.hstack( (Raw_chroma, np.zeros( ( Raw_chroma.shape[0], 1 )  ) ) ) 

  # analagously if chroma_beat_frame[ 0 ] == 0 
  if chroma_beat_frames[ 0 ] == 0:

    Raw_chroma = np.hstack( ( np.zeros( ( Raw_chroma.shape[0], 1 ), Raw_chroma ) ) ) 

  # also, rounding errors mean that sometimes the last chroma 
  # beat frame is *longer than* the chromagram itself  
  keep_frames = chroma_beat_frames < Raw_chroma.shape[ 1 ]
  
  chroma_beat_frames = chroma_beat_frames[ keep_frames ]
  beat_times = beat_times[ keep_frames ]

  # Beat-sych
  BS_chroma = librosa.feature.sync(Raw_chroma, chroma_beat_frames, aggregate=np.median)

  # Loudness
  # Fold pitches
  # Range normalise

  import matplotlib.pylab as plt
  plt.imshow( BS_chroma, aspect='auto',interpolation='nearest' )
  plt.colorbar()
  plt.show()
  dsdfs  

  return None

def sample_chords_beat( chords, chord_times, beat_times, no_chord ):

  # copy
  return None  

def train_hidden( chords, n_chords ):

  # copy
  return None

def train_obs( chroma, chords, n_chords ):

  # copy
  return None




