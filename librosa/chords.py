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
  
  # Loop through zipped files, storing chroma and 
  # beat-synched anns
  Chromagrams = []
  Beat_synch_chords = []
  for f, gt in zip( audio_files[:2], GT_files[:2] ):

    # extract training chroma
    full_audio_path = os.path.join( audio_dir, f )

    print '  Extracting chroma for ' + full_audio_path

    chroma, beat_times = extract_training_chroma( full_audio_path )

    # read chord file
    chords, chord_start_end = read_chords( os.path.join( GT_dir, gt ) )

    # beat-synch the chroma and chords
    sampled_chords = sample_chords_beat( chords, chord_start_end, beat_times, no_chord='N' )

    # append
    Chromagrams.append( chroma )
    Beat_synch_chords.append( sampled_chords )

  # Post-process the chord labels:
  # they need to be enharmonically mapped (Eb = D#)
  # and converted to ints
  states, state_labels = process_chords( Beat_synch_chords )

  # Train hidden
  Init, Trans = train_hidden( Beat_synch_chords )

  import matplotlib.pylab as plt
  plt.imshow( Trans, aspect='auto', interpolation='nearest')
  plt.show()
  gdffgd
  # pickle output

  return None

def extract_training_chroma( audio_file, beat_nfft=4096, beat_hop=64, chroma_nfft=8192, chroma_hop=2048 ):

  # load audio
  y, sr = librosa.load( audio_file, duration=15.0 )

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
  # I can't work out how to get the logfsgram frequencies to use
  # the perceptual weighting...just do the log10 myself
  BS_chroma = 10 * np.log10( ( BS_chroma + 10 ** ( -12 ) ) / ( 10 ** ( -6 ) ) )

  # Fold pitches
  Chroma = np.zeros( ( 12, BS_chroma.shape[ 1 ] ) )
  for i in range( 12 ):

    Chroma[ i, : ] = np.sum( BS_chroma[ i::12,: ], axis=0 )  

  # Range normalise
  for t in range( Chroma.shape[ 1 ] ):

    mi = min( Chroma[ :,t ] )
    ma = max( Chroma[ :, t] )

    if ( ma > mi ):

      Chroma[ :,t ] = ( Chroma[ :, t ] - mi ) / ( ma - mi )

    else:
    
      Chroma[ :, t ] = 0

  # Roll to be consistent with librosa chroma

  return Chroma, beat_times

def read_chords( chord_file ):

  chord_data = open( chord_file ).readlines()

  # sometimes reads an empty line
  chord_data = [c for c in chord_data if c != '\n']

  n_samples = len( chord_data )

  # Prepare annotations
  annotation_sample_times = np.zeros( ( n_samples, 2 ) )
  annotations = []

  for iline, line in enumerate( chord_data ):

    line_data = line.strip().split()

    annotations.append( line_data[ 2 ] )
    annotation_sample_times[ iline, 0 ] = float( line_data[ 0 ] )
    annotation_sample_times[ iline, 1 ] = float( line_data[ 1 ] )
   
  return annotations, annotation_sample_times

def sample_chords_beat(annotations, annotation_sample_times, sample_times, no_chord='N'):

  # 1. Initialisation 
  number_samples = annotation_sample_times.shape[ 0 ]

  number_windows = len(sample_times)

  sampled = [ None ] * ( number_windows + 1 ) # Store the annotations of a song

  t_anns = 1           # which label we're at in the unsampled anns
  t_prev_anns = 1      # 
  t_sample = 1         # which sampled window (ie the output) we're in

  # 1.1 For the first frame, if it is less than the start time, then no
  # chord
  while ( sample_times[ t_sample - 1 ] < annotation_sample_times[ 0, 0 ] ):

    sampled[ t_sample - 1 ] = numStates

    t_sample = t_sample + 1

  # 1.2 Assure that t_sample falls in a chord region
  while ( annotation_sample_times[ t_anns - 1, 1 ] < sample_times[ t_sample - 1 ] ):

        t_anns = t_anns + 1
     
  # 2. go though all time grid
  while ( t_sample <= number_windows and t_anns <= number_samples ):

    # 2.1 If TS(ts)<TA(ta), then this frame falls in this chord region
    if ( sample_times[ t_sample - 1 ] <= annotation_sample_times[ t_anns - 1, 1 ] ):

      # A. if the interval between two beats has more than 1 chord
      if ( t_prev_anns < t_anns ):

        # Majority vote
        countInterval = 1     

        intervalC = [ 0 ] * ( t_anns - t_prev_anns + 1 )
                
        # First interval
        if t_sample == 1:

          intervalC[ countInterval - 1 ] = annotation_sample_times[ t_prev_anns - 1, 1 ] - annotation_sample_times[ t_prev_anns - 1, 0 ]

        else:

          intervalC[ countInterval - 1 ] = annotation_sample_times[ t_prev_anns - 1, 1 ] - sample_times[ t_sample - 1 - 1 ]
                
        countInterval = countInterval + 1
                
        # Between intervals
        for j in range(t_prev_anns+1,t_anns):                      
          intervalC[countInterval - 1] = annotation_sample_times[ j - 1, 1 ] - annotation_sample_times[ j - 1, 0 ]                        
          countInterval = countInterval + 1                   
                
        # Last interval
        intervalC[countInterval - 1] = sample_times[t_sample - 1] - annotation_sample_times[ t_anns - 1, 0 ]     
                
        # Majority vote
        maxIndex = np.argmax( intervalC )

        #sampled[ t_sample - 1 ] = annotations[ t_prev_anns - 1 - 1 + maxIndex ]
        sampled[ t_sample - 1 ] = annotations[ t_prev_anns - 1 + maxIndex ]

        t_prev_anns = t_anns
        
      # B. if the interval between two beats falls in 1 chord
      else:

        sampled[t_sample - 1] = annotations[t_anns - 1]

      t_sample=t_sample + 1
        
    # 2.2 Else, find the chord interval this beat falls in
    else:
      while (t_anns <= number_samples and annotation_sample_times[ t_anns - 1, 1] < sample_times[t_sample - 1]):
        t_anns = t_anns + 1
         
  # 3. if there are still samples left, assign no chord
  if t_sample <= number_windows:
    sampled[ t_sample-1 : ] = [ numStates ] * ( len( sampled ) - t_sample + 1 )
      
  if (t_anns == number_samples): # The last chord after final beats

    sampled[number_windows] = annotations[t_anns - 1]
  
  elif t_anns < number_samples:
  
    countInterval = 1                    
    intervalC = [ 0 ] * ( number_samples - t_anns + 1 )
    
    # Majority vote
    intervalC[ countInterval - 1 ] = annotation_sample_times[ t_anns - 1, 1 ] - sample_times[ number_windows - 1 ]
    countInterval = countInterval + 1
    
    for j in range(t_anns + 1, number_samples + 1):

      intervalC[countInterval - 1] = annotation_sample_times[j - 1, 1] - annotation_sample_times[j - 1, 0]
      countInterval = countInterval + 1

    maxIndex = np.argmax(intervalC)
    sampled[number_windows] = annotations[t_anns - 1 + maxIndex]

  # 4. return the annotation samples
  return sampled

def process_chords( chords ):

  n_songs = len( chords )

  # Make an enharmonic map
  enharmonics = {'A#': 'Bb', 'B#': 'C', 'C#':'Db','D#':'Eb','E#':'F','F#':'Gb','G#':'Ab',  # Sharps to flats
                 'A':'A','B':'B','C':'C','D':'D','E':'E','F':'F','G':'G',                  # naturals
                 'Ab':'Ab','Bb':'Bb','Cb':'B','Db':'Db','Eb':'Eb','Fb':'E','Gb':'Gb',      # flats to flats
                 'N':'N', 'X':'X'} 

  processed_chords = []
  for song_chords in chords:

    song_processed_chords = []
    for chord in song_chords:

      if ':' in chord:
        rootnote, chord_type = chord.split( ':' )
        chord_type = ':' + chord_type 

      else:

        rootnote = chord
        chord_type = ''
      
      song_processed_chords.append( enharmonics[ rootnote ] + chord_type )  
    
    processed_chords.append( song_processed_chords )
     
  # Get unique states
  state_labels = []
  for song in chords:

    for chord in song:

      if chord not in state_labels:

        state_labels.append( chord )

  # sort ot make pretty
  state_labels.sort()

  # n_states
  n_states = len( state_labels )

  # don't count 'X' as a state
  if 'X' in state_labels:

    n_states = n_states - 1

  states = []
  for song in chords:

    states.append( [ state_labels.index( chord ) for chord in song ] )

  return states, state_labels

def train_hidden( chords, no_chord='N' ):

  # Init 
  Init = np.zeros( n_states )

  for ann in anns:

     if ann[ 0 ] == no_chord:

       pass
       
     else:
        
       Init[ ann[ 0 ] ] = Init[ ann[ 0 ] ] + 1
    
  # Initialise Trans
  Trans = np.zeros( ( n_states, n_states ) )

  for ann in anns: 

    for ichord, chord2 in enumerate( ann[ 1 : ] ):
      
      chord1 = ann[ ichord - 1 ]

      if chord1 == no_chord or chord2 == no_chord:

        pass

      else:
          
        Trans[ chord1, chord2 ] = Trans[ chord1, chord2 ] + 1.0
    
  # Pseudocounts
  Init = Init + 10 ** ( -6 )
  Trans = Trans + 10 ** ( -6 )

  # Normalise
  sum_init = sum( Init )

  if sum_init > 0:

    Init = Init / sum( Init )
    
  for i in range( n_states ):

    sum_trans = sum( Trans[ i, : ] )

    if sum_trans > 0:
      Trans[ i, : ] = Trans[ i, : ] / sum_trans  


  # sklearn
  Init = Init.reshape((n_states,))

  return Init, Trans      

def train_obs( chroma, chords, n_chords ):

  # copy
  return None




