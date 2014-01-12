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

def train_model( audio_dir, GT_dir, output_model_name='./chord_model.p' ):

  r'''Top-level function for training an HMM-based chord_chroma
        recognition system

    :usage:
        >>> librosa.chords.train_model( audio_dir, GT_dir, output_model_name )

    :parameters:

      - audio_dir              : string

        path to a directory of audio files

      - GT_dir                 : string

        path to a directory of chord files in Chris Harte's
        format:

              start end chord

        start and end are onset/offset times in seconds,
        chord is a chord label in the form rootnote:chordtype,
        with the possible exceptions 'N' (no chord, silence or noise)
        or 'X' (unknown chord)

      - output_model_name      : string

        where to pickle the model to (including extension)

      - outfile                : string

          path of file to write to (including extension). Defaults
          to './chord_model.p'

    :returns: 

      - HMM                    : sklearn.GaussianHMM

        A sklearn object,with fields:

           HMM.state_labels    - list of names of chords in GT,
                                 including 'N' and 'X'

           HMM.n_states        - number of actual chords in the 
                                 model, excluding 'X'. If 'X' appears
                                 in the ground truth, it will be stored
                                 as the last index:

                                 ( HMM.state_labels[-1] = 'X' )

                                 meaning that the variable HMM can be 
                                 used to quickly skip over 'X's in training

           HMM.Init            - (n_states,) tuple of initial distribution
                                 probabilities. Indexed by HMM.state_labels

           HMM.Trans           - (n_states, n_states) np.ndarray of state
                                 to state probabilities, indexed by 
                                 HMM.state_labels   

           HMM.Mu              - (12, n_states) np.ndarray 12-dimensional 
                                 mean vector for each chord, with columns 
                                 indexed by HMM.state_labels

           HMM.Sigma           - (n_states, 12, 12) np.ndarray of covariance
                                 matrices for each chord. First dimension is
                                 indexed by HMM.state_labels

         NOTE! All probabilities are in regular (non-log) form, as expected
         by sklearn.                          

    ''' 

  # Basic plan:
  #
  # 1) Get filenames and check consistency
  # 2) For each audio/GT pair
  #    compute chroma
  #    extract GT
  #    synchonrise chroma and GT
  # 3) Post-process chord labels
  # 4) Train model  
  # 5) Save in sklearn format
  # 6) save as pickle

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
  for f, gt in zip( audio_files[:1], GT_files[:1] ):  

    # extract training chroma
    full_audio_path = os.path.join( audio_dir, f )

    print '  Extracting chroma and chords for ' + full_audio_path

    chroma, beat_times = extract_training_chroma( full_audio_path )

    # read chord file
    chords, chord_start_end = read_chords( os.path.join( GT_dir, gt ) )

    # beat-synch the chroma and chords
    sampled_chords = sample_chords_beat( chords, chord_start_end, beat_times, no_chord='N' )

    # store
    Chromagrams.append( chroma )
    Beat_synch_chords.append( sampled_chords )

  # Post-process the chord labels:
  # they need to be enharmonically mapped (D# -> Eb)
  # and converted to ints for sklearn
  states, state_labels, n_states = process_chords( Beat_synch_chords )

  # Train hidden chain
  print '  Training model'
  Init, Trans = train_hidden( states, n_states )

  # Train observed
  Mu, Sigma = train_obs( Chromagrams, states, n_states )

  # Set up in sklearn's framework

  # Full covariance Gaussian emission HMM
  HMM = GaussianHMM(n_components     = n_states,
                    covariance_type  = 'full')

  # set trans, init, mu, sigma
  HMM.transmat_  = Trans
  HMM.startprob_ = Init
  HMM.means_     = Mu
  HMM.covars_    = Sigma

  # Additional state info
  HMM.state_labels = state_labels
  HMM.n_states = n_states

  # Save model
  print '  Saving model'
  pickle.dump( HMM, open( output_model_name, 'w' ) )      

  return HMM

def extract_training_chroma( audio_file, beat_nfft=4096, beat_hop=64, chroma_nfft=8192, chroma_hop=2048 ):

  """

  Extracts chroma for use in chords.train_model

  """

  # Basic steps:
  #
  # Beat track
  # HPSS
  # Loudness calculation
  # beat-synchronisation
  # Pitch summation
  # Range-normalisation

  # load audio
  y, sr = librosa.load( audio_file )

  # track beats
  bpm, beat_frames = librosa.beat.beat_track( y=y, sr=sr, n_fft=beat_nfft, 
                          hop_length=beat_hop, trim=False )

  # Need to beat-synchronise chroma, but not necessarily that case that
  # beat_nfft == chroma_nfft, beat_hop == chroma_hop
  # and in fact they definitely shouldn't be, since in the first case
  # we want good time resolution, in the second case we want good
  # frequency resolution
  # so, need to do beat_frames -> beat_times -> chroma_frames
  beat_times = librosa.core.frames_to_time( beat_frames, sr=sr, hop_length=beat_hop )

  chroma_beat_frames = librosa.core.time_to_frames(beat_times, sr=sr, hop_length=chroma_hop)  

  # compute spectrum
  S = librosa.stft( y, n_fft=chroma_nfft, hop_length=chroma_hop )

  # HPSS
  Harmonic, Percussive = librosa.decompose.hpss( S )

  # invert
  y_harmonic = librosa.istft( Harmonic )

  # Logspectrum
  # FIXME: libROSA's tuning module is currently not to be
  # trusted, set to 0.0
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

    Chroma[ i, : ] = np.sum( BS_chroma[ i : : 12, : ], axis=0 )  

  # Range normalise
  for t in range( Chroma.shape[ 1 ] ):

    mi = min( Chroma[ :, t ] )
    ma = max( Chroma[ :, t ] )

    if ( ma > mi ):

      Chroma[ :,t ] = ( Chroma[ :, t ] - mi ) / ( ma - mi )

    else:
    
      Chroma[ :, t ] = 0

  # Roll to be consistent with librosa chroma
  Chroma = np.roll( Chroma, 3, axis=0 )

  return Chroma, beat_times

def read_chords( chord_file ):


  """

  reads a chord file, returning the chord labels
  and the chord start/end times
 
  """

  # open, read
  chord_data = open( chord_file ).readlines()

  # sometimes reads an empty line
  chord_data = [c for c in chord_data if c != '\n']

  # Need 
  n_samples = len( chord_data )

  # Prepare annotations in a n_samples, 2 matrix
  annotation_sample_times = np.zeros( ( n_samples, 2 ) )
  annotations = []

  for iline, line in enumerate( chord_data ):

    line_data = line.strip().split()

    annotations.append( line_data[ 2 ] )

    annotation_sample_times[ iline, 0 ] = float( line_data[ 0 ] )

    annotation_sample_times[ iline, 1 ] = float( line_data[ 1 ] )
   
  return annotations, annotation_sample_times

def sample_chords_beat( annotations, annotation_sample_times, sample_times, no_chord='N' ):

  """

  Beat-synchronises chord annotations. 

    :parameters:

      - annotations              : list

        list of chord symbols from chords.read_chords

      - annotation_sample_times  : (number_samples, 2) np.ndarray
        
        matrix of chord start (first column) and end (second column)
        times in seconds from chords.read_chords

      - sample_times             : iterable of length number_windows
      
        times, in seconds when beats were detected. The resulting
        sampled chords will be of length number_windows + 1, since 
        we are tracking the most prevelant chord _between_ beats 
        ( number_windows - 1), but also will have the chord before
        the first and after the last beat ( + 2 )

      - no_chord                 : string
      
        symbol to use when no chord can be assigned. This can happen
        if the first beat is before the first chord symbol, or if 
        the audio is longer than the ground truth specifies. Defaults
        to 'N'

    :returns:
    
      - sampled                  : list

        list of length number_windows + 1 with the most prevelant 
        chord between each beat (or before first, after last beat)      

  """

  # plan:
  # 
  # 1) Initialse
  # 2) Fill the time before the first beat with no_chord
  # 3) Loop through annotation_sample_times and sample_times,
  #    assigning the current_chord whilst the beat window contains
  #    a single chord, or taking a majority vote if the beat window
  #    covers two or more chords
  # 4) Fill the time after the last beat with no_chord

  # 1) Initialise
  number_samples = annotation_sample_times.shape[ 0 ]

  number_windows = len( sample_times )

  # Initialise output list
  sampled = [ None ] * ( number_windows + 1 ) 

  # which row in the annotation we're in
  t_anns = 1           

  # the previous time
  t_prev_anns = 1     

  # which beat window we're in 
  t_sample = 1        

  # 2) until we reach the first chord symbol, store no_chord
  while ( sample_times[ t_sample - 1 ] < annotation_sample_times[ 0, 0 ] ):

    sampled[ t_sample - 1 ] = no_chord

    t_sample = t_sample + 1

  # Assure that t_sample falls in a chord region
  while ( annotation_sample_times[ t_anns - 1, 1 ] < sample_times[ t_sample - 1 ] ):

        t_anns = t_anns + 1
     
  # 3) go though all annotation rows and beat windows
  while ( t_sample <= number_windows and t_anns <= number_samples ):

    # Case 1: current beat is within current annotation start/end
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
        for j in range( t_prev_anns + 1 , t_anns ): 

          intervalC[ countInterval - 1 ] = annotation_sample_times[ j - 1, 1 ] - annotation_sample_times[ j - 1, 0 ] 

          countInterval = countInterval + 1                   
                
        # Last interval
        intervalC[ countInterval - 1 ] = sample_times[ t_sample - 1 ] - annotation_sample_times[ t_anns - 1, 0 ]     
                
        # Majority vote
        maxIndex = np.argmax( intervalC )

        # Store
        sampled[ t_sample - 1 ] = annotations[ t_prev_anns - 1 + maxIndex ]

        # update
        t_prev_anns = t_anns
        
      # B. This beat falls entirely within one chord label
      else:

        sampled[ t_sample - 1 ] = annotations[ t_anns - 1 ]

      # increase beat index
      t_sample = t_sample + 1
        
    # Case 2: increase annotation row 
    else:
      while ( t_anns <= number_samples ) and  ( annotation_sample_times[ t_anns - 1, 1 ] < sample_times[ t_sample - 1 ] ):
        t_anns = t_anns + 1
         
  # 4) If there are still samples left, assign no chord
  if t_sample <= number_windows:

    sampled[ t_sample - 1 : ] = [ no_chord ] * ( len( sampled ) - t_sample + 1 )
      
  # The last chord after final beat    
  if ( t_anns == number_samples ): 

    sampled[number_windows] = annotations[t_anns - 1]
  
  elif t_anns < number_samples:
  
    countInterval = 1                    
    intervalC = [ 0 ] * ( number_samples - t_anns + 1 )
    
    # Majority vote
    intervalC[ countInterval - 1 ] = annotation_sample_times[ t_anns - 1, 1 ] - sample_times[ number_windows - 1 ]
    countInterval = countInterval + 1
    
    for j in range( t_anns + 1, number_samples + 1 ):

      intervalC[ countInterval - 1 ] = annotation_sample_times[ j - 1, 1 ] - annotation_sample_times[ j - 1, 0 ]
      countInterval = countInterval + 1

    # get max
    maxIndex = np.argmax( intervalC )

    # and assign
    sampled[ number_windows ] = annotations[ t_anns - 1 + maxIndex ]

  # 4. return the annotation samples
  return sampled

def process_chords( chords ):

  """

  Converts a collection of chord symbols to state indices, 
  accounting for enharmonic equivalence. 

    :parameters:

      - chords              : list

        list of songs, each of which is a list of chord symbols
        (strings)

    :outputs
    
      - states              : list

        list of songs, each of which is a list of chord indices,
        indexed by...

      - state_labels        : list
      
        list of state labels, including 'X'

      - n_states            : int
      
        number of 'real' states in the model, excluding 'X'. 
        This parameter can be used to easily skip over 'X' in
        training      

  """
  n_songs = len( chords )

  # Make an enharmonic map. I've chose flats over sharps arbitrarily
  enharmonics = {'A#': 'Bb', 'B#': 'C', 'C#':'Db','D#':'Eb','E#':'F','F#':'Gb','G#':'Ab',  # Sharps to flats
                 'A':'A','B':'B','C':'C','D':'D','E':'E','F':'F','G':'G',                  # naturals
                 'Ab':'Ab','Bb':'Bb','Cb':'B','Db':'Db','Eb':'Eb','Fb':'E','Gb':'Gb',      # flats to flats
                 'N':'N', 'X':'X'}                                                         # others

  # Store output
  processed_chords = []

  # for each song
  for song_chords in chords:

    # temporary storage
    song_processed_chords = []

    # for each chord in this song
    for chord in song_chords:

      # find rootnote and chord type, by
      # splitting by ':'
      if ':' in chord:

        rootnote, chord_type = chord.split( ':' )

        chord_type = ':' + chord_type 

      else:

        rootnote = chord
        chord_type = ''
      
      # now, ensure no sharps in the rootnote
      song_processed_chords.append( enharmonics[ rootnote ] + chord_type )  
    
    # store this song
    processed_chords.append( song_processed_chords )
     
  # Get unique states
  state_labels = []
  for song in chords:

    for chord in song:

      if chord not in state_labels:

        state_labels.append( chord )

  # sort to make pretty, and also ensure 'X'
  # is at the end, if present at all
  state_labels.sort()

  # n_states
  n_states = len( state_labels )

  # don't count 'X' as a state
  if state_labels[ -1 ] == 'X':

    n_states = n_states - 1

  # Now map chord strings to chord indices
  states = []
  for song in chords:

    states.append( [ state_labels.index( chord ) for chord in song ] )

  return states, state_labels, n_states

def train_hidden( chords, n_states, no_chord='N' ):

  """

  Trains the hidden chain of an HMM from a list of 
  chord indices

    :parameters:

      - chords              : list

        list of songs, each of which is a list of chord indices

      - n_states            : int
      
        number of 'real' states in the model. Any state index 
        equal to or larger than n_states will be ignored in
        traininig 

    :returns:
    
      - Init                : (n_states,) tuple

        Initial distribution of chords

      - Trans               : ( n_states, n_states ) np.ndarray
        
        State transition probabilities         

  """

  # Initialise...Init
  Init = np.zeros( n_states )

  for ann in chords:

    # skip if equal or larger than n_states 
     if ann[ 0 ] < n_states:
        
       # increase count  
       Init[ ann[ 0 ] ] = Init[ ann[ 0 ] ] + 1
    
  # Initialise Trans
  Trans = np.zeros( ( n_states, n_states ) )

  for ann in chords: 

    # start at second state
    for ichord, chord2 in enumerate( ann[ 1 : ] ):
      
      # retrieve previous state
      chord1 = ann[ ichord - 1 ]

      # skip if equal or larger than n_states 
      if chord1 < n_states and chord2 < n_states:
          
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

  # Reshape for sklearn
  Init = Init.reshape((n_states,))

  return Init, Trans      

def train_obs( chroma, chords, n_states ):

  """

  Trains the observed chain of an HMM from a list of chroma
  and associated chords

    :parameters:

      - chroma              : list 

        list of chromagram feature matrices

      - chords              : list

        list of songs, each of which is a list of chord indices

      - n_states            : int
      
        number of 'real' states in the model. Any state index 
        equal to or larger than n_states will be ignored in
        traininig 

    :returns:
    
      - Mu                  : ( n_states, 12 ) np.ndarray of mean
                              pitch class energy for each chord 
                              and pitch class

      - Sigma               : ( n_states, 12, 12 ) np.ndarray of 
                              covariance matrices for each chord                        

    all probabilities are in normal (non-log) form   

  """
  
  # First, collect all chroma and chord indices 
  # into two big matrices

  # Find total number of frames. Is there a more
  # pythonic way of doing this?
  tmax = 0

  for ann, chrom in zip( chords, chroma ):

    tmax = tmax + len( ann )

  # Initialise big matrices
  all_anns = [ None ] * tmax
  all_chroma = np.zeros( ( 12, tmax ) )

  # fill these with all the chroma
  # and annotations
  t = 0

  for c, a in zip( chroma, chords ):

    all_anns[ t : t + len( a ) ] = a
    all_chroma[ :, t : t + len( a ) ] = c

    t = t + len(a)

  # OK.
  n_frames = len( all_anns )
  n_dims = all_chroma.shape[ 0 ]   
        
  # initialse Mu, Sigma      
  Mu = np.zeros( ( n_dims, n_states ) ) 

  Sigma = np.zeros( ( n_states, n_dims, n_dims ) )   
        
  # Note we go to range( n_states ), meaning that
  # we'll not train a model for 'X' if it is in the 
  # data      
  for chord in range( n_states ):

    # Collect the chroma for this chord
    chord_indices = [ ichord for ichord,ann in enumerate( all_anns ) if ann == chord ]  
    chord_chroma = all_chroma[ :, chord_indices ]

    # Take mean
    Mu[ :, chord ] = np.mean( chord_chroma, 1 )
       
    # Variance   
    sigma_prior = 0.01 * np.identity( n_dims )
    chord_chroma = np.dot( chord_chroma , chord_chroma.T ) / len( chord_indices )
    
    Sigma[chord, :] = chord_chroma -  np.dot( Mu[ :, chord ].reshape( n_dims, 1 ), Mu[ :, chord ].reshape( 1, n_dims ) ) + sigma_prior

  # Reshape for sklearn
  Mu = Mu.T

  return Mu, Sigma



