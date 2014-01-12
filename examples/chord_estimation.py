"""

This script is to demo chords.py

It extracts chroma and chord symbols from two directories
and extracts beat-synchronous chroma and chords for each
audio file / chord file found.

Then, it trains an HMM using the sklearn framework for 
every chord symbol found (except those marked 'X'),
displaying the parameters trained

Finally, it re-extracts chroma (rather inefficiently, but this
	is what will happen ) and
makes a prediction, showing a visualisation of the predicted
and actual chords, and the performance (%  beat frames 
	correctly identified )

Inputs (set below) audio_dir: directory of audio files, in any format
                              librosa can read

                   GT_dir   : directory of ground truth files, in Chris
                              Harte's format:

                                  start end chord

                              with any kind of whitespace delimitation,
                              chord labels in the form root:chord_type
                              (or 'N' or 'X')

                              For each audio example in audio_dir, there
                              must be exactly one GT in GT_dir with the 
                              same 'basename' (filename with no extension)

          model_output_dir  : where to save the pickled model

               show_params  : whether to show the trained HMM parameters 
                              ( True ) or not ( False )

               show_chords  : whether to show the predicted chords against
                              the ground truth ( True ) or not ( False )                             


"""

# ----------
# Parameters 
# ----------

# Set audio and GT directories
audio_dir = '/Users/mattmcvicar/Desktop/Work/LibROSA_chords/small_audio'
GT_dir = '/Users/mattmcvicar/Desktop/Work/LibROSA_chords/small_GT'

# Set model output name
model_output_dir = './examples/models/majmin.p'

# show HMM parameters after training?
show_params = True

# show chord predictions vs actual chords?
show_chords = True

# -----
# Setup
# -----

import sys
import os
sys.path.append(sys.path.append("/Users/mattmcvicar/Desktop/Work/audioread"))
sys.path.append(sys.path.append("/Users/mattmcvicar/Desktop/librosa"))

import librosa
import matplotlib.pylab as plt
import numpy as np

# -----------
# train model
# -----------

print ''
print '  Training model'
print '  --------------'

HMM = librosa.chords.train_model( audio_dir, GT_dir, model_output_dir )

# ------------------------
# Check out the parameters
# ------------------------
if show_params:

	plt.imshow( HMM.transmat_, aspect='auto', interpolation='nearest' )
	plt.xticks( range( HMM.n_states ), HMM.state_labels, rotation=45 )
	plt.yticks( range( HMM.n_states ), HMM.state_labels )
	plt.colorbar()
	plt.title( 'Transition Matrix' )
	plt.show()

	plt.imshow( HMM.means_.T , aspect='auto', interpolation='nearest' )
	plt.xticks( range( HMM.n_states ), HMM.state_labels, rotation=45 )
	plt.yticks( range( 12 ), ['A','','B','C','','D','','E','F','','G',''] )
	plt.colorbar()
	plt.title( 'Mean vectors' )
	plt.show()

	plt.imshow(HMM.covars_[ 1, :, : ], aspect='auto', interpolation='nearest')
	plt.colorbar()
	plt.title( 'Covariance Matrix for ' + HMM.state_labels[ 1 ] )
	plt.yticks( range( 12 ), ['A','','B','C','','D','','E','F','','G',''] )
	plt.xticks( range( 12 ), ['A','','B','C','','D','','E','F','','G',''] )
	plt.show()

# ----------
# Test model
# ----------

print ''
print '  Training model'
print '  --------------'

# Get filenames, checking for MacOSX BS
audio_files = os.listdir( audio_dir )
audio_files = [f for f in audio_files if f != '.DS_Store' ]

GT_files = os.listdir( GT_dir )
GT_files = [ f for f in GT_files if f != '.DS_Store' ]

for f, gt in zip( audio_files[:3], GT_files[:3] ):

  # extract training chroma
  full_audio_path = os.path.join( audio_dir, f )

  print '  Extracting chroma and chords for ' + full_audio_path

  chroma, beat_times = librosa.chords.extract_training_chroma( full_audio_path )

  # read chord file
  chords, chord_start_end = librosa.chords.read_chords( os.path.join( GT_dir, gt ) )

  # beat-synch the chroma and chords
  sampled_chords = librosa.chords.sample_chords_beat( chords, chord_start_end, beat_times, no_chord='N' )

  # Predict
  prediction, likelihood = librosa.chords.predict_chords( chroma, HMM )
  
  # Accuracy
  p = 100 * np.mean( [p == gt for p, gt in zip( prediction, sampled_chords ) ])

  if show_chords:

    # convert back to indices, so we can plot
    prediction_inds = [ HMM.state_labels.index( c ) for c in prediction ]
    true_inds = [ HMM.state_labels.index( c ) for c in sampled_chords ]

    plt.imshow( np.vstack( ( prediction_inds, true_inds ) ), aspect='auto', interpolation='nearest')
    plt.title(str(p) + '%')
    plt.yticks( [0,1], ['Prediction', 'Ground Truth'] )
    plt.show()

  print '  performance = ' + str(p) + '%'


