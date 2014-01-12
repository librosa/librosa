"""

This script is to demo chords.py

"""

import sys
sys.path.append(sys.path.append("/Users/mattmcvicar/Desktop/Work/audioread"))
sys.path.append(sys.path.append("/Users/mattmcvicar/Desktop/librosa"))

import librosa

# Set audio and GT directories
audio_dir = '/Users/mattmcvicar/Desktop/Work/LibROSA_chords/small_GT'
GT_dir = '/Users/mattmcvicar/Desktop/Work/LibROSA_chords/small_audio'

# Set chroma/beat time output dir
chroma_output_dir = './chroma_beat_output'

# Set model output name
model_output_dir = './models/majmin.p'

# train model
librosa.chords.train_model( audio_dir, GT_dir, 
	             chroma_output_dir, model_output_dir )
