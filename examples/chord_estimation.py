"""

This script is to demo chords.py

"""

import sys
sys.path.append(sys.path.append("/Users/mattmcvicar/Desktop/Work/audioread"))
sys.path.append(sys.path.append("/Users/mattmcvicar/Desktop/librosa"))

import librosa

# Set audio and GT directories
audio_dir = '/Users/mattmcvicar/Desktop/Work/LibROSA_chords/small_audio'
GT_dir = '/Users/mattmcvicar/Desktop/Work/LibROSA_chords/small_GT'

# Set model output name
model_output_dir = './examples/models/majmin.p'

# train model
Init, Trans, Mu, Sigma, state_labels = librosa.chords.train_model( audio_dir, GT_dir, model_output_dir )

n_states = len( state_labels )

# Check out the parameters
import matplotlib.pylab as plt
plt.imshow(Trans, aspect='auto', interpolation='nearest')
plt.xticks(range(n_states), state_labels, rotation=45)
plt.yticks(range(n_states), state_labels)
plt.colorbar()
plt.show()

plt.imshow(Mu.T, aspect='auto', interpolation='nearest')
plt.xticks(range(n_states), state_labels, rotation=45)
plt.yticks(range(12),['A','','B','C','','D','','E','F','','G',''])
plt.colorbar()
plt.show()

plt.imshow(Sigma[1,:,:], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.show()
