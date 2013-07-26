def read_chord_params(filename):

  # Read chord parameters
  import re
  import numpy as np
  chord_params = open(filename,'r').readlines()

  # Get n classes from first line
  import re
  labels = re.split(',',chord_params[0])[:]
  n_labels = len(labels)

  # Read Init
  Init_str = re.split(',', chord_params[1])[:-1]
  Init = np.zeros(n_labels)
  for i, chord in enumerate(Init_str):
    Init[i] = float(chord)

  # Read trans
  Trans_str = chord_params[2: 2 + n_labels]
  Trans = np.zeros((n_labels, n_labels))
  for i,chord in enumerate(Trans_str):
    Trans_chord = re.split(',',chord)[:-1]
    for j, note in enumerate(Trans_chord):
      Trans[i,j] = float(note)

  # Read Mu
  Mu_str = chord_params[2 + n_labels: 2 + n_labels + 12]
  Mu = np.zeros((12, n_labels))
  for i,chord in enumerate(Mu_str):
    Mu_chord = re.split(',',chord)[:-1]
    for j, note in enumerate(Mu_chord):
      Mu[i,j] = float(note)

  # Read Sigma
  Sigma_str = chord_params[2 + n_labels + 12:]
  Sigma = np.zeros((n_labels, 12, 12))
  current_chord = -1
  for i,chord in enumerate(Sigma_str):

    if np.remainder(i,12) == 0:
      current_chord = current_chord + 1
  
    pitch1 = np.remainder(i,12)  
    Sigma_chord = re.split(',',chord)[:-1]
    for pitch2, sig in enumerate(Sigma_chord):
      Sigma[current_chord, pitch1, pitch2 ] = float(sig)

  return labels, Init, Trans, Mu, Sigma

def hmm_mixgauss_prob(obs,mu,Sigma):

  import numpy as np
  # 1. Invert Sigma
  ndim = Sigma.shape[1]
  nstates = Sigma.shape[0]
  sigma_inv = np.zeros(Sigma.shape)
  sigma_det_inv = np.zeros(nstates)
  pi2 = (2.0*np.pi)**(ndim)

  for chord in range(nstates):
    sig = Sigma[chord,:,:]    
    # A. If this chord appear in the training set
    if sig.any():
        sigma_inv[chord,:,:] = np.linalg.inv(sig)
        sigma_det_inv[chord] = np.log(1.0/(np.sqrt(pi2*np.linalg.det(sig))))
    # B. Else
    else:
        sigma_inv[chord,:,chord] = sig
        sigma_det_inv[chord] = -np.inf
        
  # 1. Configuration
  obs_len = obs.shape[1]
  nstates = mu.shape[1]
  emit = np.zeros((nstates,obs_len))

  # 2. Computing the gaussian probabilities
  for chord in range(nstates):
    tempObs = np.subtract(obs,np.tile(mu[:,chord],(obs_len,1)).T)
    for l in range(obs_len):
        temp = sigma_det_inv[chord]-0.5*np.dot(np.dot(tempObs[:,l],sigma_inv[chord,:,:]),tempObs[:,l])
        emit[chord,l] = temp
  return emit

def viterbiDecoder(init_states,transitM,emitM):

  import numpy as np
  # Pseudocounts
  init_states = init_states + 10**-6
  transitM = transitM + 10**-6
  
  # Log
  init_states = np.log(init_states)
  transitM = np.log(transitM)  
    
  # 1. configuration
  [Nseq,Nexample] = emitM.shape

  viterbiM = -np.inf*(np.zeros((Nseq,Nexample)) + 1.0) # path matrix
  pathM = -np.inf*(np.zeros((Nseq,Nexample)) + 1.0)    # the trace back index matrix

  viterbiM[:,0] = init_states # Get the initial probability
    
   # 2. filling the dynamic programing table
  for i in range(1,Nexample): # for each example
    for j in range(Nseq): # for each state of the current example
        maxVal = -np.inf
        maxIndex = -1.0
        # 2.1 loop through each previous state
        for k in range(Nseq): 
            tempVal = viterbiM[k,i-1] + transitM[k,j]
            if tempVal > maxVal:
                maxVal = tempVal
                maxIndex = k
    
        # 2.2 Get the path with the max prob
        viterbiM[j,i] = maxVal + emitM[j,i]
        pathM[j,i] = maxIndex
        
  # 3. Find the most probability path and output

  viterbiPath = np.zeros((Nexample))
  maxViterbiVal = np.max(viterbiM[:,-1])
  maxIndex = np.argmax(viterbiM[:,-1])
  viterbiPath[-1] = int(maxIndex)

  for i in range(Nexample-1,1,-1):
    maxIndex = pathM[maxIndex,i]
    viterbiPath[i-1] = int(maxIndex)

  return viterbiPath, maxViterbiVal

def write_chords(chords, times, file):

  # Strip any repeated chords
  write_chords = [chords[0]]
  write_times = [times[0]]

  for i, chord in enumerate(chords):
    if chord is not write_chords[-1]:
      write_chords.append(chord)
      write_times.append(times[i])

  # Write to file
  ID = open(file,'w')
  for i,chord in enumerate(write_chords):
    ID.write(str(write_times[i]) + ' ' + chord.rstrip() + '\n')
  ID.close()    