def extract_harmony_from_audio(wavfilename,display=1,resample_rate=11025,tuning_fftlen=4096,f_ctr=400.0,f_sd=1.0,HPSS_fftlen=1024,HPSS_hoplen=512,HPSS_window=4096,gamma=0.3,alpha=0.3,kmax=1,write_wav=0,startbpm=120,beat_tightness=3):

  # Main function for extractnig harmony from audio. Main steps:
  # 1. Read wave file
  # 2. Collapse to mono
  # 3. Resample 
  # 4. Estimate tuning
  # 5. Split Harmony and Percussion
  # 6. Estimate beats
  # Returns harmony (the harmony wave), the estimated tuning, the new sample
  # rate, and the the estimated beat times
  
  # Default options below
  
  ##### 1. Read wav file #####
  if display == 1: print "Reading wave..."; 
  import scipy.io.wavfile as wavfile
  audio_data = wavfile.read(wavfilename)

  # Extract parts
  import numpy as np
  sr = audio_data[0];
  x = audio_data[1];
  n_channels = np.ndim(x);

  ##### 2. Collapse to mono and normalise #####
  if display == 1: print "Converting to mono and normalising...";   
  if n_channels > 1:
    x = np.mean(x,1)
  x = np.true_divide(x,2**15)
  
  ##### 3. Resample #####
  if display == 1: print "Resampling...";
  from scipy import signal
  if sr != resample_rate:
      l = len(x)*np.true_divide(resample_rate,sr);
      x = signal.resample(x,l);

  ##### 4. Estimate tuning #####
  if display == 1: print "Estimating tuning...";
  import HPA_tuning as tuning   
  semisoff = tuning.estimate_tuning(x,resample_rate,tuning_fftlen,f_ctr,f_sd);
  
  ##### 5. Do HPSS #####
  if display == 1: print "Splitting harmony and Percussion...";     
  #[xharmony,xperc] = HPSS(x,HPSS_fftlen,HPSS_hoplen,HPSS_window,gamma,alpha,kmax);  
  xharmony = x; 
 
  ###### 6. Write the wav? ######
  if write_wav == 1:
      print "Writing harmony wav..."; 
      write_wave(xharmony,'test.wav',resample_rate,1);
  
  #### 6. Estimate beats #####
  if display == 1: print "Estimating beat times...";
  import HPA_beat_tracker as beats
  reload(beats)
  beat_times = beats.estimate_beats(xharmony,resample_rate,startbpm,beat_tightness);


  return xharmony, semisoff, resample_rate, beat_times

def cal_hamming_window(SR, minFreq=220.0, maxFreq=1661.0, resolution_fact=5.0,tuning=0.0):

 # Function:
 # [hammingK,half_winLenK,freqBins]=cal_hamming_window(SR, minFreq, maxFreq, resolution_fact,tuning)
 # This function computes the hamming window for the constant Q transformation.
 # 
 #  INPUTS
 # SR - the sample rate
 # minFreq - the lowest frequency (e.g. 55HZ)
 # maxFreq - the highest frequency (e.g. 1661HZ)
 # resolution_fact - the resolution of Q (Q_new=Q_org*resolution)
 # tuning - tuning parameter, fk=fk*2^(tuning/bins)
 # 
 #  OUTPUTS
 # hammingK - the hamming windows for each frequency
 # half_winLenK - the half_length of each hamming window k
 # freqBins - the frequency of each hamming window
 # 
 # ---------------------------------------------
 # Function created by Y. Ni
 # Function revised by M. McVicar
 # Intelligent Systems Lab
 # University of Bristol
 # U.K.
 # 2011

 # 1. Configulation
 import numpy as np
 bins=12;
 pitchClass=12;
 pitchInterval = int(np.true_divide(bins,pitchClass));
 pitchIntervalMap = np.zeros(bins);

 for i in range(pitchClass):
   pitchIntervalMap[(i-1)*pitchInterval+1:i*pitchInterval+1] = int(i+1);
   
 # 2. Frequency bins
 K = int(np.ceil(np.log2(maxFreq/minFreq))*bins);  #The number of bins
 freqBins = np.zeros(K);

 #semiInterval = np.hstack([-bins_fact/bins, 0, bins_fact/bins]);

 for i in range(0,K-pitchInterval+1,pitchInterval):
   octaveIndex = np.floor(np.true_divide(i,bins));
   binIndex = np.mod(i,bins);
   val = minFreq*2.0**(octaveIndex+(pitchIntervalMap[binIndex]-1.0)/pitchClass);
   freqBins[i:i+pitchInterval+1] = val;  

 # Augment using tuning factor
 freqBins = freqBins*2.0**(tuning/bins);

 # 3. Constant Q and window size
 Q = 1.0/(2.0**(1.0/bins)-1)*resolution_fact;
 winLenK = np.ceil(SR*np.true_divide(Q,freqBins));

 # 4. Construct the hamming window
 half_winLenK = winLenK;
 const = 1j*-2.0*np.pi*Q
 expFactor = np.multiply(const,range(int(winLenK[0])+1));
 expFactor = np.conj(expFactor)
 hamming_k = list()
 for k in range(K):
   N = int(winLenK[k])
   half_winLenK[k] = int(np.ceil(N/2.0))
   hamming_k.append(np.hamming(N)* np.true_divide(np.exp(np.true_divide(expFactor[range(N)],N)),N));

 return hamming_k, half_winLenK, freqBins
 
def cal_CQ_chroma_loudness(x,SR, beat_times, hammingK, half_winLenK, freqK, refLabel='s', A_weightLabel=1,q_value=0,normFlag='n'):

 # Function:
 # 
 # [chromagram,normal_chromagram,sample_times]=cal_CQ_chroma_loudness(x,SR,...
 # beat_times, hammingK, half_winLenK, freqK, refLabel, A_weightLabel,q_value,normFlag)
 # 
 # Main function to compute the loudness based beat-sync (or fixed length) chromagram features
 # 
 #  INPUTS
 # x - the wave signal (from wavread)
 # SR - the sample rate 
 # beat_time - the beat time (unit: second)
 # hammingK - the hamming windows for each frequency
 # half_winLenK - the half_length of each hamming window k
 # freqK - the k'th frequency
 # (Computed by .m function: cal_hamming_window)
 # refLabel - the reference label, 'n': no reference (i.e. 1); 
 #                                 's':the standard human perceivable level, 10^(-12)
 #                                 'mean':related to the average loudness of the song (i.e. ref=mean(power sequence)); 
 #                                 'median':related to the average loudness of the song (i.e. ref=median(power sequence));
 #                                 'q':using the q-quantile power as the reference (need 0<=q_value<=1) 
 #                                 (i.e. regarding q_value percent of the frames as slience) 
 # A_weightLabel - 1:using A weighting; 0:otherwise
 # Remark: A-weighting is a scaling curve on the amplitude spectrum, see
 # http://en.wikipedia.org/wiki/A-weighting
 # q_value - used in q-quantile reference
 # normFlag - 'n': normal features and normalization used is: frame=(frame-min(frame))/(max(frame)-min(frame))
 #            's': shift version, loudness=10*log10(power/ref+1);
 #            normalization used is L1 norm: frame=frame/max(frame)
 # 
 #  OUTPUTS
 # chromagram - the loudness based chromagram features (length Nb+1)
 # normal_chromagram - the normalized loudness chromagram features (length Nb+1)
 # sample_times - the sample times for each frame, formated as a Nx2 matrix [start_times,end_times]
 # 
 # ---------------------------------------------
 # Function created by Y. Ni
 # Function revised by M. McVicar
 # Intelligent Systems Lab
 # University of Bristol
 # U.K.
 # 2011
 # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 # 1. configulation
 import numpy as np
 bins = 12;
 Nxorig = len(x);
 x = np.hstack([x,np.zeros(2.0**np.ceil(np.log2(Nxorig))-Nxorig)]); # Add the end to make the length to be 2^N 
 Nx = len(x);
 K = len(hammingK);                             #The number of frequency bins
 xf = np.fft.fft(x);

 # special: check whether hamming window length is > length(xf)%%%%%%%%%%%%%
 warningFlag = np.zeros(K);
 for k in range(K):
   if (len(hammingK[k])>Nx):
     print('In function cal_CQ_chroma_loudness: certain hamming winow is wider than the music itself.');
     warningFlag[k]=1;

 # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 # %1.1 The beat-time interva
 beatSR = np.ceil(np.multiply(beat_times,SR));  # Get the beat time (transform it into sample indices)
 beatSR = np.delete(beatSR, np.nonzero(beatSR>=Nxorig))   #delete those samples that have exceeded the end of the song
 
 if beatSR[0] is 0:
   beatSR = np.hstack([beatSR, Nxorig]);
 else:
   beatSR = np.hstack([0.0, beatSR, Nxorig]);

 numF=len(beatSR)-1;
 
 # %1.2 reference label
 if refLabel is 'n':
  refPower = 1;
 elif refLabel is 's':
   refPower=10.0**(-12.0)
 elif refLabel is 'mean':
   meanPowerK = np.zeros(K); 
 elif refLabel is 'median':
   medianPowerK = np.zeros(K)
 elif refLabel is 'q':
   quantile_matrix = np.zeros(Nxorig);  # Store the average power of each frame
   if (q_value<0.0 or q_value>1.0):
     print('Error in cal_CQ_chroma_loudness: the q value exceeds boundaries.');
 else:
   print('Error in cal_CQ_chroma_loudness: can not find the reference label.');

 # %1.3 A-weight parameters
 if A_weightLabel is 1:
   Ap1 = 12200.0**2.0;
   Ap2 = 20.6**2.0;
   Ap3 = 107.7**2.0;
   Ap4 = 737.9**2.0;

 # %2. Compute the CQ matrix for each point (row) and each frequency bin (column)
 A_offsets = np.zeros(K);
 CQ = np.zeros([K,numF]);
 if normFlag is 's': #shifted chromagram
   for k in range(K):
     # 2.1. Get the constant Q tranformation
     half_len = int(half_winLenK[k])
     w = np.hstack([hammingK[k][half_len-1:], np.zeros(Nx-len(hammingK[k])), hammingK[k][:half_len-1]]); 
     wf = np.fft.fft(w);
     if warningFlag[k] is 1:
       xft = np.hstack([xf,np.zeros(len(wf)-len(xf))]);
       convolf = xft*wf;
     else:
       convolf = xf*wf;   
     
     convol = np.fft.ifft(convolf);  
     
     # 2.2. A-weighting
     if A_weightLabel is 1:
       frequency_k2 = freqK[k]**2.0;
       A_scale = Ap1*frequency_k2**2.0/((frequency_k2+Ap2)*np.sqrt((frequency_k2+Ap3)*(frequency_k2+Ap4))*(frequency_k2+Ap1));
       A_offsets[k] = 2.0+20.0*np.log10(A_scale);
        
     # 2.3. Reference power and A weighting
     if refLabel is 'mean':
       convol = np.abs(convol[:Nxorig])**2.0;
       meanPowerK[k] = np.mean(convol);
     elif refLabel is 'median':
       convol = np.abs(convol[:Nxorig])**2.0;
       medianPowerK[k] = np.median(convol);
     elif refLabel is 'q':
       convol = np.abs(convol[:Nxorig])**2.0;
       quantile_matrix = np.add(quantile_matrix,convol);
     else:
       convol = (np.abs(convol[:Nxorig]))**2.0;
        
     # 2.4. Get the beat interval (median)
     for t in range(numF):
       t1 = int(beatSR[t])+1
       t2 = int(beatSR[t+1])
       CQ[k,t] = np.median(convol[t1-1:t2]);     
   
   # 2.5 Add the reference power (for mean/median/q-quantiles)
   if refLabel is 'mean':
     refPower = np.mean(meanPowerK);
     temp_one = 10.0*np.log10(np.add(refPower,CQ))
     temp_two = -10*np.log10(refPower)
     temp_three = np.transpose(np.tile(A_offsets,(numF,1)));
     CQ = np.add(np.add(temp_one,temp_two),temp_three)
   elif refLabel is 'median':
     refPower = np.median(medianPowerK);
     temp_one = 10.0*np.log10(np.add(refPower,CQ))
     temp_two = -10*np.log10(refPower)
     temp_three = np.transpose(np.tile(A_offsets,(numF,1)));
     CQ = np.add(np.add(temp_one,temp_two),temp_three)
   elif refLabel is 'q':
     quantile_value = np.sort(quantile_matrix); # sort the values, set reference as the value that falls in the q_value%-th of examples   
     refPower = quantile_value[int(np.floor(q_value*Nxorig))-1]/K;
     temp_one = 10.0*np.log10(np.add(refPower,CQ))
     temp_two = -10*np.log10(refPower)
     temp_three = np.transpose(np.tile(A_offsets,(numF,1)));
     CQ = np.add(np.add(temp_one,temp_two),temp_three)
   else:
     temp_one = 10.0*np.log10(np.add(refPower,CQ))
     temp_two = -10*np.log10(refPower)
     temp_three = np.transpose(np.tile(A_offsets,(numF,1)));
     CQ = np.add(np.add(temp_one,temp_two),temp_three)
   
   # 3. Computer the beat sync-ed chromagram
   chromagram = np.zeros((bins,numF));
   normal_chromagram = np.zeros((bins,numF));
    
   for i in range(bins):
     chromagram[i,:] = np.sum(CQ[i::bins,:],0);
    
   # 4. Compute the normalized chromagram features (L1 norm)
   for i in range(chromagram.shape[1]):
     temp_max = np.max(chromagram[:,i]);
     if (temp_max>0.0):
       normal_chromagram[:,i] = np.true_divide(chromagram[:,i],temp_max);
  
 elif normFlag is 'n': #normal chromagram
   for k in range(K):
     # 2.1. Get the constant Q tranformation
     half_len = int(half_winLenK[k])
     w = np.hstack([hammingK[k][half_len-1:], np.zeros(Nx-len(hammingK[k])), hammingK[k][:half_len-1]]); 
     wf = np.fft.fft(w);
     convolf = xf*wf;   
     convol = np.fft.ifft(convolf);
        
     # 2.2. A-weighting
     if A_weightLabel is 1:
       frequency_k2 = freqK[k]**2.0;
       A_scale = Ap1*frequency_k2**2.0/((frequency_k2+Ap2)*np.sqrt((frequency_k2+Ap3)*(frequency_k2+Ap4))*(frequency_k2+Ap1));
       A_offsets[k] = 2.0+20.0*np.log10(A_scale);

        
     # 2.3. Reference power and A weighting
     if refLabel is 'mean':
       convol = np.abs(convol[:Nxorig])**2.0;
       meanPowerK[k] = np.mean(convol);
     elif refLabel is 'median':
       convol = np.abs(convol[:Nxorig])**2.0;
       medianPowerK[k] = np.median(convol);
     elif refLabel is 'q':
       convol = np.abs(convol[:Nxorig])**2.0;
       quantile_matrix = np.add(quantile_matrix,convol);
     else:
       convol = (np.abs(convol[:Nxorig]))**2.0;
        
     # 2.4. Get the beat interval (median)
     for t in range(numF):
       t1 = int(beatSR[t])+1
       t2 = int(beatSR[t+1])
       CQ[k,t] = np.median(convol[t1-1:t2]);     
   
   # 2.5 Add the reference power (for mean/median/q-quantiles)
   if refLabel is 'mean':
     refPower = np.mean(meanPowerK);
     CQ = np.add(10.0*np.log10(CQ),-10.0*np.log10(refPower))
     CQ = np.add(CQ,np.transpose(np.tile(A_offsets,(numF,1))));
   elif refLabel is 'median':
     refPower = np.median(medianPowerK);
     CQ = np.add(10.0*np.log10(CQ),-10.0*np.log10(refPower))
     CQ = np.add(CQ,np.transpose(np.tile(A_offsets,(numF,1))));
   elif refLabel is 'q':
     quantile_value = np.sort(quantile_matrix); # sort the values, set reference as the value that falls in the q_value%-th of examples
     refPower = quantile_value[int(np.floor(q_value*Nxorig))-1]/K;
     CQ = np.add(10.0*np.log10(CQ),-10*np.log10(refPower))
     CQ = np.add(CQ,np.transpose(np.tile(A_offsets,(numF,1))));
   else:
     CQ = np.add(10.0*np.log10(CQ),-10.0*np.log10(refPower))
     CQ = np.add(CQ,np.transpose(np.tile(A_offsets,(numF,1))));
  
   # 3. Computer the beat sync-ed chromagram
   chromagram = np.zeros((bins,numF));
   normal_chromagram = np.zeros((bins,numF));
    
   for i in range(bins):
     chromagram[i,:] = np.sum(CQ[i::bins,:],0);
     
    
   # 4. Compute the normalized chromagram features (Linear norm)
   for i in range(chromagram.shape[1]):
     maxCol = np.max(chromagram[:,i]);
     minCol = np.min(chromagram[:,i]);
     if (maxCol>minCol):
       normal_chromagram[:,i] = np.true_divide((chromagram[:,i]-minCol),(maxCol-minCol));
     else:
	 normal_chromagram[:,i] = 0.0;   
 else:
   print('Error in cal_CQ_chroma_loudness: no such normFlag.');

 # Do the circle shift
 shift_pos = round(12.0*np.log2(freqK[0]/27.5)); # The relative position to A0
 shift_pos = int(np.mod(shift_pos,12)-3);        # since A0 should shift -3
 if not (shift_pos is 0):
   chromagram = np.roll(chromagram,shift_pos,0);
   normal_chromagram = np.roll(normal_chromagram,shift_pos,0);

 # 5. return the sample times
 beatSR = beatSR/SR; 
 sample_times = np.vstack([beatSR[:-1], beatSR[1:]]); 

 return chromagram, normal_chromagram, sample_times
 
def write_wave(x,fname,frate,nchannels):

  # Writes the wavefile in x to fname at frate frames/s and nchannes
  import wave
  import struct
  import numpy as np
  
  amp = 64000.0
  wav_file = wave.open(fname, "w")

  sampwidth = 2
  framerate = int(frate)
  nframes = len(x);
  comptype = "NONE"
  compname = "not compressed"

  # normalise
  x = np.true_divide(x,max(abs(x)));
  
  wav_file.setparams((nchannels, sampwidth, framerate, nframes,
    comptype, compname)) 

  for s in x:
    # write the audio frames to file
    wav_file.writeframes(struct.pack('h', int(s*amp/2)))

  wav_file.close()    
