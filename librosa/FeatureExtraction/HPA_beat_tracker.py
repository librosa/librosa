def estimate_beats(d,sr,startbpm,tightness):
  # Function:
  # [bts, bpm]=estimate_beats(d,sr,startbpm,tightness)
  #
  # Get beat times (in seconds) and bpm of a wav file.
  #
  # INPUTS  
  #         - d. The waveform, matlab mono vector.
  #         - sr. Waveform sampling rate.
  #        - startbpm. The start point of the search.
  #         - tightness. How tightly to stick to the startbpm.
  # 
  #  OUTPUTS 
  #         - bts. vector of beat occurances in seconds.
  #        - bpm. Estimate of the beats per minute.
  # 
  # Use "beat tracker" (last argument is 0 for noplot)

  bts = 0;
  
  #  Get the beat times
  [BeatStrength, downsample, hop_len] = calculate_beat_strength(d,sr);
  
  bts = beatestimate(BeatStrength,downsample,hop_len,startbpm,tightness);

  #bts = beat(d,sr,startbpm,tightness,0);

  # Estimate bpm as median beat difference. 
  import numpy as np
  bpm = np.true_divide(60.0,np.median(np.diff(bts)));

  # Post-processing: if bpm<120, double everything.
  if bpm < 120:
    new_beats = [bts[0]];
    for b in range(1,bts.shape[0]):
      new_beats = np.hstack([new_beats, (bts[b-1]+bts[b])/2.0, bts[b]]);
    bts=new_beats;  
    
  return bts
  
def calculate_beat_strength(x,sr,downsample=8000,win_len=256,hop_len=32,mel_channels=40):

  # Function:
  #[BeatStrength, estimateBPM, options]=calculate_beat_strength(x,sr,downsample,win_len,...
  #                                     hop_len, mel_channels)
  # 
  # This function calculates the beat strength through a file, which can be fed into
  # beatestimate.m to estimate beat positions. 
  #
  # Prerquist: hz2mel and mel2hz.
  #
  # INPUTS  - x. Mono waveform.
  #         - sr. Sample rate of x.
  #         - downsample. New sample frequency.
  #         - win_len. Length of STFT to use.
  #         - hop_len. Hop of STFT to use.
  #         - mel_channels. Number of Mel bands to use.
  #
  # OUTPUTS - BeatStrength. The beat strengths.
  #         - options. The options used in this function.
  #
  # ---------------------------------------------
  # Function created by M. McVicar
  # Function revised by Y. Ni
  # Intelligent Systems Lab
  # University of Bristol
  # U.K.
  # 2011
  
  # hcf of new sr and sr
  #import fractions
  #gg = fractions.gcd(downsample,sr);

  # Resample and use downsample from now on
  from scipy import signal
  import numpy as np
  # DEBUG: don't downsample
  downsample = sr;
  #l = len(x)*np.true_divide(downsample,sr);
  #x = signal.resample(x,l);
  #sr=downsample;

  # Take stft  
  import HPA_stftistft as stftistft
  D = stftistft.stft(x,win_len,win_len,hop_len);

  # Mel parameters
  width = 1.0;
  minfrq = 0;
  maxfrq = sr/2.0; # Use the Nyquist
  htkmel = 0;

  # Initialise weights
  wts = np.zeros([mel_channels, win_len]);
  
  # Get the normal fft frequencies 
  fftfrqs=np.true_divide(np.multiply(range(win_len/2+1),sr),win_len);
  
  # Get the centers of the Mel frequencies
  mimel_channels = hz2mel(minfrq, htkmel);
  maxmel = hz2mel(maxfrq, htkmel);
  
  # binfrqs are the nearest fft frequencies to the mel frequencies
  binfrqs = np.zeros(mel_channels+2);
  for f in range(mel_channels+2):
    binfrqs[f] = mel2hz(np.true_divide(mimel_channels+(f),(mel_channels+1))*(maxmel-mimel_channels), htkmel);

  # work out weights
  for i in range(mel_channels):

    fs = binfrqs[i+np.array([0, 1, 2])];
  
    # scale by width
    fs = fs[1]+np.multiply(width,(fs - fs[1]));
  
    # lower and upper slopes for all bins
    loslope = np.true_divide((fftfrqs - fs[0]),(fs[1] - fs[0]));
    hislope = np.true_divide((fs[2] - fftfrqs),(fs[2] - fs[1]));
    
    # intersect them with each other and zero
    temp = np.zeros(loslope.shape);
    for j in range(len(temp)):
      temp[j] = max(0,min(loslope[j],hislope[j]))

    wts[i,(range(win_len/2 + 1))] = temp;
  
  # Slaney-style mel is scaled to be approx constant E per channel
  const_one = binfrqs[range(2,mel_channels+2)];
  const_two = binfrqs[range(mel_channels)];
  wts = np.dot(np.diag(np.true_divide(2,(const_one-const_two))),wts);
  
  # Make sure 2nd half of FFT is zero
  wts[:,range((win_len/2+1),win_len)] = 0.0;

  # Use these weights to convert the fft scale to mel scale
  D_mel=np.dot(wts[:,range((win_len/2)+1)],abs(D));

  # Post-process. Take the non numerically zero part
  D_nonzero = np.maximum(1e-10,D_mel);

  # Convert to dB
  D_dB=20.0*np.log10(D_nonzero);

  # only look at top 80 dB
  D_trimmed = np.maximum(D_dB,D_dB.max()-80.0);

  # Take first difference
  D_diff=np.diff(D_trimmed,1,1);

  # Use nonzero part
  D_nz=np.maximum(0.0,D_diff);

  # Take average over mel bands
  D_raw_envelope=np.mean(D_nz,0);

  # filter with a simple window
  from scipy.signal import lfilter
  D_windowed = lfilter([1.0,-1.0], [1.0,-.99],D_raw_envelope);
  
  # Divide by standard deviation
  BeatStrength = D_windowed.copy(); #D_windowed/std(D_windowed);

  return BeatStrength, downsample, hop_len
  
def hz2mel(f,htk=0):
  #  z = hz2mel(f,htk)
  #  Convert frequencies f (in Hz) to mel 'scale'.
  #  Optional htk = 1 uses the mel axis defined in the HTKBook
  #  otherwise use Slaney's formula
  # 2005-04-19 dpwe@ee.columbia.edu

  #if nargin < 2
  #htk = 0;
  #end

  import numpy as np
  if htk == 1:
    z = 2595 * np.log10(1+np.true_divide(f,700.0));
  else:
    # Mel fn to match Slaney's Auditory Toolbox mfcc.m

    f_0 = 0.0; # 133.33333;
    f_sp = 200.0/3.0; # 66.66667;
    brkfrq = 1000.0;
    brkpt  = np.true_divide((brkfrq - f_0),f_sp);  # starting mel value for log region
    logstep = np.exp(np.true_divide(np.log(6.4),27.0)); 
    # magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 
    # 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear 
    # filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and 
    # exp(log(6.4)/27) = 1.07117028749447)
    if (f < brkfrq):
      # fill in parts separately
      z = np.true_divide((f - f_0),f_sp);
    else:
      z = brkpt+(np.true_divide(np.log(np.true_divide(f,brkfrq)),np.log(logstep)));

  mel = z;     
  return mel
  
def mel2hz(z, htk=0):
  #   f = mel2hz(z, htk)
  #   Convert 'mel scale' frequencies into Hz
  #  Optional htk = 1 means use the HTK formula
  #   else use the formula from Slaney's mfcc.m
  # 2005-04-19 dpwe@ee.columbia.edu

  if htk == 1:
    f = 700.0*(10.0**(z/2595.0)-1.0);
  else:
  
    import numpy as np
    f_0 = 0.0; # 133.33333;
    f_sp = 200.0/3.0; # 66.66667;
    brkfrq = 1000.0;
    brkpt  = np.true_divide((brkfrq - f_0),f_sp);  # starting mel value for log region
    logstep = np.exp(np.log(6.4)/27.0); # see 'magic comment in hz2mel

    if (z < brkpt):
      f = f_0 + f_sp*z;
    else:
      f = brkfrq*np.exp(np.log(logstep)*(z-brkpt));

  return f
  
def beatestimate(BeatStrength,sr,hop,startbpm,beat_tightness):
  # Function:
  # Beats=beatestimate(BeatStrength,sr,hop,startbpm,...
  #        beat_tightness)
  # 
  # This function estimates the beat times from the beat strength.
  #
  # INPUTS  - BeatStrength. The strength of the beat, time windowed by a stft
  #         - sr. Sample rate of BeatStrength (note this might have been
  #           resampled.
  #         - hop. number of samples hopped in BeatStrength
  #         - startbpm. The start point of the search.
  #         - tightness. How tightly to stick to the startbpm.
  #
  # OUTPUTS - Beats. The beat times in seconds
  #
  #---------------------------------------------
  #Function created by M. McVicar
  #Function revised by Y. Ni
  #Intelligent Systems Lab
  #University of Bristol
  #U.K.
  #2011

  Beats = 0;
  
  # 1. Estimate rough start bpm

  # Find rough global period (empirically, 0s-90s)
  duration_time = 90.0;  #in seconds
  upper_time_zone = 90.0; #in seconds
  bpm_std = 0.7; #the variance of the bpm window
  alpha = 0.8; #a update weight for part 3.

  # sample rate for specgram frames (due to the hop_length)
  import numpy as np
  fftres = np.true_divide(sr,hop);
   
  # Get the lower bound and the upper bound in the beat strength vector
  maxcol = int(min(np.round(upper_time_zone*fftres),len(BeatStrength)-1)); 
  mincol = int(max(1,maxcol-np.round(duration_time*fftres)));
  
  # Use auto-correlation out of 4 seconds (empirically set?)
  acmax = int(np.round(4*fftres));
  
  # Get autocorrelation of signal
  import matplotlib.pyplot as plt
  
  # matlab auto zero-pads, python doesn't
  if BeatStrength.shape[0] >= acmax:
    rrr = plt.xcorr(BeatStrength[range(mincol-1,maxcol)],BeatStrength[range(mincol-1,maxcol)],normed=False,maxlags=acmax)
    xcr = rrr[1]  
  else:
    maxlaglen = len(BeatStrength[range(mincol-1,maxcol)])-1;
    rrr = plt.xcorr(BeatStrength[range(mincol-1,maxcol)],BeatStrength[range(mincol-1,maxcol)],normed=False,maxlags=maxlaglen)
    xcr = rrr[1]
    des_len = acmax*2+1
    npad = (des_len-len(xcr))/2
    xcr = np.hstack([np.zeros(npad),xcr,np.zeros(npad)])
  
  # Find local max in the global auto-correlation
  rawxcr = xcr[range(acmax,2*acmax+1)]; # The right side of correlation part

  # Creating a hamming like window around default bpm
  bpms = 60.0*np.true_divide(fftres,(np.add(range(acmax+1),0.1)));
  
  num = np.log(np.true_divide(bpms,startbpm))*bpm_std;
  denom = np.log(2.0); 
  div = np.true_divide(num,denom);  
  xcrwin = np.exp(-0.5*div**2.0);
  
  # The weighted auto-correlation
  xcr = rawxcr*xcrwin;
 
  # %Add in 2x, 3x, choose largest combined peak
  # lxcr = length(xcr);
  # xcr00 = [0, xcr, 0];
  # xcr2 = xcr(1:ceil(lxcr/2))+.5*(.5*xcr00(1:2:lxcr)+xcr00(2:2:lxcr+1)+.5*xcr00(3:2:lxcr+2));
  # xcr3 = xcr(1:ceil(lxcr/3))+.33*(xcr00(1:3:lxcr)+xcr00(2:3:lxcr+1)+xcr00(3:3:lxcr+2));
  # 
  # 
  # %Get the bpm position of the peak
  # if max(xcr2) > max(xcr3)
  #   [vv, startpd] = max(xcr2);
  #   startpd = startpd -1;
  #   startpd2 = startpd*2;
  # else
  #   [vv, startpd] = max(xcr3);
  #   startpd = startpd -1;
  #   startpd2 = startpd*3;
  # end

  # %Get the local max (the picks)
  xpks = localmax(xcr);
  
  #Not include any peaks in first down slope (before goes below
  # zero for the first time)
  xpks[range(np.min(np.nonzero(xcr<0)))] = 0;

  #Largest local max away from zero
  maxpk = np.max(xcr[xpks]);

  #Find the position of the first largest pick
  z = np.nonzero((xpks*xcr) == maxpk);
  startpd = z[0];
  startpd = startpd[0];

  # Choose best peak out of .33 .5 2 3 x this period
  candpds = np.round(np.multiply([.33, .5, 2.0, 3.0],startpd)).astype(int);
  candpds = candpds[candpds < acmax];

  bestpd2 = np.argmax(xcr[candpds])+1; # to match matlab
  startpd2 = candpds[bestpd2];

  # %Weight startpd and startpd2
  # pratio = xcr(1+startpd)/(xcr(1+startpd)+xcr(1+startpd2));
  # if (pratio>0.5)
  #     startbpm=(60*fftres)/startpd;
  # else
  #     startbpm=(60*fftres)/startpd2;
  # end

  #Always use the faster one
  startbpm = np.true_divide(60.0*fftres,np.minimum(startpd,startpd2));

  ### 1. Smooth the beat strength ###

  # BeatStrength=BeatStrength/std(BeatStrength);

  startpd = int(round(60.0*fftres)/startbpm);
  pd = startpd;
  
  # Smooth beat events with a gaussian window
  templt = np.exp(-0.5*(((range(-pd,pd+1))/(np.true_divide(pd,32.0)))**2.0));
  
  # convolve the window with the BeatStrength
  import scipy.signal
  localscore = scipy.signal.fftconvolve(templt,BeatStrength);
  localscore = localscore[np.add(int(round(len(templt)/2.0)),range(BeatStrength.shape[0]))];

  ### 2.Initialise ###

  backlink = np.zeros(localscore.shape[0]);
  cumscore = np.zeros(localscore.shape[0]);

  #  search range for previous beat. prange is the number of samples to look
  #  back and forward
  # prange = round(-2*pd):-round(pd/2);
  prange = np.round(range(-2*pd,-pd/2+1));

  #  Make a score window, which begins biased towards 120bpm and skewed.
  # txwt = (-beat_tightness*abs((log(prange/-pd)).^2));
  txwt = np.exp(-0.5*(beat_tightness*(np.log(np.true_divide(prange,-pd))))**2.0);

  #  'Starting' is 1 for periods of (near) silence.
  starting = 1;

  ### 3 Forward step ###
  
  #  Main forward loop. Go through each window, padding zeros backwards if
  #  needed, and add the cumulative score to the prior (txwt).
  for i in range(localscore.shape[0]):
  #for i in range(1):
  
    #  Move the time window along  
    timerange = np.add(i,prange);
  
    #  Are we reaching back before time zero?
    zpad = np.maximum(0, np.minimum(1-timerange[0],prange.shape[0]));

    #  Search over all possible predecessors and apply transition 
    #  weighting
    scorecands = txwt*np.hstack([np.zeros(zpad),cumscore[timerange[zpad:]]]);
    
    #  Find best predecessor beat
    current_score = np.max(scorecands);
    beat_location = np.argmax(scorecands);
  
    #  Add on local score
    cumscore[i] = alpha*current_score + (1-alpha)*localscore[i];

    #  special case to catch first onset. Stop if the local score is small (ie
    #  if there's near silence)
    if ((starting == 1) and (localscore[i] < 0.01*np.max(localscore))):
      backlink[i] = -1; # default
    else:
      #  found a probable beat, store it and leave the starting/silence
      #  scenario.
      backlink[i] = timerange[beat_location];
      starting = 0;
 
  ### 4. Get the last beat ###

  # cumscore now stores the score through the song, backlink the best
  # previous frame.

  # get the median non zero score
  maxes = localmax(cumscore);
  max_indices = np.nonzero(maxes)[0];
  peak_scores = cumscore[max_indices];

  medscore = np.median(peak_scores);

  # look for beats above 0.5 * median
  bestendposs = np.nonzero(cumscore * localmax(cumscore) > 0.5*medscore)[0];
 
  # The last of these is the last beat (since the score generally increases)
  bestendx = np.max(bestendposs);
  
  ### 5. Backtrace ###
  # begin on last beat
  b = [int(bestendx)];

  # go while backlink is positive (we set it to be -1 in silence)
  while backlink[b[-1]] > 0:
    # append the previous beat
    b = np.hstack([b,int(backlink[b[-1]])]);

  ### 6. Output ###
  # Beats are currently backwards, so flip. Also return in s (need +1)
  b = b[ ::-1];
  Beats = np.true_divide(np.add(b,1),fftres);
  
  return Beats
  
def localmax(x):
  # return 1 where there are local maxima in x (columnwise).
  # don't include first point, maybe last point

  import numpy as np
     
  ''' m = localmax(a)
  return 1 where there are local maxima in x (columnwise).
  don't include first point, maybe last point'''
     
  #return np.append([0], np.logical_and(x[1:] > x[:-1], x[:-1] < x[1:]))
  #m = (x > [x(1),x(1:(lx-1))]) & (x >= [x(2:lx),1+x(lx)]);
  lx = x.shape[0];
  m = np.logical_and(x > np.hstack([x[0],x[range(lx-1)]]),x >= np.hstack([x[range(1,lx)],x[lx-1]]));

  return m
