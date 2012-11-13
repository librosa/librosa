def estimate_tuning(d,sr,fftlen,f_ctr,f_sd):

  # Get minimum/maximum frequencies
  fminl = octs2hz(hz2octs(f_ctr)-2*f_sd);
  fminu = octs2hz(hz2octs(f_ctr)-f_sd);
  fmaxl = octs2hz(hz2octs(f_ctr)+f_sd);
  fmaxu = octs2hz(hz2octs(f_ctr)+2*f_sd); 

  # Estimte pitches
  [p,m,S] = isp_ifptrack(d,fftlen,sr,fminl,fminu,fmaxl,fmaxu)
    
  # nzp=linear index of non-zero sinusoids found.
  nzp = p.flatten(1)>0;
  
  #Find significantly large magnitudes
  import numpy as np
  mflat = m.flatten(1);
  gmm = mflat > np.median(mflat[nzp])
  
  # 2. element-multiply large magnitudes with frequencies.
  nzp = nzp * gmm;
  
  # get non-zero again.
  nzp = np.nonzero(nzp)[0];
  
  # 3. convert to octaves
  #Pocts=p.copy(); not used
  pflat = p.flatten(1);
  
  # I didn't bother vectorising hz2octs....do it in a loop
  temp_hz = pflat[nzp];
  for i in range(len(temp_hz)):
      temp_hz[i] = hz2octs(temp_hz[i]);
      
  Poctsflat = p.flatten(1);  
  Poctsflat[nzp] = temp_hz;
  to_count = Poctsflat[nzp];
  
  # 4. get tuning
  nchr=12;   # size of feature

  # make histogram, resolution is 0.01, from -0.5 to 0.5
  import matplotlib.pyplot as plt
  term_one = nchr*to_count; 
  term_two = np.array(np.round(nchr*to_count),dtype=np.int)
  bins = [xxx * 0.01 for xxx in range(-50, 51)];
  
  # python uses edges, matlab uses centers so subtract half a bin size
  z = plt.hist(term_one-term_two-0.005,bins);

  hn = z[0];
  hx = z[1];

  # prepend things less than min
  nless = [sum(term_one-term_two-0.005 < -0.5)]
  hn = np.hstack([nless,hn]);

  # find peaks
  semisoff = hx[np.argmax(hn)];

  return semisoff

def hz2octs(freq, A = 440.0):

  # Convert a frequency in Hz into a real number counting 
  # the octaves above A0. So hz2octs(440) = 4.0
  # Optional A440 specifies the Hz to be treated as middle A (default 440).  
  # 2006-06-29 dpwe@ee.columbia.edu for fft2chromamx

  # A4 = A440 = 440 Hz, so A0 = 440/16 Hz
  import numpy as np
  octs = np.log(freq/(A/16))/np.log(2);
  return octs
  

def octs2hz(octs, A = 440.0):

  #hz = octs2hz(octs,A440)
  #Convert a real-number octave 
  #into a frequency in Hzfrequency in Hz into a real number counting 
  #the octaves above A0. So hz2octs(440) = 4.0.
  #Optional A440 specifies the Hz to be treated as middle A (default 440).
  #2006-06-29 dpwe@ee.columbia.edu for fft2chromamx
  
  #A4 = A440 = 440 Hz, so A0 = 440/16 Hz
  hz = (A/16)*(2**octs);
  return hz
  
def isp_ifptrack(d,w,sr,fminl = 150.0, fminu = 300.0, fmaxl = 2000.0, fmaxu = 4000.0):
    
  # DESCRIPTION
  # Track pitch based on instantaneous frequency. It looks for adjacent
  # bins with same inst freq.
  # INPUT
  #   d:
  #     Input waveform.
  #   w:
  #     STFT DFT length (window is half, hop is 1/4).
  #   sr:
  #     Sample rate.
  #   fminl, fmaxu, fmaxl, fmaxu:
  #     Define ramps at edge of sensitivity.
  #
  # OUTPUT
  #   p:
  #     Tracked pitch frequencies.
  #   m:
  #     Tracked pitch magnitudes.
  #   S:
  #     The underlying complex STFT.   
  
  # Only look at bins up to 2 kHz
  maxbin = int(round(fmaxu*float(w)/float(sr)));
  #minbin = int(round(fminl*float(w)/float(sr))); not used
  
  # Calculate the inst freq gram
  [I,S] = isp_ifgram(d,w,w/2,w/4,sr, maxbin);
  
  # Find plateaus in ifgram - stretches where delta IF is < thr
  import numpy as np
  ddif = I[np.hstack([range(1,maxbin),maxbin-1]),:]-I[np.hstack([0,range(0,maxbin-1)]),:];

  # expected increment per bin = sr/w, threshold at 3/4 that
  dgood = abs(ddif) < .75*float(sr)/float(w);

  # delete any single bins (both above and below are zero);
  logic_one = dgood[np.hstack([range(1,maxbin),maxbin-1]),:] > 0;
  logic_two = dgood[np.hstack([0,range(0,maxbin-1)]),:] > 0;
  dgood = dgood * np.logical_or(logic_one,logic_two);
    
  p = np.zeros(dgood.shape);
  m = np.zeros(dgood.shape);

  # For each frame, extract all harmonic freqs & magnitudes
  lds = np.size(dgood,0);
  for t in range(I.shape[1]):
    ds = dgood[:,t];
            
    # find nonzero regions in this vector
    logic_one = np.hstack([0,ds[range(0,lds-1)]])==0;
    logic_two = ds > 0;
    logic_oneandtwo = np.logical_and(logic_one,logic_two);
    st = np.nonzero(logic_oneandtwo)[0];
    
    logic_three = np.hstack([ds[range(1,lds)],0])==0;
    logic_twoandthree = np.logical_and(logic_two,logic_three);
    en = np.nonzero(logic_twoandthree)[0];

    # Set up inner loop    
    npks = len(st);
    frqs = np.zeros(npks);
    mags = np.zeros(npks);
    for i in range(len(st)):
      bump = np.abs(S[range(st[i],en[i]+1),t]);
      mags[i] = sum(bump);
      
      # another long division, split it up
      numer = np.dot(bump,I[range(st[i],en[i]+1),t]);
      isz = (mags[i]==0);
      denom = mags[i]+isz.astype(int);
      frqs[i] = numer/denom;
                                    
      if frqs[i] > fmaxu:
        mags[i] = 0;
        frqs[i] = 0;
      elif frqs[i] > fmaxl:
        mags[i] = mags[i] * max(0, (fmaxu - frqs[i])/(fmaxu-fmaxl));

      # downweight magnitudes below? 200 Hz
      if frqs[i] < fminl:
        mags[i] = 0;
        frqs[i] = 0;
      elif frqs[i] < fminu:
        # 1 octave fade-out
        mags[i] = mags[i] * (frqs[i] - fminl)/(fminu-fminl);

      if frqs[i] < 0: 
        mags[i] = 0;
        frqs[i] = 0;
          
    # Collect into bins      
    bin = np.round((st+en)/2.0);
    p[bin.astype(int),t] = frqs;
    m[bin.astype(int),t] = mags;

  return p,m,S
  
def isp_ifgram(X, N=256, W=256, H=256.0/2.0, SR=1, maxbin=1.0+256.0/2.0):

  # SYNTAX
  #  [F,D] = ifgram(X, N, W, H, SR, maxbin)
  #
  # DESCRIPTION
  #    Compute the instantaneous frequency (as a proportion of the sampling
  #    rate) obtained as the time-derivative of the phase of the complex
  #    spectrum as described by Toshihiro Abe et al in ICASSP'95,
  #    Eurospeech'97. Calculates regular STFT as side effect.
  #
  # INPUT
  #   X:
  #     Wave signal.
  #   N:
  #     FFT length.
  #   W:
  #     Window length.
  #   H:
  #     Step length.
  #   SR:
  #     Sampling rate.
  #   maxbin:
  #     The index of the maximum bin needed. If specified, unnecessary
  #     computations are skipped.
  # OUTPUT
  #   F:
  #     Instantaneous frequency spectrogram.
  #   D:
  #     Short time Fourier transform spectrogram.
  
  Flen = maxbin;
  s = X.size;

  # Make a Hanning window 
  import numpy as np
  win = 0.5*(1-np.cos(np.true_divide(np.arange(W)*2*np.pi,W)));

  # Window for discrete differentiation
  T = float(W)/float(SR);
  dwin = (-np.pi/T)*np.sin(np.true_divide(np.arange(W)*2*np.pi,W));

  # sum(win) takes out integration due to window, 2 compensates for neg frq
  norm = 2/sum(win);

  # How many complete windows?
  import math
  nhops = 1 + int(math.floor((s - W)/H));
  
  F = np.zeros((Flen, nhops));
  D = np.zeros((Flen, nhops),dtype=complex);

  nmw1 = int(math.floor((N-W)/2));

  ww = 2*np.pi*np.arange(Flen)*SR/N;

  wu = np.zeros(N);
  du = np.zeros(N);
  
  # Main loop
  for h in range(nhops):
      u = X[h*H:(W+h*H)];

      # Pad or truncate samples if N != W
      # Apply windows now, while the length is right
      if N >= W:
        wu[nmw1:(nmw1+W)] = win*u;
        du[nmw1:(nmw1+W)] = dwin*u;
      elif N < W:
        # Can't make sense of Dan's code here:
        #wu = win[1-nmw1:N-nmw1]*u[1-nmw1:N-nmw1];
        #du = dwin[1-nmw1:N-nmw1]*u[1-nmw1:N-nmw1];
        print 'Error, N must be at least window size'

      # FFTs of straight samples plus differential-weighted ones
      # Replaced call to fftshift with inline version. Jesper Hjvang Jensen, Aug 2007
      # t1 = fft(fftshift(du));
      # t2 = fft(fftshift(wu));
      split = int(math.ceil(du.size/2.0) + 1);
      
      # Need to reverse front and last parts of du and wu      
      temp_du = np.hstack([du[split-1:],du[0:split-1]]);
      temp_wu = np.hstack([wu[split-1:],wu[0:split-1]]);
      
      t1 = np.fft.fft(temp_du);
      t2 = np.fft.fft(temp_wu);
      
      t1 = t1[0:Flen];
      t2 = t2[0:Flen];  
      
      # Scale down to factor out length & window effects
      D[:,h] = t2*norm;
      
      # Calculate instantaneous frequency from phase of differential spectrum
      t = t1 + 1j*(ww*t2);
      a = t2.real;
      b = t2.imag;
      da = t.real;
      db = t.imag;
      
      # split this confusing divsion into chunks!
      # instf = (1/(2*pi))*(a.*db - b.*da)./((a.*a + b.*b)+(t2==0));
      num_one = 1.0/(2*np.pi);
      num_two = (a*db - b*da);
      denom_one = (a*a + b*b);
      isz = (t2==0);
      instf = np.true_divide(num_one*num_two,denom_one+isz.astype(int));
      F[:,h] = instf;
 
  return F, D
