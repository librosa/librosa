def HPSS(x,nfft,noverlap,window,gamma,alpha,kmax):

 # [xharmony,xpercussion]=HPSS(x,nfft,noverlap,window,gamma,alpha,kmax)
 # 
 #  Splits the harmony and percussion of a wavfile.
 #  Prerquist: Fourier transformation function: stft.m/istft.m
 # 
 #  INPUTS            
 #                    x - wavfile, mono
 #                 nfft - length of fft to use
 #             noverlap - length of overlap of fft
 #               window - length of window to use
 #                gamma - power spectrum reduction factor
 #                alpha - the degree of inital seperation
 #                 kmax - number of iterations.
 #        
 #  OUTPUTS  
 #             xharmony - the harmonic componant
 #          xpercussion - the percussive componant
 # 
 # ---------------------------------------------
 # Function created by M. McVicar
 # Function revised by Y. Ni
 # Intelligent Systems Lab
 # University of Bristol
 # U.K.
 # 2011
  
  F = stft(x,nfft,window,noverlap);

  # Power spectrum, reduced by gamma
  import numpy as np
  W = np.absolute(F)**(2.0*gamma);

  # 2. Initialise Harmony, Percussion and index.
  H=0.5*W;
  P=0.5*W;

  # 3. Evaluate the constants
  a1=alpha*0.25;
  a2=(1-alpha)*0.25;
  [rowLen,colLen] = W.shape;
  
  # 4. Iteratively update the Harmony and Percussion matrix
  Hnew=H.copy();
  Pnew=P.copy();

  for k in range(kmax):
     
    print k
    # 4.1 Do the boundary update. Its a square.
    # Top
    row=1-1;
    for col in range(colLen):
      Hnew[row,col] = min(max(2*H[row,col],0),W[row,col]);
      Pnew[row,col] = W[row,col]-Hnew[row,col];

    # Bottom
    row=rowLen-1;
    for col in range(colLen):
      Hnew[row,col] = min(max(2*H[row,col],0),W[row,col]);
      Pnew[row,col] = W[row,col]-Hnew[row,col];
     
    # Left
    col=1-1;
    for row in range(1,rowLen-1):
      Hnew[row,col] = min(max(2*H[row,col],0),W[row,col]);
      Pnew[row,col] = W[row,col]-Hnew[row,col];
     
    # Right
    col=colLen-1;
    for row in range(1,rowLen-1):
      Hnew[row,col] = min(max(2*H[row,col],0),W[row,col]);
      Pnew[row,col] = W[row,col]-Hnew[row,col];     
     
    # 4.2 calculate delta, update H and P
    for col in range(1,colLen-1):
      for row in range(1,rowLen-1):
        delk = a1*(H[row,col-1]-2*H[row,col]+H[row,col+1])-a2*(P[row-1,col]-2*P[row,col]+P[row+1,col]);
        Hnew[row,col] = min(max(H[row,col]+delk,0),W[row,col]);
        Pnew[row,col] = W[row,col]-Hnew[row,col];

    # 4.3 Each iteration
    H=Hnew.copy();
    P=Pnew.copy();
  
  # Binarise the result
  Hfinal=H.copy();
  Pfinal=P.copy();

  # 5. Get final P and H by binarising, i.e. taking the largest magnitude of H or P
  for col in range(colLen):
    for row in range(rowLen):
        if H[row,col] < P[row,col]:
            Hfinal[row,col] = 0;
            Pfinal[row,col] = W[row,col];
        else:
            Hfinal[row,col] = W[row,col];
            Pfinal[row,col] = 0;
  

  # 6. now convert back to waveform
  omegaH=np.zeros(F.shape,dtype=complex)
  omegaP=np.zeros(F.shape,dtype=complex)
  
  update_factor=1.0/(2.0*gamma);
  import cmath  
  for row in range(F.shape[0]):
    for col in range(F.shape[1]):
        phasepart = cmath.exp(1j*cmath.phase(F[row,col]));
        omegaH[row,col] = (Hfinal[row,col]**update_factor)*phasepart;
        omegaP[row,col] = (Pfinal[row,col]**update_factor)*phasepart;
  
  # 7. Take inverse short time Fourier transformation
  xharmony = istft(omegaH,nfft,window,noverlap);
  xpercussion = istft(omegaP,nfft,window,noverlap);
  
  return xharmony, xpercussion

def stft(x, f, w, h):
 # D = stft(X, F, W, H)                            
 # Short-time Fourier transform.
 # Returns some frames of short-term Fourier transform of x.  Each 
 # column of the result is one F-point fft; each successive frame is 
 # offset by H points until X is exhausted.  Data is hamm-windowed 
 # at W pts, or rectangular if W=0, or with W if it is a vector.
 # See also 'istft.m'.
 # dpwe 1994may05.  Uses built-in 'fft'
 # $Header: /homes/dpwe/public_html/resources/matlab/pvoc/RCS/stft.m,v 1.2 2009/01/07 04:32:42 dpwe Exp $

  d = 0;
  s = len(x);

  # sort out window types
  import numpy as np
  if type(w) is int:
    if w == 0:
      # special case: rectangular window
      win = np.ones(f);
    else:
      if (w % 2) == 0:   # force window to be odd-len
        w = w + 1;
        halflen = int((w-1)/2.0);
        halff = int(f/2.0);   # midpoint of win
        halfwin = 0.5*(1+np.cos(np.pi*np.true_divide(range(halflen+1),halflen)))
        win = np.zeros(f);
        acthalflen = min(halff, halflen);
        win[range(halff,halff+acthalflen)] = halfwin[range(acthalflen)];
        win[range(halff,halff-acthalflen,-1)] = halfwin[range(acthalflen)];
  else:
    win = w;
    w = len(w);
  
  c = 0;
  # pre-allocate output array
  d = np.zeros(((1+f/2.0),1+np.floor((s-f)/h)),dtype=complex);

  for b in range(0,s-f+1,h):
    u = win*x[range(b,b+f)];
    t = np.fft.fft(u);
    d[:,c] = t[range(1+int(f/2.0))];
    c = c+1;

  return d
  
def istft(d, ftsize, w, h):
 # Function:
 # X = istft(D, F, W, H)
 # Inverse short-time Fourier transform.
 # Performs overlap-add resynthesis from the short-time Fourier transform
 # data in D.  Each column of D is taken as the result of an F-point
 # fft; each successive frame was offset by H points. Data is
 # hamm-windowed at W pts.
 # W = 0 gives a rectangular window; W as a vector uses that as window.
 # dpwe 1994may24.  Uses built-in 'ifft' etc.
 # $Header: /homes/dpwe/public_html/resources/matlab/pvoc/RCS/istft.m,v 1.4 2009/01/07 04:20:00 dpwe Exp $

  s = d.shape;
  cols = s[1];
  xlen = ftsize + (cols-1)*h;
  import numpy as np
  x = np.zeros(xlen);

  # sort out window types
  if type(w) is int:
    if w == 0:
      # special case: rectangular window
      win = np.ones(ftsize);
    else:
      if (w % 2) == 0:   # force window to be odd-len
        w = w + 1;
        halflen = int((w-1)/2.0);
        halff = int(ftsize/2.0);   # midpoint of win
        halfwin = 0.5*(1+np.cos(np.pi*np.true_divide(range(halflen+1),halflen)))
        win = np.zeros(ftsize);
        acthalflen = min(halff, halflen);
        win[range(halff,halff+acthalflen)] = halfwin[range(acthalflen)];
        win[range(halff,halff-acthalflen,-1)] = halfwin[range(acthalflen)];
        # 2009-01-06: Make stft-istft loop be identity
        win = np.true_divide(2.0*win,3.0);
  else:
    win = w;
    w = len(w);
    
  # Grab the ifft of each column, weighted by win
  for b in range(0,(h*(cols-1)),h):
    ft = d[:,int(b/h)];
    r = range(int(ftsize/2)-1,0,-1);
    part_two = np.conj(ft[r]);
    ft = np.hstack([ft,part_two]);
    px = np.real(np.fft.ifft(ft));
    x[range(b,b+ftsize)] = x[range(b,b+ftsize)]+px*win;

  return x 
 
