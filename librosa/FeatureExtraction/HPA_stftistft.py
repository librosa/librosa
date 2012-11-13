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
