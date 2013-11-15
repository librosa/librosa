#!/usr/bin/env python
"""Core IO, DSP and utility functions."""

import os.path
import audioread

import numpy as np
import numpy.fft as fft
import scipy.signal
import scipy.ndimage

# Do we have scikits.samplerate?
try:
    import scikits.samplerate as samplerate
    _HAS_SAMPLERATE = True
except ImportError:
    _HAS_SAMPLERATE = False



#-- CORE ROUTINES --#
def load(path, sr=22050, mono=True, offset=0.0, duration=None):
    """Load an audio file as a floating point time series.

    :parameters:
      - path : string
          path to the input file

      - sr   : int > 0
          target sample rate.
          'None' uses the native sampling rate

      - mono : boolean
          convert signal to mono

      - offset : float
          start reading after this time (in seconds)

      - duration : float
          only load up to this much audio (in seconds)

    :returns:
      - y    : np.ndarray
          audio time series

      - sr   : int  
          sampling rate of y

    """

    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate

        s_start = np.floor(sr_native * offset) * input_file.channels
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (np.ceil(sr_native * duration) 
                                * input_file.channels)


        Z = float(1<<15)

        y = []
        n = 0

        for frame in input_file:
            frame   = np.frombuffer(frame, '<i2').astype(float)
            n_prev  = n
            n       = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue
            
            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start < n:
                # beginning is in this frame
                frame = frame[s_start - n_prev : ]

            # tack on the current frame
            y.append(frame)


        y = np.concatenate(y) / Z
        if input_file.channels > 1:
            if mono:
                y = 0.5 * (y[::2] + y[1::2])
            else:
                y = y.reshape( (-1, 2)).T

    if sr is not None:
        y = resample(y, sr_native, sr)
    else:
        sr = sr_native

    return (y, sr)

def resample(y, orig_sr, target_sr, res_type='sinc_fastest'):
    """Resample a signal from orig_sr to target_sr

    :parameters:
      - y           : np.ndarray
          audio time series 

      - orig_sr     : int
          original sample rate of y

      - target_sr   : int
          target sample rate

      - res_type    : str
          resample type (see note)
    
    :returns:
      - y_hat       : np.ndarray
          y resampled from orig_sr to target_sr

    .. note::
        If scikits.samplerate is installed, resample will use res_type
        otherwise, it will fall back on scipy.signal.resample

    """

    if orig_sr == target_sr:
        return y

    if _HAS_SAMPLERATE:
        y_hat = samplerate.resample(y.T, float(target_sr) / orig_sr, res_type).T
    else:
        n_samples = y.shape[-1] * target_sr / orig_sr
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)

    return y_hat

def stft(y, n_fft=256, hop_length=None, hann_w=None, window=None):
    """Short-time fourier transform

    :parameters:
      - y           : np.ndarray
          the input signal

      - n_fft       : int
          number of FFT components

      - hop_length  : int
          number audio of frames between STFT columns.
          If unspecified, defaults hann_w / 2.

      - hann_w      : int
          The size of Hann window. 
          If unspecified, defaults to n_fft

      - window      : np.ndarray
          (optional) user-specified window

    :returns:
      - D           : np.ndarray, dtype=complex
          STFT matrix

    """
    num_samples = len(y)

    # if there is no user-specified window, construct it
    if window is None:
        if hann_w is None:
            hann_w = n_fft

        if hann_w == 0:
            window = np.ones((n_fft,))
        else:
            lpad = (n_fft - hann_w)/2
            window = np.pad( scipy.signal.hann(hann_w, sym=False), 
                                (lpad, n_fft - hann_w - lpad), 
                                mode='constant')
    else:
        window = np.asarray(window)
        if window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and window size')

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(n_fft / 2)

    n_specbins  = 1 + int(n_fft / 2)
    n_frames    = 1 + int( (num_samples - n_fft) / hop_length)

    # allocate output array
    stft_matrix = np.empty( (n_specbins, n_frames), dtype=np.complex)

    for i in xrange(n_frames):
        sample  = i * hop_length
        frame   = fft.fft(window * y[sample:(sample+n_fft)])

        # Conjugate here to match phase from DPWE code
        stft_matrix[:, i]  = frame[:n_specbins].conj()

    return stft_matrix

def istft(stft_matrix, n_fft=None, hop_length=None, hann_w=None, window=None):
    """
    Inverse short-time fourier transform

    :parameters:
      - stft_matrix : np.ndarray
          STFT matrix from stft()

      - n_fft       : int
          number of FFT components.
          If unspecified, n_fft is inferred from the shape of stft_matrix.

      - hop_length  : int
          Number of audio frames between STFT columns.
          If unspecified, defaults to hann_w / 2.

      - hann_w      : int
          size of Hann window
          If unspecified, defaults to the value of n_fft.

      - window      : np.ndarray
          (optional) user-specified window

    :returns:
      - y           : np.ndarray
          time domain signal reconstructed from stft_matrix

    """

    # n = Number of stft frames
    n_frames    = stft_matrix.shape[1]

    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[0] - 1)

    # if there is no user-specified window, construct it
    if window is None: 
        if hann_w is None:
            hann_w = n_fft

        if hann_w == 0:
            window = np.ones(n_fft)
        else:
            #   magic number alert!
            #   2/3 scaling is to make stft(istft(.)) identity for 25% hop
            lpad = (n_fft - hann_w)/2
            window = np.pad( scipy.signal.hann(hann_w, sym=False) * 2.0 / 3.0, 
                                (lpad, n_fft - hann_w - lpad), 
                                mode='constant')

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = n_fft / 2

    y = np.zeros(n_fft + hop_length * (n_frames - 1))

    for i in xrange(n_frames):
        sample  = i * hop_length
        spec    = stft_matrix[:, i].flatten()
        spec    = np.concatenate((spec.conj(), spec[-2:0:-1] ), 0)
        ytmp    = window * fft.ifft(spec).real

        y[sample:(sample+n_fft)] = y[sample:(sample+n_fft)] + ytmp

    return y

def ifgram(y, sr=22050, n_fft=256, hop_length=None, win_length=None):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as 
    described by Toshihir Abe et al. in ICASSP'95, Eurospeech'97. Calculates regular
    STFT as a side effect.

    :parameters:
      - y       : np.ndarray
          audio time series
      - sr      : int > 0
          sampling rate
      - n_fft   : int > 0
          FFT window size
      - hop_length : int > 0
          hop length. If not supplied, defaults to n_fft / 4
      - win_length : int > 0, <= n_fft
          hann window length. Defaults to n_fft.

    :returns:
      - F : np.ndarray, dtype=real
          Instantaneous frequency spectrogram
      - D : np.ndarray, dtype=complex
          Short-time fourier transform
    '''

    num_samples = len(y)

    if hop_length is None:
        hop_length = n_fft / 4

    if win_length is None:
        win_length = n_fft

    # Construct a padded hann window
    lpad = (n_fft - win_length)/2
    window = np.pad( scipy.signal.hann(win_length, sym=False), 
                        (lpad, n_fft - win_length- lpad), 
                        mode='constant')

    # Window for discrete differentiation
    dwin = -np.pi * sr / n_fft * np.sin( np.linspace(0, 2 * np.pi, n_fft, endpoint=False))

    # Construct output arrays
    n_specbins = 1 + n_fft / 2

    F = np.zeros((n_specbins, 1 + (num_samples - n_fft) / hop_length))
    D = np.zeros_like(F, dtype=np.complex)

    # FIXME:  2013-11-15 08:20:12 by Brian McFee <brm2132@columbia.edu>
    # rename this variable. looks like angular frequency?

    ww = np.linspace(0, 2 * np.pi * F.shape[0] * sr / n_fft, F.shape[0], endpoint=False)

    # Main loop: fill in F and D
    for i in xrange(F.shape[1]):
        sample = y[i * hop_length : i * hop_length + n_fft]

        # Store the STFT. Conjugate here to match DWPE's matlab code.
        D[:, i] = fft.fft(window * sample)[:n_specbins].conj()

        # Calculate the instantaneous frequency from phase of differential spectrum

        # Compute the fft. Conjugate here is to match DPWE
        t       = fft.fft(dwin * sample)[:n_specbins].conj()
        t       = t - 1.j * ww * D[:, i]

        z_t2    = np.abs(D[:, i])**2

        # Compensate for zeros
        z_t2[z_t2 == 0] = 1.0

        F[:, i] = (t.conj() * D[:, i]).imag / (2 * np.pi * z_t2)

    # FIXME:  2013-11-15 09:01:22 by Brian McFee <brm2132@columbia.edu>
    # This guy is n_fft/4 smaller than that returned by stft()
    # does this need to be normalized?  or should we normalize in STFT?

    # Compensate for windowing effects, store STFT
    # sum(window) takes out integration due to window, 2 compensates for negative
    # frequency
    D = D * 2.0 / window.sum()

    return F, D

def logamplitude(S, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram

    :parameters:
      - S       : np.ndarray
          input spectrogram

      - amin    : float
          minimum amplitude threshold 

      - top_db  : float
          threshold ``max(log(S)) - top_db``

    :returns:
      log_S   : np.ndarray
          ``log_S ~= 10 * log10(S)``

    """

    log_S   =   10.0 * np.log10(np.maximum(amin, np.abs(S)))

    if top_db is not None:
        log_S = np.maximum(log_S, log_S.max() - top_db)

    return log_S

def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that D = S * P.

    :parameters:
      - D       : np.ndarray, dtype=complex
          input complex-valued spectrogram

    :returns:
      - S       : np.ndarray, dtype=real
          magnitude of D
      - P       : np.ndarray, dtype=complex
          exp(1.j * phi) where phi is the phase of D

    """

    S = np.abs(D)
    P = np.exp(1.j * np.angle(D))

    return S, P

def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, time-stretch by a 
    factor of rate.

    :parameters:
      - D       : np.ndarray, dtype=complex
          STFT matrix
      - rate    :  float, positive
          time-stretch factor
      - hop_length : int or None
        hop length of D.  If None, defaults to 2*(D.shape[0]-1)/4

    :returns:
      - D_stretched  : np.ndarray, dtype=complex
        time-stretched STFT

    .. note::
      - This implementation was ported from the following:
      - @misc{Ellis02-pvoc
            author = {D. P. W. Ellis},
            year = {2002},
            title = {A Phase Vocoder in {M}atlab},
            note = {Web resource},
            url = {http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/},}
            
    """
    
    n_fft = 2 * ( D.shape[0] - 1 )
    
    if hop_length is None:
        hop_length = n_fft / 4
    
    time_steps = np.arange(0, D.shape[1], rate, dtype=np.float)
    
    # Create an empty output array
    D_r = np.zeros((D.shape[0], len(time_steps)), D.dtype)
    
    # Expected phase advance in each bin
    dphi = (2 * np.pi * hop_length * 
                np.arange(D.shape[0], dtype=np.float) / n_fft)
    
    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])
    
    # Pad an extra 0 column to simplify boundary logic
    D = np.hstack( (D, np.zeros((D.shape[0], 2), dtype=D.dtype)))

    idx = np.array([0, 1], dtype=np.int)
    for (t, step) in enumerate(time_steps):
        
        D_cols = D[:, int(step) + idx]
        
        # Weighting for magnitude interpolation
        tf     = step - int(step)
        D_mag  = (1.0-tf) * np.abs(D_cols[:, 0]) + tf * np.abs(D_cols[:, 1])
        
        # Compute phase advance
        dp     = np.angle(D_cols[:, 1]) - np.angle(D_cols[:, 0]) - dphi
        
        # Wrap to -pi:pi range
        dp     = dp - 2*np.pi * np.round(dp / (2*np.pi))
        
        # Store to output array
        D_r[:, t] = D_mag * np.exp(1.j * phase_acc)
        
        # Accumulate phase
        phase_acc = phase_acc + dphi + dp
    
    return D_r

#-- FREQUENCY UTILITIES AND CONVERTERS--#
def midi_to_hz( notes ):
    """Get the frequency (Hz) of MIDI note(s)

    :parameters:
      - note_num      : int, np.ndarray
          number of the note(s)

    :returns:
      - frequency     : float, np.ndarray
          frequency of the note in Hz
    """

    return 440.0 * (2.0 ** ((notes - 69)/12.0))

def hz_to_midi( frequency ):
    """Get the closest MIDI note number(s) for given frequencies

    :parameters:
      - frequencies   : float, np.ndarray
          target frequencies

    :returns:
      - note_nums     : int, np.ndarray
          closest MIDI notes

    """

    return 12 * (np.log2(frequency) - np.log2(440.0)) + 69

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    :parameters:
      - frequencies   : np.ndarray, float
          scalar or array of frequencies
      - htk           : boolean
          use HTK formula instead of Slaney

    :returns:
      - mels        : np.ndarray
          input frequencies in Mels

    """

    if np.isscalar(frequencies):
        frequencies = np.array([frequencies], dtype=float)
    else:
        frequencies = frequencies.astype(float)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    
    # Fill in the linear part
    f_min   = 0.0
    f_sp    = 200.0 / 3

    mels    = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    
    min_log_hz  = 1000.0                        # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep     = np.log(6.4) / 27.0            # step size for log region

    log_t       = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels

def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    :parameters:
      - mels          : np.ndarray, float
          mel bins to convert
      - htk           : boolean
          use HTK formula instead of Slaney

    :returns:
      - frequencies   : np.ndarray
          input mels in Hz

    """

    if np.isscalar(mels):
        mels = np.array([mels], dtype=float)
    else:
        mels = mels.astype(float)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min       = 0.0
    f_sp        = 200.0 / 3
    freqs       = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz  = 1000.0                        # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep     = np.log(6.4) / 27.0            # step size for log region
    log_t       = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs

def hz_to_octs(frequencies, A440=440.0):
    """Convert frquencies (Hz) to octave numbers

    :parameters:
      - frequencies   : np.ndarray, float
          scalar or vector of frequencies
      - A440          : float
          frequency of A440

    :returns:
      - octaves       : np.ndarray
          octave number for each frequency

    """
    return np.log2(frequencies / (A440 / 16.0))

def octs_to_hz(octs, A440=440.0):
    """Convert octaves numbers to frequencies

    :parameters:
      - octaves       : np.ndarray
          octave number for each frequency
      - A440          : float
          frequency of A440

    :returns:
      - frequencies   : np.ndarray, float
          scalar or vector of frequencies

    """
    return (A440/16)*(2**octs)

def mel_frequencies(n_mels=40, fmin=0.0, fmax=11025.0, htk=False):
    """Compute the center frequencies of mel bands

    :parameters:
      - n_mels    : int
          number of Mel bins  
      - fmin      : float
          minimum frequency (Hz)
      - fmax      : float
          maximum frequency (Hz)
      - htk       : boolean
          use HTK formula instead of Slaney

    :returns:
      - bin_frequencies : ndarray
          ``n_mels+1``-dimensional vector of Mel frequencies

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel  = hz_to_mel(fmin, htk=htk)
    maxmel  = hz_to_mel(fmax, htk=htk)

    mels    = np.arange(minmel, maxmel + 1, (maxmel - minmel)/(n_mels + 1.0))
    
    return  mel_to_hz(mels, htk=htk)

#-- UTILITIES --#
def frames_to_time(frames, sr=22050, hop_length=128):
    """Converts frame counts to time (seconds)

    :parameters:
      - frames     : np.ndarray
          vector of frame numbers

      - sr         : int
          audio sampling rate 

      - hop_length : int
          hop length

    :returns:
      - times : np.ndarray 
          time (in seconds) of each given frame number:
          ``times[i] = frames[i] * hop_length / sr``

    """
    return (frames * hop_length) / float(sr)

def autocorrelate(y, max_size=None):
    """Bounded auto-correlation

    :parameters:
      - y         : np.ndarray
          vector to autocorrelate

      - max_size  : int
          maximum correlation lag.
          If unspecified, defaults to ``len(y)``

    :returns:
      - z         : np.ndarray
          truncated autocorrelation ``y*y``

    """

    result = scipy.signal.fftconvolve(y, y[::-1], mode='full')

    result = result[len(result)/2:]

    if max_size is None:
        return result
    
    return result[:max_size]

def localmax(x):
    """Return 1 where there are local maxima in x (column-wise)
       left edges do not fire, right edges might.

    :parameters:
      - x     : np.ndarray
          input vector

    :returns:
      - m     : np.ndarray, dtype=boolean
          boolean indicator vector of local maxima:
          m[i] <=> x[i] is a local maximum
    """

    return np.logical_and(x > np.hstack([x[0], x[:-1]]), 
                             x >= np.hstack([x[1:], x[-1]]))

def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    '''Uses a flexible heuristic to pick peaks in a signal.
    
    :parameters:
      - x         : np.ndarray
          input signal to peak picks from
      - pre_max   : int
          number of samples before n over which max is computed
      - post_max  : int
          number of samples after n over which max is computed
      - pre_avg   : int
          number of samples before n over which mean is computed
      - post_avg  : int
          number of samples after n over which mean is computed
      - delta     : float
          threshold offset for mean
      - wait      : int
          number of samples to wait after picking a peak

    :returns:
      - peaks     : np.ndarray, dtype=int
          indices of peaks in x
    
    .. note::
      A sample n is selected as an peak if the corresponding x[n]
      fulfills the following three conditions:
      1. x[n] = max(x[n - pre_max:n + post_max])
      2. x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta
      3. n - previous_n > wait
      where previous_n is the last sample n picked as a peak (greedily).
    
    .. note::
      S. Bock, F. Krebs and M. Schedl (2012)
      Evaluating the Online Capabilities of Onset Detection Methods
      13th International Society for Music Information Retrieval Conference
    
    .. note::
      Implementation based on 
      https://github.com/CPJKU/onset_detection/blob/master/onset_program.py
    '''

    # Get the maximum of the signal over a sliding window
    max_length  = pre_max + post_max + 1
    max_origin  = np.floor((pre_max - post_max)/2)
    mov_max     = scipy.ndimage.filters.maximum_filter1d(x, max_length, 
                                                            mode='constant', 
                                                            origin=max_origin)

    # Get the mean of the signal over a sliding window
    avg_length  = pre_avg + post_avg + 1
    avg_origin  = int(np.floor((pre_avg - post_avg)/2))
    mov_avg     = scipy.ndimage.filters.uniform_filter1d(x, avg_length, 
                                                            mode='constant', 
                                                            origin=avg_origin)

    # First mask out all entries not equal to the local max
    detections = x*(x == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections*(detections >= mov_avg + delta)
    
    # Initialize peaks array, to be filled greedily
    peaks = []
    
    # Remove onsets which are close together in time
    last_onset = -np.inf
    
    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i
    
    return np.array( peaks )

