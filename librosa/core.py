#!/usr/bin/env python
"""Core IO, DSP and utility functions."""

import os.path
import audioread

from . import filters, feature

import re

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
def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
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

      - dtype : numeric type
          data type of y

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


        scale = float(1<<15)

        y = []
        n = 0

        for frame in input_file:
            frame   = np.frombuffer(frame, '<i2').astype(dtype)
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


        y = np.concatenate(y) / scale
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

def stft(y, n_fft=2048, hop_length=None, hann_w=None, window=None):
    """Short-time fourier transform

    :parameters:
      - y           : np.ndarray
          the input signal

      - n_fft       : int
          number of FFT components

      - hop_length  : int
          number audio of frames between STFT columns.
          If unspecified, defaults hann_w / 4.

      - hann_w      : int
          The size of Hann window. 
          If unspecified, defaults to n_fft

      - window      : np.ndarray
          (optional) user-specified window

    :returns:
      - D           : np.ndarray, dtype=complex
          STFT matrix

    """

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
        hop_length = int(n_fft / 4)

    n_specbins  = 1 + int(n_fft / 2)
    n_frames    = 1 + int( (len(y) - n_fft) / hop_length)

    # allocate output array
    stft_matrix = np.empty( (n_specbins, n_frames), dtype=np.complex64)

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
          If unspecified, defaults to hann_w / 4.

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
        hop_length = n_fft / 4

    y = np.zeros(n_fft + hop_length * (n_frames - 1))

    for i in xrange(n_frames):
        sample  = i * hop_length
        spec    = stft_matrix[:, i].flatten()
        spec    = np.concatenate((spec.conj(), spec[-2:0:-1] ), 0)
        ytmp    = window * fft.ifft(spec).real

        y[sample:(sample+n_fft)] = y[sample:(sample+n_fft)] + ytmp

    return y

def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None, norm=False):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as 
    described by Toshihiro Abe et al. in ICASSP'95, Eurospeech'97. 
    
    Calculates regular STFT as a side effect.

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

      - norm : bool
          Normalize the STFT. 

    :returns:
      - if_gram : np.ndarray, dtype=real
          Instantaneous frequency spectrogram:
          if_gram[f, t] is the frequency at bin f, time t

      - D : np.ndarray, dtype=complex
          Short-time fourier transform

    .. note::

      @inproceedings{abe1995harmonics,
            title={Harmonics tracking and pitch extraction based on instantaneous frequency},
            author={Abe, Toshihiko and Kobayashi, Takao and Imai, Satoshi},
            booktitle={Acoustics, Speech, and Signal Processing, 1995. ICASSP-95., 1995 International Conference on},
            volume={1},
            pages={756--759},
            year={1995},
            organization={IEEE}
            }
    '''

    if hop_length is None:
        hop_length = n_fft / 4

    if win_length is None:
        win_length = n_fft

    # Construct a padded hann window
    lpad = (n_fft - win_length)/2
    window = np.pad( scipy.signal.hann(win_length, sym=False), 
                        (lpad, n_fft - win_length - lpad), 
                        mode='constant')

    # Window for discrete differentiation
    freq_angular    = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)
    d_window        = np.sin( - freq_angular ) * np.pi / n_fft

    # Construct output arrays
    if_gram = np.zeros((1 + n_fft / 2, 1 + (len(y) - n_fft) / hop_length), dtype=np.float32)
    D       = np.zeros_like(if_gram, dtype=np.complex64)

    # Main loop: fill in if_gram and D
    for i in xrange(D.shape[1]):
        sample = i * hop_length

        #-- Store the STFT
        # Conjugate here to match DWPE's matlab code.
        frame   = fft.fft(window * y[sample:(sample + n_fft)]).conj()
        D[:, i] = frame[:D.shape[0]]

        # Compute power per bin in this frame
        power               = np.abs(frame)**2
        power[power == 0]   = 1.0
        
        #-- Calculate the instantaneous frequency 
        # phase of differential spectrum
        d_frame = fft.fft(d_window * y[sample:(sample + n_fft)])

        t = freq_angular + (d_frame * frame).imag / power

        if_gram[:, i] = t[:if_gram.shape[0]] * sr / (2 * np.pi)

    # Compensate for windowing effects, store STFT
    # sum(window) takes out integration due to window, 2 compensates for negative
    # frequency
    if norm:
        D = D * 2.0 / window.sum()

    return if_gram, D

def cqt(y, sr, hop_length=512, fmin=None, fmax=None, bins_per_octave=12, tuning=None, 
        resolution=1, aggregate=np.mean, samples=None):
    '''Compute the constant-Q transform of an audio signal.
    
    :parameters:
      - y : np.ndarray
          audio time series
    
      - sr : int > 0
          sampling rate of y
        
      - hop_length : int > 0
          hop length 
    
      - fmin : float > 0
          Minimum frequency. Defaults to C1 ~= 16.35 Hz
        
      - fmax : float > 0
          Maximum frequency. Defaults to C9 ~= 4816.01 Hz
        
      - bins_per_octave : int > 0
          Number of bins per octave
        
      - tuning : None or float in [-0.5, 0.5)
          Tuning offset in fractions of a bin (cents)
          If None, tuning will be automatically estimated.
        
      - resolution : float > 0
          Filter resolution factor. Larger values use longer windows.
        
      - aggregate : function
          Aggregator function to merge filter response power within frames.
        
      - samples : None or array-like
          Aggregate power at times ``y[samples[i]:samples[i+1]]``, 
          instead of ``y[i * hop_length : (i+1)*hop_length]``
        
          Note that boundary sample times (0, len(y)) will be automatically added.
        
    :returns:
      - CQT : np.ndarray
          Constant-Q power for each frequency at each time.    
    '''
    
    # Do we have tuning?
    def __get_tuning():
        '''Helper function to compute tuning from y,sr'''
        pitches, mags = feature.ifptrack(y, sr=sr)[:2]
        threshold = np.median(mags)
        return feature.estimate_tuning( pitches[mags>threshold], 
                                        bins_per_octave=bins_per_octave)

    if tuning is None:
        tuning = __get_tuning()

    # Generate the CQT filters
    basis = filters.constant_q(sr, 
                        fmin=fmin, 
                        fmax=fmax, 
                        bins_per_octave=bins_per_octave, 
                        tuning=tuning, 
                        resolution=resolution)
    
    if samples is None:
        samples    = np.arange(0, len(y), hop_length)
    else:
        samples    = np.asarray([samples]).flatten()
    
    cqt_power = []
    for filt in basis:
        r_power  = np.abs(scipy.signal.fftconvolve(y, filt, mode='same'))**2
        cqt_power.append(feature.sync(r_power, samples, aggregate=aggregate))
    
    return np.asarray(cqt_power).squeeze()

def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram.

    :parameters:
      - S       : np.ndarray
          input spectrogram

      - ref_power : float
          reference power against which S is compared

      - amin    : float
          minimum amplitude threshold 

      - top_db  : float
          threshold log amplitude at top_db below the peak:
          ``max(log(S)) - top_db``

    :returns:
      log_S   : np.ndarray
          ``log_S ~= 10 * log10(S) - 10 * log10(ref_power)``

    """

    log_spec    =   10.0 * np.log10(np.maximum(amin, np.abs(S))) 
    log_spec    -=  10.0 * np.log10(ref_power)

    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that D = S * P.

    :parameters:
      - D       : np.ndarray, dtype=complex
          input complex-valued spectrogram

    :returns:
      - D_mag   : np.ndarray, dtype=real
          magnitude of D
      - D_phase : np.ndarray, dtype=complex
          exp(1.j * phi) where phi is the phase of D

    """

    mag   = np.abs(D)
    phase = np.exp(1.j * np.angle(D))

    return mag, phase

def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, time-stretch by a 
    factor of rate.

    :parameters:
      - D       : np.ndarray, dtype=complex
          STFT matrix

      - rate    :  float, positive
          time-stretch factor

      - hop_length : int or None
          hop length of D.  If None, defaults to n_fft/4 = (D.shape[0]-1)/2

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
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype)
    
    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])
    
    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):
        
        columns  = D[:, int(step):int(step + 2)]
        
        # Weighting for linear magnitude interpolation
        alpha   = np.mod(step, 1.0)
        mag   = (1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, 1])
        
        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase  = np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance
        
        # Wrap to -pi:pi range
        dphase  = dphase - 2*np.pi * np.round(dphase / (2*np.pi))
        
        # Accumulate phase
        phase_acc += phi_advance + dphase
    
    return d_stretch

#-- FREQUENCY UTILITIES AND CONVERTERS--#
def note_to_midi(note):
    '''Convert one or more spelled notes to MIDI number(s).
    
    Notes may be spelled out with optional accidentals or octave numbers.

    The leading note name is case-insensitive.

    Sharps are indicated with ``#``, flats may be indicated with ``!`` or ``b``.

    For example:

    - ``note_to_midi('C') == 0``
    - ``note_to_midi('C#3') == 37``
    - ``note_to_midi('f4') == 53``
    - ``note_to_midi('Bb-1') == -2``
    - ``note_to_midi('A!8') == 104``

    :parameters:
      - note : str or iterable of str
        One or more note names.

    :returns:
      - midi : int or np.array
        Midi note numbers corresponding to inputs.
    '''

    if not isinstance(note, str):
        return np.array(map(note_to_midi, note))
    
    pitch_map   = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map     = {'#': 1, '': 0, 'b': -1, '!': -1}
    
    try:
        match = re.match(r'^(?P<note>[A-Ga-g])(?P<offset>[#b!]?)(?P<octave>[+-]?\d+)$', note)
        
        pitch = match.group('note').upper()
        offset = acc_map[match.group('offset')]
        octave = int(match.group('octave'))
    except:
        raise ValueError('Improper note format: %s' % note)
    
    return 12 * octave + pitch_map[pitch] + offset

def midi_to_note(midi, octave=True, cents=False):
    '''Convert one or more MIDI numbers to note strings.

    MIDI numbers will be rounded to the nearest integer.

    Notes will be of the format 'C0', 'C#0', 'D0', ...

    :parameters:
      - midi : int or iterable of int
          Midi numbers to convert.

      - octave: boolean
          If true, include the octave number

      - cents: boolean
          If true, cent markers will be appended for fractional notes.
          Eg, ``midi_to_note(69.3, cents=True)`` == ``A5+03``

    :returns:
      - notes : str or iterable of str
          Strings describing each midi note.
    '''

    if not np.isscalar(midi):
        return map(lambda x: midi_to_note(x, octave=octave, cents=cents), midi)
    
    note_map    = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    note_num    = int(np.round(midi))
    note_cents  = int(100 * np.around(midi - note_num, 2))

    note        = note_map[note_num % 12]

    if octave:
        note = '%s%0d' % (note, note_num / 12)
    if cents:
        note = '%s%+02d' % (note, note_cents)

    return note

def midi_to_hz( notes ):
    """Get the frequency (Hz) of MIDI note(s)

    :parameters:
      - note_num      : int, np.ndarray
          number of the note(s)

    :returns:
      - frequency     : float, np.ndarray
          frequency of the note in Hz
    """

    notes = np.asarray([notes]).flatten()
    return 440.0 * (2.0 ** ((notes - 69)/12.0))

def hz_to_midi( frequencies ):
    """Get the closest MIDI note number(s) for given frequencies

    :parameters:
      - frequencies   : float, np.ndarray
          target frequencies

    :returns:
      - note_nums     : int, np.ndarray
          closest MIDI notes

    """

    frequencies = np.asarray([frequencies]).flatten()
    return 12 * (np.log2(frequencies) - np.log2(440.0)) + 69

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

    frequencies = np.asarray([frequencies]).flatten()
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

    mels = np.asarray([mels], dtype=float).flatten()

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
    frequencies = np.asarray([frequencies]).flatten()
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
    octs = np.asarray([octs]).flatten()
    return (A440/16)*(2**octs)

def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of ``np.fft.fftfreqs``

    :parameters:
      - sr : int > 0
          Audio sampling rate

      - n_fft : int > 0
          FFT window size

    :returns:
      - freqs : np.ndarray, shape = (1 + n_fft/2,)
          Frequencies (0, sr/n_fft, 2*sr/n_fft, ..., sr/2)
    '''

    return np.linspace(0, sr/2, 1 + n_fft/2, endpoint=True)

def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    """Compute the center frequencies of Constant-Q bins.

    :parameters:
      - n_bins  : int > 0
          Number of constant-Q bins

      - fmin    : float >0
          Minimum frequency

      - bins_per_octave : int > 0
          Number of bins per octave

      - tuning : float in [-0.5, +0.5)
          Deviation from A440 tuning in fractional bins (cents)

    :returns:
      - frequencies : np.ndarray, shape=(n_bins,)
          Center frequency for each CQT bin
    """

    correction = 2.0**(float(tuning) / bins_per_octave)

    return correction * fmin * 2.0**(np.arange(0, n_bins, dtype=float)/bins_per_octave)

def mel_frequencies(n_mels=40, fmin=0.0, fmax=11025.0, htk=False, extra=False):
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
      - extra     : boolean
          include extra frequencies necessary for building Mel filters

    :returns:
      - bin_frequencies : ndarray
          ``n_mels+1``-dimensional vector of Mel frequencies

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel  = hz_to_mel(fmin, htk=htk)
    maxmel  = hz_to_mel(fmax, htk=htk)

    mels    = np.arange(minmel, maxmel + 1, (maxmel - minmel)/(n_mels + 1.0))

    if not extra:
        mels = mels[:n_mels]

    return  mel_to_hz(mels, htk=htk)

def A_weighting(frequencies):
    '''Compute the A-weighting of a set of frequencies.

    :parameters:
      - frequencies : scalar or np.ndarray
          One or more frequencies (in Hz)

    :returns:
      - A_weighting : scalar or np.ndarray
          A[i] is the A-weighting of frequencies[i]
    '''

    # Vectorize to make our lives easier
    frequencies = np.asarray([frequencies]).flatten()

    # Pre-compute squared frequeny
    f_sq    = frequencies**2.0

    const   = np.array([12200, 20.6, 107.7, 737.9])**2.0

    r_a     = const[0] * f_sq**2
    r_a     /= (f_sq + const[0]) * (f_sq + const[1])
    r_a     /= np.sqrt((f_sq + const[2]) * (f_sq + const[3]))

    return 2.0 + 20 * np.log10(r_a)

#-- UTILITIES --#
def frames_to_time(frames, sr=22050, hop_length=512):
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

def time_to_frames(times, sr=22050, hop_length=512):
    """Converts time stamps into STFT frames.

    :parameters:
      - times : np.ndarray
          vector of time stamps

      - sr : int > 0
          Audio sampling rate

      - hop_length : int > 0
          Hop length of FFT.

    :returns:
      - frames : np.ndarray, dtype=int
          Frame numbers corresponding to the given times:
          ``frames[i] = floor( times[i] * sr / hop_length )``
    """
    return np.floor(times * np.float(sr) / hop_length).astype(int)

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
    max_origin  = 0.5 * (pre_max - post_max)
    mov_max     = scipy.ndimage.filters.maximum_filter1d(x, int(max_length), 
                                                            mode='constant', 
                                                            origin=int(max_origin))

    # Get the mean of the signal over a sliding window
    avg_length  = pre_avg + post_avg + 1
    avg_origin  = 0.5 * (pre_avg - post_avg)
    mov_avg     = scipy.ndimage.filters.uniform_filter1d(x, int(avg_length), 
                                                            mode='constant', 
                                                            origin=int(avg_origin))

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

