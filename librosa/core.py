#!/usr/bin/env python
"""Core IO, DSP and utility functions."""

import os.path
import audioread

from . import filters, feature, util

import re

import numpy as np
import numpy.fft as fft
import scipy.signal
import scipy.ndimage

# Do we have scikits.samplerate?
try:
    # Pylint won't handle dynamic imports, so we suppress this warning
    import scikits.samplerate as samplerate     # pylint: disable=import-error
    _HAS_SAMPLERATE = True
except ImportError:
    _HAS_SAMPLERATE = False



#-- CORE ROUTINES --#
def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    """Load an audio file as a floating point time series.

    :usage:
        >>> # Load a wav file
        >>> y, sr = librosa.load('file.wav')

        >>> # Load a wav file and resample to 11 KHz
        >>> y, sr = librosa.load('file.wav', sr=11025)

        >>> # Load 5 seconds of a wav file, starting 15 seconds in
        >>> y, sr = librosa.load('file.wav', offset=15.0, duration=5.0)

    :parameters:
      - path : string
          path to the input file.  
          Any format supported by ``audioread`` will work.

      - sr   : int > 0
          target sampling rate
          'None' uses the native sampling rate

      - mono : bool
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
          sampling rate of ``y``

    """

    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate

        s_start = int(np.floor(sr_native * offset) * input_file.channels)
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

    :usage:
        >>> # Downsample from 22 KHz to 8 KHz
        >>> y, sr   = librosa.load('file.wav', sr=22050)
        >>> y_8k    = librosa.resample(y, sr, 8000)

    :parameters:
      - y           : np.ndarray
          audio time series 

      - orig_sr     : int
          original sampling rate of ``y``

      - target_sr   : int
          target sampling rate

      - res_type    : str
          resample type (see note)
    
    :returns:
      - y_hat       : np.ndarray
          ``y`` resampled from ``orig_sr`` to ``target_sr``

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

def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None):
    """Short-time Fourier transform.

    Returns a complex-valued matrix D such that
      - ``np.abs(D[f, t])`` is the magnitude of frequency bin ``f`` at time ``t``
      - ``np.angle(D[f, t])`` is the phase of frequency bin ``f`` at time ``t``

    :usage:
        >>> y, sr = librosa.load('file.wav')
        >>> D = librosa.stft(y)

    :parameters:
      - y           : np.ndarray, real-valued
          the input signal (audio time series)

      - n_fft       : int
          FFT window size

      - hop_length  : int
          number audio of frames between STFT columns.
          If unspecified, defaults ``win_length / 4``.

      - win_length  : int <= n_fft
          Each frame of audio is windowed by the ``window`` function (see below).
          The window will be of length ``win_length`` and then padded with zeros
          to match ``n_fft``.

          If unspecified, defaults to ``win_length = n_fft``.

      - window      : None, function, np.ndarray
          - None (default): use an asymmetric Hann window
          - a window function, such as ``scipy.signal.hanning``
          - a vector or array of length ``n_fft``

    :returns:
      - D           : np.ndarray, dtype=complex
          STFT matrix

    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window
        fft_window = scipy.signal.hann(win_length, sym=False)

    elif hasattr(window, '__call__'):
        # User supplied a window function
        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
        if fft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    lpad        = (n_fft - win_length)/2
    fft_window  = np.pad(fft_window, (lpad, n_fft - win_length - lpad), mode='constant')
    
    # Reshape so that the window can be broadcast
    fft_window  = fft_window.reshape((-1, 1))

    # Window the time series. 
    y_frames    = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # RFFT and Conjugate here to match phase from DPWE code
    stft_matrix = fft.rfft(fft_window * y_frames, axis=0).conj().astype(np.complex64)

    return stft_matrix

def istft(stft_matrix, hop_length=None, win_length=None, window=None):  
    """
    Inverse short-time Fourier transform.

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``.

    :usage:
        >>> y, sr   = librosa.load('file.wav')
        >>> D       = librosa.stft(y)
        >>> y_hat   = librosa.istft(D)

    :parameters:
      - stft_matrix : np.ndarray, shape=(1 + n_fft/2, t)
          STFT matrix from ``stft()``

      - hop_length  : int
          Number of frames between STFT columns.
          If unspecified, defaults to ``win_length / 4``.

      - win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
          When reconstructing the time series, each frame is windowed
          according to the ``window`` function (see below).
          
          If unspecified, defaults to ``n_fft``.

      - window      : None, function, np.ndarray
          - None (default): use an asymmetric Hann window * 2/3
          - a window function, such as ``scipy.signal.hanning``
          - a user-specified window vector of length ``n_fft``

    :returns:
      - y           : np.ndarray
          time domain signal reconstructed from ``stft_matrix``

    """

    n_fft       = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = win_length / 4

    if window is None: 
        # Default is an asymmetric Hann window.
        # 2/3 scaling is to make stft(istft(.)) identity for 25% hop
        ifft_window =  scipy.signal.hann(win_length, sym=False) * (2.0 / 3)

    elif hasattr(window, '__call__'):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and window size')

    # Pad out to match n_fft
    lpad = (n_fft - win_length)/2
    ifft_window = np.pad( ifft_window, (lpad, n_fft - win_length - lpad), mode='constant')

    n_frames    = stft_matrix.shape[1]
    y           = np.zeros(n_fft + hop_length * (n_frames - 1))

    for i in xrange(n_frames):
        sample  = i * hop_length
        spec    = stft_matrix[:, i].flatten()
        spec    = np.concatenate((spec.conj(), spec[-2:0:-1] ), 0)
        ytmp    = ifft_window * fft.ifft(spec).real

        y[sample:(sample+n_fft)] = y[sample:(sample+n_fft)] + ytmp

    return y

def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None, norm=False):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as 
    described by Toshihiro Abe et al. in ICASSP'95, Eurospeech'97. 
    
    Calculates regular STFT as a side effect.

    :usage:
        >>> y, sr = librosa.load('file.wav')
        >>> frequencies, D = librosa.ifgram(y, sr=sr)

    :parameters:
      - y       : np.ndarray
          audio time series

      - sr      : int > 0
          sampling rate of ``y``

      - n_fft   : int > 0
          FFT window size

      - hop_length : int > 0
          hop length, number samples between subsequent frames.
          If not supplied, defaults to ``win_length / 4``.

      - win_length : int > 0, <= n_fft
          Window length. Defaults to n_fft.
          See ``stft()`` for details.

      - norm : bool
          Normalize the STFT. 

    :returns:
      - if_gram : np.ndarray, dtype=real
          Instantaneous frequency spectrogram:
          ``if_gram[f, t]`` is the frequency at bin ``f``, time ``t``

      - D : np.ndarray, dtype=complex
          Short-time Fourier transform

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


    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length / 4

    # Construct a padded hann window
    lpad = (n_fft - win_length)/2
    window = np.pad( scipy.signal.hann(win_length, sym=False), 
                        (lpad, n_fft - win_length - lpad), 
                        mode='constant')

    # Window for discrete differentiation
    freq_angular    = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)

    d_window        = np.sin( - freq_angular ) * np.pi / n_fft

    # Reshape windows for broadcast
    window          = window.reshape((-1, 1))
    d_window        = d_window.reshape((-1, 1))

    # Pylint does not correctly infer the type here, but it's correct.
    freq_angular    = freq_angular.reshape((-1, 1)) # pylint: disable=maybe-no-member
    
    # Frame up the audio
    y_frame         = util.frame(y, frame_length=n_fft, hop_length=hop_length)
    
    # compute STFT and differential spectrogram
    stft_matrix     = fft.rfft(window   * y_frame, axis=0).conj()
    diff_stft       = fft.rfft(d_window * y_frame, axis=0)
    
    # Compute power normalization. Suppress zeros.
    power               = np.abs(stft_matrix)**2
    power[power == 0]   = 1.0
    
    if_gram = (freq_angular[:n_fft/2 + 1] + (stft_matrix * diff_stft).imag / power) * sr / (2 * np.pi)
    
    if norm:
        stft_matrix = stft_matrix * 2.0 / window.sum()

    return if_gram, stft_matrix

def cqt(y, sr, hop_length=512, fmin=None, fmax=None, bins_per_octave=12, tuning=None, 
        resolution=2, aggregate=None, samples=None, basis=None):
    '''Compute the constant-Q transform of an audio signal.
    
    :usage:
        >>> y, sr = librosa.load('file.wav')
        >>> C = librosa.cqt(y, sr)

        >>> # Limit the frequency range
        >>> C = librosa.cqt(y, sr, fmin=librosa.midi_to_hz(36), fmax=librosa.midi_to_hz(96))

        >>> # Use a pre-computed CQT basis
        >>> basis = librosa.filters.constant_q(sr, ...)
        >>> C = librosa.cqt(y, sr, basis=basis)

    :parameters:
      - y : np.ndarray
          audio time series
    
      - sr : int > 0
          sampling rate of ``y``
        
      - hop_length : int > 0
          number of samples between successive CQT columns.
    
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
          Default: np.mean
        
      - samples : None or array-like
          Aggregate power at times ``y[samples[i]:samples[i+1]]``, 
          instead of ``y[i * hop_length : (i+1)*hop_length]``
        
          Note that boundary sample times ``(0, len(y))`` will be automatically added.

      - basis : None or list of arrays
          (optinal) alternate set of CQT basis filters.
          See ``librosa.filters.constant_q`` for details.

    :returns:
      - CQT : np.ndarray
          Constant-Q power for each frequency at each time.    
    '''

    if aggregate is None:
        aggregate = np.mean

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
    if basis is None:
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

    cqt_power = np.empty((len(basis), len(y)), dtype=np.float32, order='F')
    
    for i, filt in enumerate(basis):
        cqt_power[i]  = np.abs(scipy.signal.fftconvolve(y, filt, mode='same'))**2
    
    cqt_power = feature.sync(cqt_power, samples, aggregate=aggregate)
    
    return cqt_power
    
def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram.

    :usage:
        >>> # Get a power spectrogram from a waveform y
        >>> S       = np.abs(librosa.stft(y)) ** 2
        >>> log_S   = librosa.logamplitude(S)

        >>> # Compute dB relative to peak power
        >>> log_S   = librosa.logamplitude(S, ref_power=S.max())

    :parameters:
      - S       : np.ndarray
          input spectrogram

      - ref_power : float
          reference against which ``S`` is compared.

      - amin    : float
          minimum amplitude threshold 

      - top_db  : float
          threshold log amplitude at top_db below the peak:
          ``max(log(S)) - top_db``

    :returns:
      log_S   : np.ndarray
          ``log_S ~= 10 * log10(S) - 10 * log10(abs(ref_power))``
    """

    log_spec    =   10.0 * np.log10(np.maximum(amin, np.abs(S))) 
    log_spec    -=  10.0 * np.log10(np.abs(ref_power))

    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that ``D = S * P``.

    :usage:
        >>> D = librosa.stft(y)
        >>> S, P = librosa.magphase(D)
        >>> D == S * P

    :parameters:
      - D       : np.ndarray, dtype=complex
          complex-valued spectrogram

    :returns:
      - D_mag   : np.ndarray, dtype=real
          magnitude of ``D``
      - D_phase : np.ndarray, dtype=complex
          ``exp(1.j * phi)`` where ``phi`` is the phase of ``D``
    """

    mag   = np.abs(D)
    phase = np.exp(1.j * np.angle(D))

    return mag, phase

def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of ``rate``

    :usage:
        >>> # Play at double speed
        >>> y, sr   = librosa.load('file.wav')
        >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
        >>> D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
        >>> y_fast  = librosa.istft(D_fast, hop_length=512)

        >>> # Or play at 1/3 speed
        >>> D_slow  = librosa.phase_vocoder(D, 1./3, hop_length=512)
        >>> y_slow  = librosa.istft(D_slow, hop_length=512)

    :parameters:
      - D       : np.ndarray, dtype=complex
          STFT matrix

      - rate    :  float, positive
          Speed-up factor: ``rate > 1`` is faster, ``rate < 1`` is slower.

      - hop_length : int or None
          The number of samples between successive columns of ``D``.
          If None, defaults to ``n_fft/4 = (D.shape[0]-1)/2``

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

    :usage:
        >>> librosa.note_to_midi('C')
        0
        >>> librosa.note_to_midi('C#3')
        37
        >>> librosa.note_to_midi('f4')
        53
        >>> librosa.note_to_midi('Bb-1')
        -2
        >>> librosa.note_to_midi('A!8') 
        104

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

    :usage:
        >>> librosa.midi_to_note(0)
        'C0'
        >>> librosa.midi_to_note(37)
        'C#3'
        >>> librosa.midi_to_note(-2)
        'A#-1'
        >>> librosa.midi_to_note(104.7)
        'A8'
        >>> librosa.midi_to_note(104.7, cents=True)
        'A8-30'

    :parameters:
      - midi : int or iterable of int
          Midi numbers to convert.

      - octave: bool
          If True, include the octave number

      - cents: bool
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

    :usage:
        >>> librosa.midi_to_hz(36)
        array([ 65.40639133])

        >>> librosa.midi_to_hz(np.arange(36, 48))
        array([  65.40639133,   69.29565774,   73.41619198,   77.78174593,
                 82.40688923,   87.30705786,   92.49860568,   97.998859  ,
                103.82617439,  110.        ,  116.54094038,  123.47082531])

    :parameters:
      - notes       : int, np.ndarray
          midi number(s) of the note(s)

    :returns:
      - frequency   : float, np.ndarray
          frequency (frequencies) of ``notes`` in Hz
    """

    notes = np.asarray([notes]).flatten()
    return 440.0 * (2.0 ** ((notes - 69)/12.0))

def hz_to_midi( frequencies ):
    """Get the closest MIDI note number(s) for given frequencies

    :usage:
        >>> librosa.hz_to_midi(60)
        array([ 34.50637059])
        >>> librosa.hz_to_midi([110, 220, 440])
        array([ 45.,  57.,  69.])

    :parameters:
      - frequencies   : float, np.ndarray
          frequencies to convert

    :returns:
      - note_nums     : int, np.ndarray
          closest MIDI notes to ``frequencies``
    """

    frequencies = np.asarray([frequencies]).flatten()
    return 12 * (np.log2(frequencies) - np.log2(440.0)) + 69

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    :usage:
        >>> librosa.hz_to_mel(60)
        array([0.9])
        >>> librosa.hz_to_mel([110, 220, 440])
        array([ 1.65,  3.3 ,  6.6 ])

    :parameters:
      - frequencies   : np.ndarray, float
          scalar or array of frequencies
      - htk           : bool
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

    :usage:
        >>> librosa.mel_to_hz(3)
        array([ 200.])

        >>> librosa.mel_to_hz([1,2,3,4,5])
        array([  66.66666667,  133.33333333,  200.        ,  266.66666667,
                333.33333333])

    :parameters:
      - mels          : np.ndarray, float
          mel bins to convert
      - htk           : bool
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
    """Convert frequencies (Hz) to (fractional) octave numbers.

    :usage:
        >>> librosa.hz_to_octs(440.0)
        array([ 4.])
        >>> librosa.hz_to_octs([32, 64, 128, 256])
        array([ 0.21864029,  1.21864029,  2.21864029,  3.21864029])

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
    """Convert octaves numbers to frequencies.

    Octaves are counted relative to A.

    :usage:
        >>> librosa.octs_to_hz(1)
        array([ 55.])
        >>> librosa.octs_to_hz([-2, -1, 0, 1, 2])
        array([   6.875,   13.75 ,   27.5  ,   55.   ,  110.   ])

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
    return (A440/16)*(2.0**octs)

def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of ``np.fft.fftfreqs``

    :usage:
        >>> librosa.fft_frequencies(sr=22050, n_fft=16)
        array([     0.   ,   1378.125,   2756.25 ,   4134.375,   5512.5  ,
                 6890.625,   8268.75 ,   9646.875,  11025.   ])

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

    :usage:
        >>> # Get the CQT frequencies for 24 notes, starting at C2
        >>> librosa.cqt_frequencies(24, fmin=librosa.midi_to_hz(librosa.note_to_midi('C2')))
        array([  32.70319566,   34.64782887,   36.70809599,   38.89087297,
                 41.20344461,   43.65352893,   46.24930284,   48.9994295 ,
                 51.9130872 ,   55.        ,   58.27047019,   61.73541266,
                 65.40639133,   69.29565774,   73.41619198,   77.78174593,
                 82.40688923,   87.30705786,   92.49860568,   97.998859  ,
                103.82617439,  110.        ,  116.54094038,  123.47082531])

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

    :usage:
        >>> librosa.mel_frequencies(n_mels=40)
        array([    0.        ,    81.15543818,   162.31087636,   243.46631454,
                324.62175272,   405.7771909 ,   486.93262907,   568.08806725,
                649.24350543,   730.39894361,   811.55438179,   892.70981997,
                973.86525815,  1058.38224675,  1150.77458676,  1251.23239132,
                1360.45974173,  1479.22218262,  1608.3520875 ,  1748.75449257,
                1901.4134399 ,  2067.39887435,  2247.87414245,  2444.10414603,
                2657.46420754,  2889.44970936,  3141.68657445,  3415.94266206,
                3714.14015814,  4038.36904745,  4390.90176166,  4774.2091062 ,
                5190.97757748,  5644.12819182,  6136.83695801,  6672.55713712,
                7255.04344548,  7888.37837041,  8577.0007833 ,  9325.73705043])

    :parameters:
      - n_mels    : int
          number of Mel bins  
      - fmin      : float
          minimum frequency (Hz)
      - fmax      : float
          maximum frequency (Hz)
      - htk       : bool
          use HTK formula instead of Slaney
      - extra     : bool
          include extra frequencies necessary for building Mel filters

    :returns:
      - bin_frequencies : ndarray
          vector of Mel frequencies

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel  = hz_to_mel(fmin, htk=htk)
    maxmel  = hz_to_mel(fmax, htk=htk)

    mels    = np.arange(minmel, maxmel + 1, (maxmel - minmel)/(n_mels + 1.0))

    if not extra:
        mels = mels[:n_mels]

    return  mel_to_hz(mels, htk=htk)

# A-weighting should be capitalized: suppress the naming warning
def A_weighting(frequencies, min_db=-80.0):     # pylint: disable=invalid-name
    '''Compute the A-weighting of a set of frequencies.

    :usage:
        >>> # Get the A-weighting for 20 Mel frequencies
        >>> freqs   = librosa.mel_frequencies(20)
        >>> librosa.A_weighting(freqs)
        array([-80.        , -13.35467911,  -6.59400464,  -3.57422971,
                -1.87710933,  -0.83465455,  -0.15991521,   0.3164558 ,
                0.68372258,   0.95279329,   1.13498903,   1.23933477,
                1.27124465,   1.23163355,   1.1163366 ,   0.91575476,
                0.6147545 ,   0.1929889 ,  -0.37407714,  -1.11314196])

    :parameters:
      - frequencies : scalar or np.ndarray
          One or more frequencies (in Hz)

      - min_db : float or None
          Clip weights below this threshold.
          If ``None``, no clipping is performed.

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

    weights = 2.0 + 20 * np.log10(r_a)
    
    if min_db is not None:
        weights = np.maximum(min_db, weights)

    return weights

#-- UTILITIES --#
def frames_to_time(frames, sr=22050, hop_length=512):
    """Converts frame counts to time (seconds)

    :usage:
        >>> y, sr = librosa.load('file.wav')
        >>> tempo, beats = librosa.beat.beat_track(y, sr, hop_length=64)
        >>> beat_times   = librosa.frames_to_time(beats, sr, hop_length=64)

    :parameters:
      - frames     : np.ndarray
          vector of frame numbers

      - sr         : int > 0
          audio sampling rate 

      - hop_length : int
          number of samples between successive frames

    :returns:
      - times : np.ndarray 
          time (in seconds) of each given frame number:
          ``times[i] = frames[i] * hop_length / sr``

    """
    return (frames * hop_length) / float(sr)

def time_to_frames(times, sr=22050, hop_length=512):
    """Converts time stamps into STFT frames.

    :usage:
        >>> # Get the frame numbers for every 100ms
        >>> librosa.time_to_frames(np.arange(0, 1, 0.1), sr=22050, hop_length=512)
        array([ 0,  4,  8, 12, 17, 21, 25, 30, 34, 38])

    :parameters:
      - times : np.ndarray
          vector of time stamps

      - sr : int > 0
          audio sampling rate

      - hop_length : int > 0
          number of samples between successive frames

    :returns:
      - frames : np.ndarray, dtype=int
          Frame numbers corresponding to the given times:
          ``frames[i] = floor( times[i] * sr / hop_length )``
    """
    return np.floor(times * np.float(sr) / hop_length).astype(int)

def autocorrelate(y, max_size=None):
    """Bounded auto-correlation

    :usage:
        >>> # Compute full autocorrelation of y
        >>> y, sr   = librosa.load('file.wav')
        >>> y_ac    = librosa.autocorrelate(y)

        >>> # Compute autocorrelation up to 4 seconds lag
        >>> y_ac_4  = librosa.autocorrelate(y, 4 * sr)

    :parameters:
      - y         : np.ndarray
          vector to autocorrelate

      - max_size  : int
          maximum correlation lag.
          If unspecified, defaults to ``len(y)`` (unbounded)

    :returns:
      - z         : np.ndarray
          truncated autocorrelation ``y*y``

    """

    result = scipy.signal.fftconvolve(y, y[::-1], mode='full')

    result = result[len(result)/2:]

    if max_size is None:
        return result
    else:
        max_size = int(max_size)
    
    return result[:max_size]

def localmax(x):
    """Return 1 where there are local maxima in x (column-wise)
       left edges do not fire, right edges might.

    :usage:
        >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
        >>> librosa.localmax(x)
        array([False, False, False,  True, False,  True, False, True], dtype=bool)

    :parameters:
      - x     : np.ndarray
          input vector

    :returns:
      - m     : np.ndarray, dtype=bool
          indicator vector of local maxima:
          ``m[i] == True`` if ``x[i]`` is a local maximum
    """

    return np.logical_and(x > np.hstack([x[0], x[:-1]]), 
                             x >= np.hstack([x[1:], x[-1]]))

def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    '''Uses a flexible heuristic to pick peaks in a signal.

    :usage:
        >>> # Look +-3 steps
        >>> # compute the moving average over +-5 steps
        >>> # peaks must be > avg + 0.5
        >>> # skip 10 steps before taking another peak
        >>> librosa.peak_pick(x, 3, 3, 5, 5, 0.5, 10)
    
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

        1. ``x[n] == max(x[n - pre_max:n + post_max])``
        2. ``x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta``
        3. ``n - previous_n > wait``

      where ``previous_n`` is the last sample picked as a peak (greedily).
    
    .. note::
        @inproceedings{bock2012evaluating,
            title={Evaluating the Online Capabilities of Onset Detection Methods.},
            author={B{\"o}ck, Sebastian and Krebs, Florian and Schedl, Markus},
            booktitle={ISMIR},
            pages={49--54},
            year={2012}}
    
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

