Changes
=======

##v0.2.1

## v0.2.0

Bug fixes
    - fixed default ``librosa.core.stft, istft, ifgram`` to match specification

    - fixed a float->int bug in peak_pick
    
    - better memory efficiency
    
    - ``librosa.segment.recurrence_matrix`` corrects for width suppression
    
    - fixed a divide-by-0 error in the beat tracker
    
    - fixed a bug in tempo estimation with short windows
    
    - ``librosa.feature.sync`` now supports 1d arrays
    
    - fixed a bug in beat trimming
    
    - fixed a bug in ``librosa.core.stft`` when calculating window size
    
    - fixed ``librosa.core.resample`` to support stereo signals

Features
    - added filters option to cqt
    - added window function support to istft
    - added an IPython notebook demo
    - added ``librosa.features.delta`` for computing temporal difference features
    - new ``examples`` scripts:  tuning, hpss
    - added optional trimming to ``librosa.segment.stack_memory``
    - ``librosa.onset.onset_strength`` now takes generic spectrogram function ``feature`` 
    - compute reference power directly in ``librosa.core.logamplitude``
    - color-blind-friendly default color maps in ``librosa.display.cmap``
    - ``librosa.core.onset_strength`` now accepts an aggregator
    - added ``librosa.feature.perceptual_weighting``
    - added tuning estimation to ``librosa.feature.chromagram``
    - added ``librosa.core.A_weighting``
    - vectorized frequency converters
    - added ``librosa.core.cqt_frequencies`` to get CQT frequencies
    - ``librosa.core.cqt`` basic constant-Q transform implementation
    - ``librosa.filters.cq_to_chroma`` to convert log-frequency to chroma
    - added ``librosa.core.fft_frequencies``
    - ``librosa.decompose.hpss`` can now return masking matrices
    - added reversal for ``librosa.segment.structure_feature``
    - added ``librosa.core.time_to_frames``
    - added cent notation to ``librosa.core.midi_to_note``
    - added time-series or spectrogram input options to ``chromagram``, ``logfsgram``,
      ``melspectrogram``, and ``mfcc``
    - new module: ``librosa.display``
    - ``librosa.output.segment_csv`` => ``librosa.output.frames_csv``
    - migrated frequency converters to ``librosa.core``
    - new module: ``librosa.filters``
    - ``librosa.decompose.hpss`` now supports complex-valued STFT matrices
    - ``librosa.decompose.decompose()`` supports ``sklearn`` decomposition objects
    - added ``librosa.core.phase_vocoder``
    - new module: ``librosa.onset``; migrated onset strength from ``librosa.beat``
    - added ``librosa.core.pick_peaks``
    - ``librosa.core.load()`` supports offset and duration parameters
    - ``librosa.core.magphase()`` to separate magnitude and phase from a complex matrix
    - new module: ``librosa.segment``

Other changes
    - ``onset_estimate_bpm => estimate_tempo``
    - removed ``n_fft`` from ``librosa.core.istft()``
    - ``librosa.core.mel_frequencies`` returns ``n_mels`` values by default
    - changed default ``librosa.decompose.hpss`` window to 31
    - disabled onset de-trending by default in ``librosa.onset.onset_strength``
    - added complex-value warning to ``librosa.display.specshow``
    - broke compatibilty with ``ifgram.m``; ``librosa.core.ifgram`` now matches ``stft``
    - changed default beat tracker settings
    - migrated ``hpss`` into ``librosa.decompose``
    - changed default ``librosa.decompose.hpss`` power parameter to ``2.0``
    - ``librosa.core.load()`` now returns single-precision by default
    - standardized ``n_fft=2048``, ``hop_length=512`` for most functions
    - refactored tempo estimator

## v0.1.0

Initial public release.
