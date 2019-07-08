Changelog
=========

v0.7.0
------
2019-07-1

Note: the 0.7 series will be the last to officially support Python 2.7.


New features
   - `#772`_ `librosa.core.stream`: Stream generator to process long audio files into smaller pieces. *Brian McFee*
   - `#845`_ `librosa.core.load`: Replaced the default audio decoder with `pysoundfile`, and only use `audioread` as backup. *Brian McFee*
   - `#843`_ `librosa.core.griffinlim`: Phase retrieval from magnitude spectrograms using the (accelerated) Griffin-Lim method. *Brian McFee*
   - `#843`_ `librosa.feature.inverse`: New module for feature inversion, based on the Griffin-Lim phase retrieval algorithm. Includes `mel_to_audio` and `mfcc_to_audio`. *Brian McFee*
   - `#725`_ `librosa.core.lpc`: Linear prediction coefficients (LPC). *Adam Weiss*
   - `#907`_ `librosa.sequence.rqa`: Recurrence Quantification Analysis (RQA) for sequence alignment. *Brian McFee*
   - `#739`_ `librosa.beat.plp`: Predominant local pulse (PLP) for variable-tempo beat tracking. *Brian McFee*
   - `#894`_ `librosa.feature.fourier_tempogram`: Fourier Tempogram for representing rhythm in the frequency domain. *Brian McFee*
   - `#891`_ `librosa.core.pcen` Per-channel energy normalization (PCEN) now allows logarithmic range compression at the limit power->0. *Vincent Lostanlen*
   - `#863`_ `librosa.effects.pitch_shift` supports custom resampling modes. *Taewoon Kim*
   - `#857`_ `librosa.core.cqt` and `librosa.core.icqt` Forward and inverse constant-Q transform now support custom resampling modes. *Brian McFee*
   - `#842`_ `librosa.segment.path_enhance`: Near-diagonal path enhancement for recurrence, self- or cross-similarity matrices. *Brian McFee*
   - `#840`_ `librosa.segment.recurrence_matrix` now supports a keyword argument, `self=False`. If set to `True`, the recurrence matrix includes self-loops. *Brian McFee*
   - `#776`_ `librosa.core.piptrack` now supports a keyword argument, `ref=None`, allowing users to override the reference thresholding behavior for determining which bins correspond to pitches. *Brian McFee*
   - `#770`_ `librosa.segment.cross_similarity`: Cross-similarity function for comparing two feature sequences. *Rachel Bittner, Brian McFee*
   - `#709`_ `librosa.onset.onset_strength_multi` now supports a user-specified reference spectrum via the `ref` keyword argument. *Brian McFee*
   - `#576`_ `librosa.core.resample` now supports `mode='polyphase'`. *Brian McFee*
   - `#519`_ `librosa.onset.onset_strength_multi`: Setting `aggregate=False` disables the aggregation of onset strengths across frequency bins. *Brian McFee*


Bug fixes
   - `#900`_ `librosa.effects.pitch_shift` now preserves length. *Vincent Lostanlen*
   - `#891`_ `librosa.core.pcen` Dynamic range compression in PCEN is more numerically stable for small values of the exponent. *Vincent Lostanlen*
   - `#888`_ `librosa.core.ifgram` Instantaneous frequency spectrogram now correctly estimates center frequencies when using windows other than `hann`. *Brian McFee*
   - `#869`_ `librosa.sequence.dtw` Fixed a bug in dynamic time warping when `subseq=True`. *Viktor Andreevitch Morozov*
   - `#851`_ `librosa.core.pcen` now initializes its autoregressive filtering in the steady state, not with silence. *Jan Schlüter, Brian McFee*
   - `#833`_ `librosa.segment.recurrence_matrix`: `width` parameter now cannot exceed data length. *Brian McFee*
   - `#825`_ Filter bank constructors `mel`, `chroma`, `constant_q`, and `cq_to_chroma` are now type-stable. *Vincent Lostanlen, Brian McFee*
   - `#802`_ `librosa.core.icqt` Inverse constant-Q transform has been completely rewritten and is more numerically stable. *Brian McFee*


Removed features (deprecated in v0.6)
   - Discrete cosine transform. We recommend using `scipy.fftpack.dct`
   - The `delta` function no longer support the `trim` keyword argument. 
   - Root mean square error (`rmse`) has been renamed to `rms`.
   - `iirt` now uses `sos` mode by default.


Documentation
   - `#891`_ Improved the documentation of PCEN. *Vincent Lostanlen*
   - `#884`_ Improved installation documentation. *Darío Hereñú*
   - `#882`_ Improved code style for plot generation. *Alex Metsai*
   - `#874`_ Improved the documentation of spectral features. *Brian McFee*
   - `#804`_ Improved the documentation of MFCC. *Brian McFee*
   - `#849`_ Removed a redundant link in the `util` documentation. *Keunwoo Choi*
   - `#827`_ Improved the docstring of `recurrence_matrix`. *Brian McFee*
   - `#813`_ Improved the docstring of `load`. *Andy Sarroff*


Other changes
   - `#917`_ The `output` module is now deprecated, and will be removed in version 0.8.
   - `#878`_ More informative exception handling. *Jack Mason*
   - `#857`_ `librosa.core.resample()` now supports `mode='fft'`, equivalent to the previous `scipy` mode. *Brian McFee*
   - `#854`_ More efficient length-aware ISTFT and ICQT. *Vincent Lostanlen*
   - `#846`_ Nine librosa functions now store jit-compiled, numba-accelerated caches across sessions. *Brian McFee*
   - `#841`_ `librosa.core.load` no longer relies on `realpath()`. *Brian McFee*
   - `#834`_ All spectral feature extractors now expose all STFT parameters. *Brian McFee*
   - `#829`_ Refactored `librosa.cache`. *Brian McFee*
   - `#818`_ Thanks to `np.fft.rfft`, functions `stft`, `istft`, `ifgram`, and `fmt` are faster and have a reduced memory footprint. *Brian McFee*

.. _#772: https://github.com/librosa/librosa/issues/772
.. _#845: https://github.com/librosa/librosa/issues/845
.. _#907: https://github.com/librosa/librosa/issues/907
.. _#739: https://github.com/librosa/librosa/issues/739
.. _#894: https://github.com/librosa/librosa/issues/894
.. _#891: https://github.com/librosa/librosa/issues/891
.. _#863: https://github.com/librosa/librosa/issues/863
.. _#857: https://github.com/librosa/librosa/issues/857
.. _#843: https://github.com/librosa/librosa/issues/843
.. _#842: https://github.com/librosa/librosa/issues/842
.. _#840: https://github.com/librosa/librosa/issues/840
.. _#776: https://github.com/librosa/librosa/issues/776
.. _#770: https://github.com/librosa/librosa/issues/770
.. _#725: https://github.com/librosa/librosa/issues/725
.. _#709: https://github.com/librosa/librosa/issues/709
.. _#576: https://github.com/librosa/librosa/issues/576
.. _#519: https://github.com/librosa/librosa/issues/519
.. _#900: https://github.com/librosa/librosa/issues/900
.. _#888: https://github.com/librosa/librosa/issues/888
.. _#869: https://github.com/librosa/librosa/issues/869
.. _#851: https://github.com/librosa/librosa/issues/851
.. _#833: https://github.com/librosa/librosa/issues/833
.. _#825: https://github.com/librosa/librosa/issues/825
.. _#802: https://github.com/librosa/librosa/issues/802
.. _#884: https://github.com/librosa/librosa/issues/884
.. _#882: https://github.com/librosa/librosa/issues/882
.. _#874: https://github.com/librosa/librosa/issues/874
.. _#804: https://github.com/librosa/librosa/issues/804
.. _#849: https://github.com/librosa/librosa/issues/849
.. _#827: https://github.com/librosa/librosa/issues/827
.. _#813: https://github.com/librosa/librosa/issues/813
.. _#878: https://github.com/librosa/librosa/issues/878
.. _#857: https://github.com/librosa/librosa/issues/857
.. _#854: https://github.com/librosa/librosa/issues/854
.. _#846: https://github.com/librosa/librosa/issues/846
.. _#841: https://github.com/librosa/librosa/issues/841
.. _#834: https://github.com/librosa/librosa/issues/834
.. _#829: https://github.com/librosa/librosa/issues/829
.. _#818: https://github.com/librosa/librosa/issues/818
.. _#917: https://github.com/librosa/librosa/issues/917

v0.6.3
------
2019-02-13

Bug fixes
    - `#806`_ Fixed a bug in `estimate_tuning`. *@robrib, Monsij Biswal, Brian McFee*
    - `#799`_ Enhanced stability of elliptical filter implementation in `iirt`. *Frank Zalkow*

New features
    - `#766`_ made smoothing optional in `feature.chroma_cens`. *Kyungyun Lee*
    - `#760`_ allow explicit units for time axis decoration in `display`. *Kyungyun Lee*

Other changes
    - `#813`_ updated `core.load` documentation to cover bit depth truncation. *Andy Sarroff*
    - `#805`_ updated documentation for `core.localmax`. *Brian McFee*
    - `#801`_ renamed `feature.rmse` to `feature.rms`. *@nullmightybofo*
    - `#793`_ updated comments in `stft`. *Dan Ellis*
    - `#791`_ updated documentation for `write_wav`. *Brian McFee*
    - `#790`_ removed dependency on deprecated `imp` module. *Brian McFee* 
    - `#787`_ fixed typos in CONTRIBUTING documentation. *Vincent Lostanlen*
    - `#785`_ removed all run-time assertions in favor of proper exceptions. *Brian McFee*
    - `#783`_ migrated test infrastructure from `nose` to `pytest`. *Brian McFee*
    - `#777`_ include LICENSE file in source distribution. *toddrme2178*
    - `#769`_ updated documentation in `core.istft`. *Shayenne Moura*

.. _#813: https://github.com/librosa/librosa/issues/813
.. _#806: https://github.com/librosa/librosa/issues/806
.. _#805: https://github.com/librosa/librosa/issues/805
.. _#801: https://github.com/librosa/librosa/issues/801
.. _#799: https://github.com/librosa/librosa/issues/799
.. _#793: https://github.com/librosa/librosa/issues/793
.. _#791: https://github.com/librosa/librosa/issues/791
.. _#790: https://github.com/librosa/librosa/issues/790
.. _#787: https://github.com/librosa/librosa/issues/787
.. _#785: https://github.com/librosa/librosa/issues/785
.. _#783: https://github.com/librosa/librosa/issues/783
.. _#777: https://github.com/librosa/librosa/issues/777
.. _#769: https://github.com/librosa/librosa/issues/769
.. _#766: https://github.com/librosa/librosa/issues/766
.. _#760: https://github.com/librosa/librosa/issues/760

v0.6.2
------
2018-08-09

Bug fixes
    - `#730`_ Fixed cache support for ``joblib>=0.12``.  *Matt Vollrath*

New features
    - `#735`_ Added `core.times_like` and `core.samples_like` to generate time and sample indices
      corresponding to an existing feature matrix or shape specification. *Steve Tjoa*
    - `#750`_, `#753`_ Added `core.tone` and `core.chirp` signal generators. *Ziyao Wei*

Other changes
    - `#727`_ updated documentation for `core.get_duration`. *Zhen Wang*
    - `#731`_ fixed a typo in documentation for `core.fft_frequencies`. *Ziyao Wei*
    - `#734`_ expanded documentation for `feature.spectrall_rolloff`. *Ziyao Wei*
    - `#751`_ fixed example documentation for proper handling of phase in dB-scaling. *Vincent Lostanlen*
    - `#755`_ forward support and future-proofing for fancy indexing with ``numpy>1.15``. *Brian McFee*

.. _#730: https://github.com/librosa/librosa/pull/730
.. _#735: https://github.com/librosa/librosa/pull/735
.. _#750: https://github.com/librosa/librosa/pull/750
.. _#753: https://github.com/librosa/librosa/pull/753
.. _#727: https://github.com/librosa/librosa/pull/727
.. _#731: https://github.com/librosa/librosa/pull/731
.. _#734: https://github.com/librosa/librosa/pull/734
.. _#751: https://github.com/librosa/librosa/pull/751
.. _#755: https://github.com/librosa/librosa/pull/755

v0.6.1
------
2018-05-24

Bug fixes
  - `#677`_ `util.find_files` now correctly de-duplicates files on case-insensitive platforms. *Brian McFee*
  - `#713`_ `util.valid_intervals` now checks for non-negative durations. *Brian McFee, Dana Lee*
  - `#714`_ `util.match_intervals` can now explicitly fail when no matches are possible. *Brian McFee, Dana Lee*

New features
  - `#679`_, `#708`_ `core.pcen`, per-channel energy normalization. *Vincent Lostanlen, Brian McFee*
  - `#682`_ added different DCT modes to `feature.mfcc`. *Brian McFee*
  - `#687`_ `display` functions now accept target axes. *Pius Friesch*
  - `#688`_ numba-accelerated `util.match_events`. *Dana Lee*
  - `#710`_ `sequence` module and Viterbi decoding for generative, discriminative, and multi-label hidden Markov models. *Brian McFee*
  - `#714`_ `util.match_intervals` now supports tie-breaking for disjoint query intervals. *Brian McFee*

Other changes
  - `#677`_, `#705`_ added continuous integration testing for Windows. *Brian McFee*, *Ryuichi Yamamoto*
  - `#680`_ updated display module tests to support matplotlib 2.1. *Brian McFee*
  - `#684`_ corrected documentation for `core.stft` and `core.ifgram`. *Keunwoo Choi*
  - `#699`_, `#701`_ corrected documentation for `filters.semitone_filterbank` and `filters.mel_frequencies`. *Vincent Lostanlen*
  - `#704`_ eliminated unnecessary side-effects when importing `display`. *Brian McFee*
  - `#707`_ improved test coverage for dynamic time warping. *Brian McFee*
  - `#714`_ `util.match_intervals` matching logic has changed from raw intersection to Jaccard similarity.  *Brian McFee*


API Changes and compatibility
  - `#716`_ `core.dtw` has moved to `sequence.dtw`, and `core.fill_off_diagonal` has moved to
    `util.fill_off_diagonal`.  *Brian McFee*

.. _#716: https://github.com/librosa/librosa/pull/716
.. _#714: https://github.com/librosa/librosa/pull/714
.. _#713: https://github.com/librosa/librosa/pull/713
.. _#710: https://github.com/librosa/librosa/pull/710
.. _#708: https://github.com/librosa/librosa/pull/708
.. _#707: https://github.com/librosa/librosa/pull/707
.. _#705: https://github.com/librosa/librosa/pull/705
.. _#704: https://github.com/librosa/librosa/pull/704
.. _#701: https://github.com/librosa/librosa/pull/701
.. _#699: https://github.com/librosa/librosa/pull/699
.. _#688: https://github.com/librosa/librosa/pull/688
.. _#687: https://github.com/librosa/librosa/pull/687
.. _#684: https://github.com/librosa/librosa/pull/684
.. _#682: https://github.com/librosa/librosa/pull/682
.. _#680: https://github.com/librosa/librosa/pull/680
.. _#679: https://github.com/librosa/librosa/pull/679
.. _#677: https://github.com/librosa/librosa/pull/677

v0.6.0
------
2018-02-17

Bug fixes
  - `#663`_ fixed alignment errors in `feature.delta`. *Brian McFee*
  - `#646`_ `effects.trim` now correctly handles all-zeros signals. *Rimvydas Naktinis*
  - `#634`_ `stft` now conjugates the correct half of the spectrum. *Brian McFee*
  - `#630`_ fixed display decoration errors with `cqt_note` mode. *Brian McFee*
  - `#619`_ `effects.split` no longer returns out-of-bound sample indices. *Brian McFee*
  - `#616`_ Improved `util.valid_audio` to avoid integer type errors. *Brian McFee*
  - `#600`_ CQT basis functions are now correctly centered. *Brian McFee*
  - `#597`_ fixed frequency bin centering in `display.specshow`. *Brian McFee*
  - `#594`_ `dtw` fixed a bug which ignored weights when `step_sizes_sigma` did not match length. *Jackie Wu*
  - `#593`_ `stft` properly checks for valid input signals. *Erik Peterson*
  - `#587`_ `show_versions` now shows correct module names. *Ryuichi Yamamoto*

New features
  - `#648`_ `feature.spectral_flatness`. *Keunwoo Choi*
  - `#633`_ `feature.tempogram` now supports multi-band analysis. *Brian McFee*
  - `#439`_ `core.iirt` implements the multi-rate filterbank from Chroma Toolbox. *Stefan Balke*
  - `#435`_ `core.icqt` inverse constant-Q transform (unstable). *Brian McFee*

Other changes
  - `#674`_ Improved `write_wav` documentation with cross-references to `soundfile`. *Brian McFee*
  - `#671`_ Warn users when phase information is lost in dB conversion. *Carl Thome*
  - `#666`_ Expanded documentation for `load`'s resampling behavior. *Brian McFee*
  - `#656`_ Future-proofing numpy data type checks. *Carl Thome*
  - `#642`_ Updated unit tests for compatibility with matplotlib 2.1. *Brian McFee*
  - `#637`_ Improved documentation for advanced I/O. *Siddhartha Kumar*
  - `#636`_ `util.normalize` now preserves data type. *Brian McFee*
  - `#632`_ refined the validation requirements for `util.frame`. *Brian McFee*
  - `#628`_ all time/frequency conversion functions preserve input shape. *Brian McFee*
  - `#625`_ Numba is now a hard dependency. *Brian McFee*
  - `#622`_ `hz_to_midi` documentation corrections. *Carl Thome*
  - `#621`_ `dtw` is now symmetric with respect to input arguments. *Stefan Balke*
  - `#620`_ Updated requirements to prevent installation with (incompatible) sklearn 0.19.0. *Brian McFee*
  - `#609`_ Improved documentation for `segment.recurrence_matrix`. *Julia Wilkins*
  - `#598`_ Improved efficiency of `decompose.nn_filter`. *Brian McFee*
  - `#574`_ `dtw` now supports pre-computed distance matrices. *Curtis Hawthorne*

API changes and compatibility
  - `#627`_ The following functions and features have been removed:
      - `real=` parameter in `cqt`
      - `core.logamplitude` (replaced by `amplitude_to_db`)
      - `beat.estimate_tempo` (replaced by `beat.tempo`)
      - `n_fft=` parameter to `feature.rmse`
      - `ref_power=` parameter to `power_to_db`

  - The following features have been deprecated, and will be removed in 0.7.0:
      - `trim=` parameter to `feature.delta`

  - `#616`_ `write_wav` no longer supports integer-typed waveforms. This is due to enforcing
    consistency with `util.valid_audio` checks elsewhere in the codebase. If you have existing
    code that requires integer-valued output, consider using `soundfile.write` instead.

.. _#674: https://github.com/librosa/librosa/pull/674
.. _#671: https://github.com/librosa/librosa/pull/671
.. _#663: https://github.com/librosa/librosa/pull/663
.. _#646: https://github.com/librosa/librosa/pull/646
.. _#634: https://github.com/librosa/librosa/pull/634
.. _#630: https://github.com/librosa/librosa/pull/630
.. _#619: https://github.com/librosa/librosa/pull/619
.. _#616: https://github.com/librosa/librosa/pull/616
.. _#600: https://github.com/librosa/librosa/pull/600
.. _#597: https://github.com/librosa/librosa/pull/597
.. _#594: https://github.com/librosa/librosa/pull/594
.. _#593: https://github.com/librosa/librosa/pull/593
.. _#587: https://github.com/librosa/librosa/pull/587
.. _#648: https://github.com/librosa/librosa/pull/648
.. _#633: https://github.com/librosa/librosa/pull/633
.. _#439: https://github.com/librosa/librosa/pull/439
.. _#435: https://github.com/librosa/librosa/pull/435
.. _#666: https://github.com/librosa/librosa/pull/666
.. _#656: https://github.com/librosa/librosa/pull/656
.. _#642: https://github.com/librosa/librosa/pull/642
.. _#637: https://github.com/librosa/librosa/pull/637
.. _#636: https://github.com/librosa/librosa/pull/636
.. _#632: https://github.com/librosa/librosa/pull/632
.. _#628: https://github.com/librosa/librosa/pull/628
.. _#625: https://github.com/librosa/librosa/pull/625
.. _#622: https://github.com/librosa/librosa/pull/622
.. _#621: https://github.com/librosa/librosa/pull/621
.. _#620: https://github.com/librosa/librosa/pull/620
.. _#609: https://github.com/librosa/librosa/pull/609
.. _#598: https://github.com/librosa/librosa/pull/598
.. _#574: https://github.com/librosa/librosa/pull/574
.. _#627: https://github.com/librosa/librosa/pull/627

v0.5.1
------
2017-05-08

Bug fixes
  - `#555`_ added safety check for frequency bands in `spectral_contrast`. *Brian McFee*
  - `#554`_ fix interactive display for `tonnetz` visualization. *Brian McFee*
  - `#553`_ fix bug in `feature.spectral_bandwidth`. *Brian McFee*
  - `#539`_ fix `chroma_cens` to support scipy >=0.19. *Brian McFee*

New features
  - `#565`_ `feature.stack_memory` now supports negative delay. *Brian McFee*
  - `#563`_ expose padding mode in `stft/ifgram/cqt`. *Brian McFee*
  - `#559`_ explicit length option for `istft`. *Brian McFee*
  - `#557`_ added `show_versions`. *Brian McFee*
  - `#551`_ add `norm=` option to `filters.mel`. *Dan Ellis*

Other changes
  - `#569`_ `feature.rmse` now centers frames in the time-domain by default. *Brian McFee*
  - `#564`_ `display.specshow` now rasterizes images by default. *Brian McFee*
  - `#558`_ updated contributing documentation and issue templates. *Brian McFee*
  - `#556`_ updated tutorial for 0.5 API compatibility. *Brian McFee*
  - `#544`_ efficiency improvement in CQT. *Carl Thome*
  - `#523`_ support reading files with more than two channels. *Paul Brossier*

.. _#523: https://github.com/librosa/librosa/pull/523
.. _#544: https://github.com/librosa/librosa/pull/544
.. _#556: https://github.com/librosa/librosa/pull/556
.. _#558: https://github.com/librosa/librosa/pull/558
.. _#564: https://github.com/librosa/librosa/pull/564
.. _#551: https://github.com/librosa/librosa/pull/551
.. _#557: https://github.com/librosa/librosa/pull/557
.. _#559: https://github.com/librosa/librosa/pull/559
.. _#563: https://github.com/librosa/librosa/pull/563
.. _#565: https://github.com/librosa/librosa/pull/565
.. _#539: https://github.com/librosa/librosa/pull/539
.. _#553: https://github.com/librosa/librosa/pull/553
.. _#554: https://github.com/librosa/librosa/pull/554
.. _#555: https://github.com/librosa/librosa/pull/555
.. _#569: https://github.com/librosa/librosa/pull/569

v0.5.0
------
2017-02-17

Bug fixes
  - `#371`_ preserve integer hop lengths in constant-Q transforms. *Brian McFee*
  - `#386`_ fixed a length check in ``librosa.util.frame``. *Brian McFee*
  - `#416`_ ``librosa.output.write_wav`` only normalizes floating point, and normalization is disabled by
    default. *Brian McFee*
  - `#417`_ ``librosa.cqt`` output is now scaled continuously across octave boundaries. *Brian McFee, Eric
    Humphrey*
  - `#450`_ enhanced numerical stability for ``librosa.util.softmask``. *Brian McFee*
  - `#467`_ correction to chroma documentation. *Seth Kranzler*
  - `#501`_ fixed a numpy 1.12 compatibility error in ``pitch_tuning``. *Hojin Lee*

New features
  - `#323`_ ``librosa.dtw`` dynamic time warping. *Stefan Balke*
  - `#404`_ ``librosa.cache`` now supports priority levels, analogous to logging levels. *Brian McFee*
  - `#405`_ ``librosa.interp_harmonics`` for estimating harmonics of time-frequency representations. *Brian
    McFee*
  - `#410`_ ``librosa.beat.beat_track`` and ``librosa.onset.onset_detect`` can return output in frames,
    samples, or time units. *Brian McFee*
  - `#413`_ full support for scipy-style window specifications. *Brian McFee*
  - `#427`_ ``librosa.salience`` for computing spectrogram salience using harmonic peaks. *Rachel Bittner*
  - `#428`_ ``librosa.effects.trim`` and ``librosa.effects.split`` for trimming and splitting waveforms. *Brian
    McFee*
  - `#464`_ ``librosa.amplitude_to_db``, ``db_to_amplitude``, ``power_to_db``, and ``db_to_power`` for
    amplitude conversions.  This deprecates ``logamplitude``.  *Brian McFee*
  - `#471`_ ``librosa.util.normalize`` now supports ``threshold`` and ``fill_value`` arguments. *Brian McFee*
  - `#472`_ ``librosa.feature.melspectrogram`` now supports ``power`` argument. *Keunwoo Choi*
  - `#473`_ ``librosa.onset.onset_backtrack`` for backtracking onset events to previous local minima of
    energy. *Brian McFee*
  - `#479`_ ``librosa.beat.tempo`` replaces ``librosa.beat.estimate_tempo``, supports time-varying estimation.
    *Brian McFee*
  

Other changes
  - `#352`_ removed ``seaborn`` integration. *Brian McFee*
  - `#368`_ rewrite of the ``librosa.display`` submodule.  All plots are now in natural coordinates. *Brian
    McFee*
  - `#402`_ ``librosa.display`` submodule is not automatically imported. *Brian McFee*
  - `#403`_ ``librosa.decompose.hpss`` now returns soft masks. *Brian McFee*
  - `#407`_ ``librosa.feature.rmse`` can now compute directly in the time domain. *Carl Thome*
  - `#432`_ ``librosa.feature.rmse`` renames ``n_fft`` to ``frame_length``. *Brian McFee*
  - `#446`_ ``librosa.cqt`` now disables tuning estimation by default. *Brian McFee*
  - `#452`_ ``librosa.filters.__float_window`` now always uses integer length windows. *Brian McFee*
  - `#459`_ ``librosa.load`` now supports ``res_type`` argument for resampling. *CJ Carr*
  - `#482`_ ``librosa.filters.mel`` now warns if parameters will generate empty filter channels. *Brian McFee*
  - `#480`_ expanded documentation for advanced IO use-cases. *Fabian Robert-Stoeter*

API changes and compatibility
  - The following functions have permanently moved:
        - ``core.peak_peak`` to ``util.peak_pick``
        - ``core.localmax`` to ``util.localmax``
        - ``feature.sync`` to ``util.sync``

  - The following functions, classes, and constants have been removed:
        - ``core.ifptrack``
        - ``feature.chromagram``
        - ``feature.logfsgram``
        - ``filters.logfrequency``
        - ``output.frames_csv``
        - ``segment.structure_Feature``
        - ``display.time_ticks``
        - ``util.FeatureExtractor``
        - ``util.buf_to_int``
        - ``util.SMALL_FLOAT``

  - The following parameters have been removed:
        - ``librosa.cqt``: `resolution`
        - ``librosa.cqt``: `aggregate`
        - ``feature.chroma_cqt``: `mode`
        - ``onset_strength``: `centering`

  - Seaborn integration has been removed, and the ``display`` submodule now requires matplotlib >= 1.5.
        - The `use_sns` argument has been removed from `display.cmap`
        - `magma` is now the default sequential colormap.

  - The ``librosa.display`` module has been rewritten.
        - ``librosa.display.specshow`` now plots using `pcolormesh`, and supports non-uniform time and frequency axes.
        - All plots can be rendered in natural coordinates (e.g., time or Hz)
        - Interactive plotting is now supported via ticker and formatter objects

  - ``librosa.decompose.hpss`` with `mask=True` now returns soft masks, rather than binary masks.

  - ``librosa.filters.get_window`` wraps ``scipy.signal.get_window``, and handles generic callables as well pre-registered
    window functions.  All windowed analyses (e.g., ``stft``, ``cqt``, or ``tempogram``) now support the full range
    of window functions and parameteric windows via tuple parameters, e.g., ``window=('kaiser', 4.0)``.
        
  - ``stft`` windows are now explicitly asymmetric by default, which breaks backwards compatibility with the 0.4 series.

  - ``cqt`` now returns properly scaled outputs that are continuous across octave boundaries.  This breaks
    backwards compatibility with the 0.4 series.

  - ``cqt`` now uses `tuning=0.0` by default, rather than estimating the tuning from the signal.  Tuning
    estimation is still supported, and enabled by default for chroma analysis (``librosa.feature.chroma_cqt``).

  - ``logamplitude`` is deprecated in favor of ``amplitude_to_db`` or ``power_to_db``.  The `ref_power` parameter
    has been renamed to `ref`.


.. _#501: https://github.com/librosa/librosa/pull/501
.. _#480: https://github.com/librosa/librosa/pull/480
.. _#467: https://github.com/librosa/librosa/pull/467
.. _#450: https://github.com/librosa/librosa/pull/450
.. _#417: https://github.com/librosa/librosa/pull/417
.. _#416: https://github.com/librosa/librosa/pull/416
.. _#386: https://github.com/librosa/librosa/pull/386
.. _#371: https://github.com/librosa/librosa/pull/371
.. _#479: https://github.com/librosa/librosa/pull/479
.. _#473: https://github.com/librosa/librosa/pull/473
.. _#472: https://github.com/librosa/librosa/pull/472
.. _#471: https://github.com/librosa/librosa/pull/471
.. _#464: https://github.com/librosa/librosa/pull/464
.. _#428: https://github.com/librosa/librosa/pull/428
.. _#427: https://github.com/librosa/librosa/pull/427
.. _#413: https://github.com/librosa/librosa/pull/413
.. _#410: https://github.com/librosa/librosa/pull/410
.. _#405: https://github.com/librosa/librosa/pull/405
.. _#404: https://github.com/librosa/librosa/pull/404
.. _#323: https://github.com/librosa/librosa/pull/323
.. _#482: https://github.com/librosa/librosa/pull/482
.. _#459: https://github.com/librosa/librosa/pull/459
.. _#452: https://github.com/librosa/librosa/pull/452
.. _#446: https://github.com/librosa/librosa/pull/446
.. _#432: https://github.com/librosa/librosa/pull/432
.. _#407: https://github.com/librosa/librosa/pull/407
.. _#403: https://github.com/librosa/librosa/pull/403
.. _#402: https://github.com/librosa/librosa/pull/402
.. _#368: https://github.com/librosa/librosa/pull/368
.. _#352: https://github.com/librosa/librosa/pull/352



v0.4.3
------
2016-05-17

Bug fixes
  - `#315`_ fixed a positioning error in ``display.specshow`` with logarithmic axes. *Brian McFee*
  - `#332`_ ``librosa.cqt`` now throws an exception if the signal is too short for analysis. *Brian McFee*
  - `#341`_ ``librosa.hybrid_cqt`` properly matches the scale of ``librosa.cqt``. *Brian McFee*
  - `#348`_ ``librosa.cqt`` fixed a bug introduced in v0.4.2. *Brian McFee*
  - `#354`_ Fixed a minor off-by-one error in ``librosa.beat.estimate_tempo``. *Brian McFee*
  - `#357`_ improved numerical stability of ``librosa.decompose.hpss``. *Brian McFee*

New features
  - `#312`_ ``librosa.segment.recurrence_matrix`` can now construct sparse self-similarity matrices. *Brian
    McFee*
  - `#337`_ ``librosa.segment.recurrence_matrix`` can now produce weighted affinities and distances. *Brian
    McFee*
  - `#311`_ ``librosa.decompose.nl_filter`` implements several self-similarity based filtering operations
    including non-local means. *Brian McFee*
  - `#320`_ ``librosa.feature.chroma_cens`` implements chroma energy normalized statistics (CENS) features.
    *Stefan Balke*
  - `#354`_ ``librosa.core.tempo_frequencies`` computes tempo (BPM) frequencies for autocorrelation and
    tempogram features. *Brian McFee*
  - `#355`_ ``librosa.decompose.hpss`` now supports harmonic-percussive-residual separation. *CJ Carr, Brian McFee*
  - `#357`_ ``librosa.util.softmask`` computes numerically stable soft masks. *Brian McFee*

Other changes
  - ``librosa.cqt``, ``librosa.hybrid_cqt`` parameter `aggregate` is now deprecated.
  - Resampling is now handled by the ``resampy`` library
  - ``librosa.get_duration`` can now operate directly on filenames as well as audio buffers and feature
    matrices.
  - ``librosa.decompose.hpss`` no longer supports ``power=0``.

.. _#315: https://github.com/librosa/librosa/pull/315
.. _#332: https://github.com/librosa/librosa/pull/332
.. _#341: https://github.com/librosa/librosa/pull/341
.. _#348: https://github.com/librosa/librosa/pull/348
.. _#312: https://github.com/librosa/librosa/pull/312
.. _#337: https://github.com/librosa/librosa/pull/337
.. _#311: https://github.com/librosa/librosa/pull/311
.. _#320: https://github.com/librosa/librosa/pull/320
.. _#354: https://github.com/librosa/librosa/pull/354
.. _#355: https://github.com/librosa/librosa/pull/355
.. _#357: https://github.com/librosa/librosa/pull/357

v0.4.2
------
2016-02-20

Bug fixes
  - Support for matplotlib 1.5 color properties in the ``display`` module
  - `#308`_ Fixed a per-octave scaling error in ``librosa.cqt``. *Brian McFee*

New features
  - `#279`_ ``librosa.cqt`` now provides complex-valued output with argument `real=False`.
    This will become the default behavior in subsequent releases.
  - `#288`_ ``core.resample`` now supports multi-channel inputs. *Brian McFee*
  - `#295`_ ``librosa.display.frequency_ticks``: like ``time_ticks``. Ticks can now dynamically
    adapt to scale (mHz, Hz, KHz, MHz, GHz) and use automatic precision formatting (``%g``). *Brian McFee*


Other changes
  - `#277`_ improved documentation for OSX. *Stefan Balke*
  - `#294`_ deprecated the ``FeatureExtractor`` object. *Brian McFee*
  - `#300`_ added dependency version requirements to install script. *Brian McFee*
  - `#302`_, `#279`_ renamed the following parameters
      - ``librosa.display.time_ticks``: `fmt` is now `time_fmt`
      - ``librosa.feature.chroma_cqt``: `mode` is now `cqt_mode`
      - ``librosa.cqt``, ``hybrid_cqt``, ``pseudo_cqt``, ``librosa.filters.constant_q``: `resolution` is now `filter_scale`
  - `#308`_ ``librosa.cqt`` default `filter_scale` parameter is now 1 instead of 2.

.. _#277: https://github.com/librosa/librosa/pull/277
.. _#279: https://github.com/librosa/librosa/pull/279
.. _#288: https://github.com/librosa/librosa/pull/288
.. _#294: https://github.com/librosa/librosa/pull/294
.. _#295: https://github.com/librosa/librosa/pull/295
.. _#300: https://github.com/librosa/librosa/pull/300
.. _#302: https://github.com/librosa/librosa/pull/302
.. _#308: https://github.com/librosa/librosa/pull/308

v0.4.1
------
2015-10-17

Bug fixes
  - Improved safety check in CQT for invalid hop lengths
  - Fixed division by zero bug in ``core.pitch.pip_track``
  - Fixed integer-type error in ``util.pad_center`` on numpy v1.10
  - Fixed a context scoping error in ``librosa.load`` with some audioread backends
  - ``librosa.autocorrelate`` now persists type for complex input

New features
  - ``librosa.clicks`` sonifies timed events such as beats or onsets
  - ``librosa.onset.onset_strength_multi`` computes onset strength within multiple sub-bands
  - ``librosa.feature.tempogram`` computes localized onset strength autocorrelation
  - ``librosa.display.specshow`` now supports ``*_axis='tempo'`` for annotating tempo-scaled data
  - ``librosa.fmt`` implements the Fast Mellin Transform

Other changes
  - Rewrote ``display.waveplot`` for improved efficiency
  - ``decompose.deompose()`` now supports pre-trained transformation objects
  - Nullified side-effects of optional seaborn dependency
  - Moved ``feature.sync`` to ``util.sync`` and expanded its functionality
  - ``librosa.onset.onset_strength`` and ``onset_strength_multi`` support superflux-style lag and max-filtering
  - ``librosa.core.autocorrelate`` can now operate along any axis of multi-dimensional input
  - the ``segment`` module functions now support arbitrary target axis
  - Added proper window normalization to ``librosa.core.istft`` for better reconstruction 
    (`PR #235 <https://github.com/librosa/librosa/pull/235>`_).
  - Standardized ``n_fft=2048`` for ``piptrack``, ``ifptrack`` (deprecated), and
    ``logfsgram`` (deprecated)
  - ``onset_strength`` parameter ``'centering'`` has been deprecated and renamed to
    ``'center'``
  - ``onset_strength`` always trims to match the input spectrogram duration
  - added tests for ``piptrack``
  - added test support for Python 3.5




v0.4.0
------
2015-07-08

Bug fixes
   -  Fixed alignment errors with ``offset`` and ``duration`` in ``load()``
   -  Fixed an edge-padding issue with ``decompose.hpss()`` which resulted in percussive noise leaking into the harmonic component.
   -  Fixed stability issues with ``ifgram()``, added options to suppress negative frequencies.
   -  Fixed scaling and padding errors in ``feature.delta()``
   -  Fixed some errors in ``note_to_hz()`` string parsing
   -  Added robust range detection for ``display.cmap``
   -  Fixed tick placement in ``display.specshow``
   -  Fixed a low-frequency filter alignment error in ``cqt``
   -  Added aliasing checks for ``cqt`` filterbanks
   -  Fixed corner cases in ``peak_pick``
   -  Fixed bugs in ``find_files()`` with negative slicing
   -  Fixed tuning estimation errors
   -  Fixed octave numbering in to conform to scientific pitch notation

New features
   -  python 3 compatibility
   -  Deprecation and moved-function warnings
   -  added ``norm=None`` option to ``util.normalize()``
   -  ``segment.recurrence_to_lag``, ``lag_to_recurrence``
   -  ``core.hybrid_cqt()`` and ``core.pseudo_cqt()``
   -  ``segment.timelag_filter``
   -  Efficiency enhancements for ``cqt``
   -  Major rewrite and reformatting of documentation
   -  Improvements to ``display.specshow``:
      -  added the ``lag`` axis format
      -  added the ``tonnetz`` axis format
      -  allow any combination of axis formats
   -  ``effects.remix()``
   -  Added new time and frequency converters:
      -  ``note_to_hz()``, ``hz_to_note()``
      -  ``frames_to_samples()``, ``samples_to_frames()``
      -  ``time_to_samples()``, ``samples_to_time()``
   -  ``core.zero_crossings``
   -  ``util.match_events()``
   -  ``segment.subsegment()`` for segmentation refinement
   -  Functional examples in almost all docstrings
   -  improved numerical stability in ``normalize()``
   -  audio validation checks
   -  ``to_mono()``
   -  ``librosa.cache`` for storing pre-computed features
   -  Stereo output support in ``write_wav``
   -  Added new feature extraction functions:
      -  ``feature.spectral_contrast``
      -  ``feature.spectral_bandwidth``
      -  ``feature.spectral_centroid``
      -  ``feature.spectral_rolloff``
      -  ``feature.poly_features``
      -  ``feature.rmse``
      -  ``feature.zero_crossing_rate``
      -  ``feature.tonnetz``
   - Added ``display.waveplot``

Other changes
   -  Internal refactoring and restructuring of submodules
   -  Removed the ``chord`` module
   -  input validation and better exception reporting for most functions
   -  Changed the default colormaps in ``display``
   -  Changed default parameters in onset detection, beat tracking
   -  Changed default parameters in ``cqt``
   -  ``filters.constant_q`` now returns filter lengths
   -  Chroma now starts at ``C`` by default, instead of ``A``
   -  ``pad_center`` supports multi-dimensional input and ``axis`` parameter
   - switched from ``np.fft`` to ``scipy.fftpack`` for FFT operations
   - changed all librosa-generated exception to a new class librosa.ParameterError

Deprecated functions
   -  ``util.buf_to_int``
   -  ``output.frames_csv``
   -  ``segment.structure_feature``
   -  ``filters.logfrequency``
   -  ``feature.logfsgram``

v0.3.1
------
2015-02-18

Bug fixes
   -  Fixed bug #117: ``librosa.segment.agglomerative`` now returns a numpy.ndarray instead of a list
   -  Fixed bug #115: off-by-one error in ``librosa.core.load`` with fixed duration
   -  Fixed numerical underflow errors in ``librosa.decompose.hpss``
   -  Fixed bug #104: ``librosa.decompose.hpss`` failed with silent, complex-valued input
   -  Fixed bug #103: ``librosa.feature.estimate_tuning`` fails when no bins exceed the threshold

Features
   -  New function ``librosa.core.get_duration()`` computes the duration of an audio signal or spectrogram-like input matrix
   -  ``librosa.util.pad_center`` now accepts multi-dimensional input

Other changes
   -  Adopted the ISC license
   -  Python 3 compatibility via futurize
   -  Fixed issue #102: segment.agglomerative no longer depends on the deprecated Ward module of sklearn; it now depends on the newer Agglomerative module.
   -  Issue #108: set character encoding on all source files
   -  Added dtype persistence for resample, stft, istft, and effects functions

v0.3.0
------
2014-06-30

Bug fixes
   -  Fixed numpy array indices to force integer values
   -  ``librosa.util.frame`` now warns if the input data is non-contiguous
   -  Fixed a formatting error in ``librosa.display.time_ticks()``
   -  Added a warning if ``scikits.samplerate`` is not detected

Features
   -  New module ``librosa.chord`` for training chord recognition models
   -  Parabolic interpolation piptracking ``librosa.feature.piptrack()``
   -  ``librosa.localmax()`` now supports multi-dimensional slicing
   -  New example scripts
   -  Improved documentation
   -  Added the ``librosa.util.FeatureExtractor`` class, which allows librosa functions to act as feature extraction stages in ``sklearn``
   -  New module ``librosa.effects`` for time-domain audio processing
   -  Added demo notebooks for the ``librosa.effects`` and ``librosa.util.FeatureExtractor``
   -  Added a full-track audio example, ``librosa.util.example_audio_file()``
   -  Added peak-frequency sorting of basis elements in ``librosa.decompose.decompose()``

Other changes
   -  Spectrogram frames are now centered, rather than left-aligned. This removes the need for window correction in ``librosa.frames_to_time()``
   -  Accelerated constant-Q transform ``librosa.cqt()``
   -  PEP8 compliance
   -  Removed normalization from ``librosa.feature.logfsgram()``
   -  Efficiency improvements by ensuring memory contiguity
   -  ``librosa.logamplitude()`` now supports functional reference power, in addition to scalar values
   -  Improved ``librosa.feature.delta()``
   -  Additional padding options to ``librosa.feature.stack_memory()``
   -  ``librosa.cqt`` and ``librosa.feature.logfsgram`` now use the same parameter formats ``(fmin, n_bins, bins_per_octave)``.
   -  Updated demo notebook(s) to IPython 2.0
   -  Moved ``perceptual_weighting()`` from ``librosa.feature`` into ``librosa.core``
   -  Moved ``stack_memory()`` from ``librosa.segment`` into ``librosa.feature``
   -  Standardized ``librosa.output.annotation`` input format to match ``mir_eval``
   -  Standardized variable names (e.g., ``onset_envelope``).

v0.2.1
------
2014-01-21

Bug fixes
   -  fixed an off-by-one error in ``librosa.onset.onset_strength()``
   -  fixed a sign-flip error in ``librosa.output.write_wav()``
   -  removed all mutable object default parameters

Features
   -  added option ``centering`` to ``librosa.onset.onset_strength()`` to resolve frame-centering issues with sliding window STFT
   -  added frame-center correction to ``librosa.core.frames_to_time()`` and ``librosa.core.time_to_frames()``
   -  added ``librosa.util.pad_center()``
   -  added ``librosa.output.annotation()``
   -  added ``librosa.output.times_csv()``
   -  accelerated ``librosa.core.stft()`` and ``ifgram()``
   -  added ``librosa.util.frame`` for in-place signal framing
   -  ``librosa.beat.beat_track`` now supports user-supplied tempo
   -  added ``librosa.util.normalize()``
   -  added ``librosa.util.find_files()``
   -  added ``librosa.util.axis_sort()``
   -  new module: ``librosa.util()``
   -  ``librosa.filters.constant_q`` now support padding
   -  added boolean input support for ``librosa.display.cmap()``
   -  speedup in ``librosa.core.cqt()``

Other changes
   -  optimized default parameters for ``librosa.onset.onset_detect``
   -  set ``librosa.filters.mel`` parameter ``n_mels=128`` by default
   -  ``librosa.feature.chromagram()`` and ``logfsgram()`` now use power instead of energy
   -  ``librosa.display.specshow()`` with ``y_axis='chroma'`` now labels as ``pitch class``
   -  set ``librosa.core.cqt`` parameter ``resolution=2`` by default
   -  set ``librosa.feature.chromagram`` parameter ``octwidth=2`` by default

v0.2.0
------
2013-12-14

Bug fixes
   -  fixed default ``librosa.core.stft, istft, ifgram`` to match specification
   -  fixed a float->int bug in peak\_pick
   -  better memory efficiency
   -  ``librosa.segment.recurrence_matrix`` corrects for width suppression
   -  fixed a divide-by-0 error in the beat tracker
   -  fixed a bug in tempo estimation with short windows
   -  ``librosa.feature.sync`` now supports 1d arrays
   -  fixed a bug in beat trimming
   -  fixed a bug in ``librosa.core.stft`` when calculating window size
   -  fixed ``librosa.core.resample`` to support stereo signals

Features
   -  added filters option to cqt
   -  added window function support to istft
   -  added an IPython notebook demo
   -  added ``librosa.features.delta`` for computing temporal difference features
   -  new ``examples`` scripts: tuning, hpss
   -  added optional trimming to ``librosa.segment.stack_memory``
   -  ``librosa.onset.onset_strength`` now takes generic spectrogram function ``feature``
   -  compute reference power directly in ``librosa.core.logamplitude``
   -  color-blind-friendly default color maps in ``librosa.display.cmap``
   -  ``librosa.core.onset_strength`` now accepts an aggregator
   -  added ``librosa.feature.perceptual_weighting``
   -  added tuning estimation to ``librosa.feature.chromagram``
   -  added ``librosa.core.A_weighting``
   -  vectorized frequency converters
   -  added ``librosa.core.cqt_frequencies`` to get CQT frequencies
   -  ``librosa.core.cqt`` basic constant-Q transform implementation
   -  ``librosa.filters.cq_to_chroma`` to convert log-frequency to chroma
   -  added ``librosa.core.fft_frequencies``
   -  ``librosa.decompose.hpss`` can now return masking matrices
   -  added reversal for ``librosa.segment.structure_feature``
   -  added ``librosa.core.time_to_frames``
   -  added cent notation to ``librosa.core.midi_to_note``
   -  added time-series or spectrogram input options to ``chromagram``, ``logfsgram``, ``melspectrogram``, and ``mfcc``
   -  new module: ``librosa.display``
   -  ``librosa.output.segment_csv`` => ``librosa.output.frames_csv``
   -  migrated frequency converters to ``librosa.core``
   -  new module: ``librosa.filters``
   -  ``librosa.decompose.hpss`` now supports complex-valued STFT matrices
   -  ``librosa.decompose.decompose()`` supports ``sklearn`` decomposition objects
   -  added ``librosa.core.phase_vocoder``
   -  new module: ``librosa.onset``; migrated onset strength from ``librosa.beat``
   -  added ``librosa.core.pick_peaks``
   -  ``librosa.core.load()`` supports offset and duration parameters
   -  ``librosa.core.magphase()`` to separate magnitude and phase from a complex matrix
   -  new module: ``librosa.segment``

Other changes
   -  ``onset_estimate_bpm => estimate_tempo``
   -  removed ``n_fft`` from ``librosa.core.istft()``
   -  ``librosa.core.mel_frequencies`` returns ``n_mels`` values by default
   -  changed default ``librosa.decompose.hpss`` window to 31
   -  disabled onset de-trending by default in ``librosa.onset.onset_strength``
   -  added complex-value warning to ``librosa.display.specshow``
   -  broke compatibilty with ``ifgram.m``; ``librosa.core.ifgram`` now matches ``stft``
   -  changed default beat tracker settings
   -  migrated ``hpss`` into ``librosa.decompose``
   -  changed default ``librosa.decompose.hpss`` power parameter to ``2.0``
   -  ``librosa.core.load()`` now returns single-precision by default
   -  standardized ``n_fft=2048``, ``hop_length=512`` for most functions
   -  refactored tempo estimator

v0.1.0
------

Initial public release.
