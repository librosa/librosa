def extract_chroma(filename, disp=1, rs=11025, tune_fft=4096, tune_ctr=400.0, tune_sd=1.0, 
                   HPSS_len=1024, HPSS_hop=512, HPSS_win=4096, HPSS_alph=0.3, HPSS_gamma=0.3, 
                   HPSS_kmax=1, wavewrite=0, beat_startbpm=120, beat_tight=3, 
                   fmin=220.0, fmax=1661.0, res_fact=5.0, LBC_ref='s',LBC_Aweight=1,
                   LBC_q=0.0, LBC_norm='n'):

  # Main process for extracting a chromagram. 
  # Use keyword arguments in the three main functions.
  
  # 1. HPSS, Tuning and beat estimate
  import HPA_feature_extraction as HPA_feat
  reload(HPA_feat)
  [xharmony, semisoff, resample_rate, beat_times] = HPA_feat.extract_harmony_from_audio(filename,
                     display=disp, resample_rate=rs, tuning_fftlen=tune_fft, f_ctr=tune_ctr,
                     f_sd=tune_sd, HPSS_fftlen=HPSS_len, HPSS_hoplen=HPSS_hop, 
                     HPSS_window=HPSS_win, gamma=HPSS_gamma, alpha=HPSS_gamma, kmax=HPSS_kmax,
                     write_wav=wavewrite, startbpm=beat_startbpm, beat_tightness=beat_tight)
  
  # 2. Calculate hamming windows
  [hamming_k, half_winLenK, freqBins] = HPA_feat.cal_hamming_window(resample_rate,
                     minFreq=fmin, maxFreq=fmax, resolution_fact=res_fact,tuning=semisoff)
  
  # 3. Extract chroma
  [chromagram, normal_chromagram, sample_times] = HPA_feat.cal_CQ_chroma_loudness(xharmony, 
                     resample_rate, beat_times, hamming_k, half_winLenK, freqBins, 
                     refLabel=LBC_ref, A_weightLabel=LBC_ref, q_value=LBC_ref, normFlag=LBC_norm)
  
  return chromagram, normal_chromagram, sample_times, semisoff
  