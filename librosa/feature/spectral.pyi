from _typeshed import Incomplete

def spectral_centroid(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., n_fft: int = ..., hop_length: int = ..., freq: Incomplete | None = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ...): ...
def spectral_bandwidth(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ..., freq: Incomplete | None = ..., centroid: Incomplete | None = ..., norm: bool = ..., p: int = ...): ...
def spectral_contrast(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ..., freq: Incomplete | None = ..., fmin: float = ..., n_bands: int = ..., quantile: float = ..., linear: bool = ...): ...
def spectral_rolloff(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ..., freq: Incomplete | None = ..., roll_percent: float = ...): ...
def spectral_flatness(*, y: Incomplete | None = ..., S: Incomplete | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ..., amin: float = ..., power: float = ...): ...
def rms(*, y: Incomplete | None = ..., S: Incomplete | None = ..., frame_length: int = ..., hop_length: int = ..., center: bool = ..., pad_mode: str = ...): ...
def poly_features(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ..., order: int = ..., freq: Incomplete | None = ...): ...
def zero_crossing_rate(y, *, frame_length: int = ..., hop_length: int = ..., center: bool = ..., **kwargs): ...
def chroma_stft(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., norm=..., n_fft: int = ..., hop_length: int = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ..., tuning: Incomplete | None = ..., n_chroma: int = ..., **kwargs): ...
def chroma_cqt(*, y: Incomplete | None = ..., sr: int = ..., C: Incomplete | None = ..., hop_length: int = ..., fmin: Incomplete | None = ..., norm=..., threshold: float = ..., tuning: Incomplete | None = ..., n_chroma: int = ..., n_octaves: int = ..., window: Incomplete | None = ..., bins_per_octave: int = ..., cqt_mode: str = ...): ...
def chroma_cens(*, y: Incomplete | None = ..., sr: int = ..., C: Incomplete | None = ..., hop_length: int = ..., fmin: Incomplete | None = ..., tuning: Incomplete | None = ..., n_chroma: int = ..., n_octaves: int = ..., bins_per_octave: int = ..., cqt_mode: str = ..., window: Incomplete | None = ..., norm: int = ..., win_len_smooth: int = ..., smoothing_window: str = ...): ...
def tonnetz(*, y: Incomplete | None = ..., sr: int = ..., chroma: Incomplete | None = ..., **kwargs): ...
def mfcc(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., n_mfcc: int = ..., dct_type: int = ..., norm: str = ..., lifter: int = ..., **kwargs): ...
def melspectrogram(*, y: Incomplete | None = ..., sr: int = ..., S: Incomplete | None = ..., n_fft: int = ..., hop_length: int = ..., win_length: Incomplete | None = ..., window: str = ..., center: bool = ..., pad_mode: str = ..., power: float = ..., **kwargs): ...
