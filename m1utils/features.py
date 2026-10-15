"""Extraction de caractéristiques acoustiques.

Les paramètres par défaut de ``log_mel`` sont ceux de Whisper : fenêtre de 25 ms (n_fft=400),
pas de 10 ms (hop=160), 80 bandes Mel → 100 trames par seconde.
"""
from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE


def stft_magnitude(y: np.ndarray, n_fft: int = 512, hop_length: int = 160, window: str = "hann") -> np.ndarray:
    """|STFT| de forme (1 + n_fft/2, trames)."""
    import librosa

    return np.abs(librosa.stft(np.asarray(y, dtype=np.float32), n_fft=n_fft, hop_length=hop_length, window=window))


def log_mel(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160) -> np.ndarray:
    """Log-Mel (dB) de forme (n_mels, trames). Défauts = paramètres de Whisper (25 ms / 10 ms)."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = 13, deltas: bool = False) -> np.ndarray:
    """MFCC (n_mfcc, trames) ; avec ``deltas=True`` : concatène Δ et ΔΔ → (3·n_mfcc, trames)."""
    import librosa

    m = librosa.feature.mfcc(y=np.asarray(y, dtype=np.float32), sr=sr, n_mfcc=n_mfcc, n_fft=400, hop_length=160)
    if deltas:
        m = np.concatenate([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)], axis=0)
    return m.astype(np.float32)


def f0_contour(y: np.ndarray, sr: int = SAMPLE_RATE, fmin: float = 75, fmax: float = 400,
               time_step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """F0 via Parselmouth (algorithme de Praat). Retourne (temps_s, f0_hz) avec NaN sur les trames non voisées.

    Conseil : ``fmin=75, fmax=300`` pour une voix d'homme, ``fmin=100, fmax=500`` pour une voix de femme/enfant.
    """
    import parselmouth

    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    return pitch.xs(), f0


def f0_semitones(f0: np.ndarray, ref_hz: float | None = None) -> np.ndarray:
    """Convertit une F0 (Hz) en demi-tons relatifs à ``ref_hz`` (défaut : médiane du locuteur).

    Normalisation indispensable pour comparer des locuteurs (voix graves/aiguës) : 12 demi-tons = une octave.
    """
    ref = np.nanmedian(f0) if ref_hz is None else ref_hz
    return 12 * np.log2(f0 / ref)


def formants(y: np.ndarray, sr: int = SAMPLE_RATE, at_s: float | None = None, max_formant: float = 5500) -> dict:
    """F1, F2, F3 (Hz) au milieu du signal (ou à ``at_s``), via Parselmouth (Burg).

    ``max_formant`` : 5000 Hz pour une voix d'homme, 5500 Hz pour une voix de femme.
    """
    import parselmouth

    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)
    form = snd.to_formant_burg(maximum_formant=max_formant)
    t = snd.duration / 2 if at_s is None else at_s
    return {f"F{i}": float(form.get_value_at_time(i, t)) for i in (1, 2, 3)}


def frame_rate(model_family: str) -> float:
    """Nombre de trames par seconde produites en entrée/sortie par une famille de modèles."""
    return {"whisper": 100.0, "wav2vec2": 50.0, "hubert": 50.0, "mfcc": 100.0}[model_family]
