"""Prétraitement des enregistrements.

Pipeline de référence du module (atelier S2-NB02, TP2) :
    load → mono → resample 16 kHz → retrait DC → normalisation → trim → [débruitage] → [VAD/segmentation]

⚠️ Après la coupe des silences, on **remet une marge de 0,2 s** : mesure du cours (Simba-H, fongbe lu), couper au ras
de la parole fait passer le WER de 0,27 à 0,63 (le modèle a appris avec des silences autour des énoncés) ; avec la
marge, 0,285.

⚠️ Le débruitage est **désactivé par défaut** : il améliore le confort d'écoute mais introduit des artefacts
qui peuvent dégrader la reconnaissance. On le mesure avant de l'adopter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import SAMPLE_RATE

_EPS = 1e-10


@dataclass
class PreprocessConfig:
    target_sr: int = SAMPLE_RATE
    normalization: str = "peak"          # "peak" | "rms" | "lufs" | "none"
    peak_dbfs: float = -1.0
    rms_dbfs: float = -20.0
    lufs: float = -23.0
    trim_top_db: float = 30.0
    pad_s: float = 0.2                   # marge de silence remise après la coupe (cf. mesure ci-dessous)
    denoise: bool = False
    vad: str | None = None               # None | "energy" | "silero"
    min_segment_s: float = 0.5
    max_segment_s: float = 15.0
    extra: dict = field(default_factory=dict)


@dataclass
class PreprocessReport:
    duration_in_s: float
    duration_out_s: float
    snr_before_db: float | None
    snr_after_db: float | None
    clipping_ratio: float
    segments: list[tuple[float, float]]


def dbfs(y: np.ndarray, kind: str = "peak") -> float:
    """Niveau en dBFS (0 dBFS = pleine échelle) : ``kind`` = "peak" ou "rms"."""
    y = np.asarray(y, dtype=np.float64)
    level = np.max(np.abs(y)) if kind == "peak" else np.sqrt(np.mean(y**2))
    return float(20 * np.log10(level + _EPS))


def remove_dc(y: np.ndarray) -> np.ndarray:
    """Soustrait la composante continue (moyenne)."""
    return (np.asarray(y, dtype=np.float32) - np.mean(y)).astype(np.float32)


def normalize_peak(y: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    """Met le pic absolu à ``peak_dbfs`` dBFS."""
    y = np.asarray(y, dtype=np.float32)
    peak = np.max(np.abs(y))
    if peak < _EPS:
        return y
    return (y * (10 ** (peak_dbfs / 20) / peak)).astype(np.float32)


def normalize_rms(y: np.ndarray, rms_dbfs: float = -20.0) -> np.ndarray:
    """Met le RMS à ``rms_dbfs`` dBFS. Si le gain fait saturer, il est réduit pour garder un pic à -1 dBFS."""
    y = np.asarray(y, dtype=np.float32)
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    if rms < _EPS:
        return y
    gain = 10 ** (rms_dbfs / 20) / rms
    peak_after = np.max(np.abs(y)) * gain
    if peak_after > 10 ** (-1 / 20):
        gain *= 10 ** (-1 / 20) / peak_after
    return (y * gain).astype(np.float32)


def normalize_loudness(y: np.ndarray, sr: int = SAMPLE_RATE, lufs: float = -23.0) -> np.ndarray:
    """Normalisation de sonie ITU-R BS.1770 (pyloudnorm). Repli RMS pour les signaux < 0,4 s."""
    import warnings

    if len(y) < 0.4 * sr:
        warnings.warn("Signal trop court pour une mesure LUFS (< 0,4 s) : repli sur la normalisation RMS.")
        return normalize_rms(y, rms_dbfs=lufs + 3)
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(np.asarray(y, dtype=np.float64))
    if not np.isfinite(loud):
        return np.asarray(y, dtype=np.float32)
    out = pyln.normalize.loudness(np.asarray(y, dtype=np.float64), loud, lufs)
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak * 10 ** (-1 / 20)
    return out.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> tuple[np.ndarray, tuple[int, int]]:
    """Retire les silences de début/fin. Retourne (signal, (début, fin) en échantillons)."""
    import librosa

    yt, idx = librosa.effects.trim(np.asarray(y, dtype=np.float32), top_db=top_db)
    return yt.astype(np.float32), (int(idx[0]), int(idx[1]))


def detect_clipping(y: np.ndarray, threshold: float = 0.999) -> float:
    """Proportion d'échantillons saturés (|y| ≥ threshold)."""
    y = np.asarray(y)
    return float(np.mean(np.abs(y) >= threshold)) if len(y) else 0.0


def frame_energy_db(y: np.ndarray, sr: int = SAMPLE_RATE, frame_s: float = 0.03, hop_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Énergie RMS par trame en dB. Retourne (centres des trames en s, énergie en dB)."""
    frame, hop = int(frame_s * sr), int(hop_s * sr)
    y = np.asarray(y, dtype=np.float64)
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1))
    return (np.arange(n) * hop + frame / 2) / sr, 20 * np.log10(rms + _EPS)


def _energy_vad(y, sr, min_speech_s, min_silence_s, frame_s=0.03, hop_s=0.01):
    """VAD énergétique « à la main » : seuil adaptatif entre plancher de bruit et niveau de parole,
    avec hystérésis (seuil haut pour entrer en parole, seuil bas pour en sortir)."""
    times, e = frame_energy_db(y, sr, frame_s, hop_s)
    noise_floor = np.percentile(e, 10)
    speech_level = np.percentile(e, 95)
    if speech_level - noise_floor < 6:  # pas de contraste : tout ou rien
        return [(0.0, len(y) / sr)] if speech_level > -50 else []
    high = noise_floor + 0.5 * (speech_level - noise_floor)
    low = noise_floor + 0.3 * (speech_level - noise_floor)
    segments, active, start = [], False, 0.0
    for t, v in zip(times, e):
        if not active and v > high:
            active, start = True, max(0.0, t - frame_s / 2)
        elif active and v < low:
            active = False
            segments.append((start, t))
    if active:
        segments.append((start, len(y) / sr))
    # fusion des segments séparés par un silence trop court
    merged: list[list[float]] = []
    for s, e_ in segments:
        if merged and s - merged[-1][1] < min_silence_s:
            merged[-1][1] = e_
        else:
            merged.append([s, e_])
    return [(round(s, 3), round(e_, 3)) for s, e_ in merged if e_ - s >= min_speech_s]


_SILERO = None


def _silero_vad(y, sr, min_speech_s, min_silence_s):
    global _SILERO
    from silero_vad import get_speech_timestamps, load_silero_vad

    from .audio import resample

    if _SILERO is None:
        _SILERO = load_silero_vad()
    if sr != SAMPLE_RATE:
        y, sr = resample(y, sr, SAMPLE_RATE), SAMPLE_RATE
    import torch

    ts = get_speech_timestamps(torch.from_numpy(np.asarray(y, dtype=np.float32)), _SILERO, sampling_rate=sr,
                               min_speech_duration_ms=int(min_speech_s * 1000),
                               min_silence_duration_ms=int(min_silence_s * 1000), return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def vad_segments(y: np.ndarray, sr: int = SAMPLE_RATE, method: str = "silero", min_speech_s: float = 0.25,
                 min_silence_s: float = 0.3) -> list[tuple[float, float]]:
    """Segments de parole (début_s, fin_s). ``method`` : "silero" (réseau de neurones) ou "energy" (seuil)."""
    if method == "energy":
        return _energy_vad(y, sr, min_speech_s, min_silence_s)
    if method == "silero":
        return _silero_vad(y, sr, min_speech_s, min_silence_s)
    raise ValueError(f"Méthode de VAD inconnue : {method!r} (attendu : 'energy' ou 'silero')")


def estimate_snr(y: np.ndarray, sr: int = SAMPLE_RATE, method: str = "vad") -> float:
    """SNR estimé (dB) = 10·log10(P_parole / P_bruit), zones détectées par VAD énergétique.

    Renvoie ``nan`` s'il n'y a pas de silence exploitable (énoncé coupé au ras de la parole).
    """
    y = np.asarray(y, dtype=np.float64)
    segs = _energy_vad(y, sr, min_speech_s=0.1, min_silence_s=0.1)
    if not segs:
        return float("nan")
    mask = np.zeros(len(y), dtype=bool)
    for s, e in segs:
        mask[int(s * sr): int(e * sr)] = True
    if mask.all() or (~mask).sum() < 0.05 * sr:
        return float("nan")
    p_speech = np.mean(y[mask] ** 2)
    p_noise = np.mean(y[~mask] ** 2) + _EPS
    # la parole mesurée contient aussi du bruit : P_signal ≈ P_parole - P_bruit
    return float(10 * np.log10(max(p_speech - p_noise, _EPS) / p_noise))


def split_on_segments(y: np.ndarray, segments: list[tuple[float, float]], sr: int = SAMPLE_RATE,
                      pad_s: float = 0.1, max_segment_s: float = 15.0) -> list[np.ndarray]:
    """Découpe le signal selon les segments (avec marge) ; re-découpe les segments trop longs
    au point d'énergie minimale (pour respecter les limites des modèles, ex. 30 s pour Whisper)."""
    out: list[np.ndarray] = []

    def cut(chunk: np.ndarray):
        if len(chunk) <= max_segment_s * sr:
            out.append(chunk.astype(np.float32))
            return
        # chercher le minimum d'énergie dans la zone centrale [30 %, 70 %]
        times, e = frame_energy_db(chunk, sr)
        lo, hi = int(0.3 * len(e)), int(0.7 * len(e))
        k = lo + int(np.argmin(e[lo:hi])) if hi > lo else len(e) // 2
        split = int(times[k] * sr)
        cut(chunk[:split])
        cut(chunk[split:])

    for s, e in segments:
        a, b = max(0, int((s - pad_s) * sr)), min(len(y), int((e + pad_s) * sr))
        if b > a:
            cut(np.asarray(y[a:b]))
    return out


def denoise(y: np.ndarray, sr: int = SAMPLE_RATE, stationary: bool = True, prop_decrease: float = 0.8) -> np.ndarray:
    """Réduction de bruit par soustraction spectrale (noisereduce)."""
    import noisereduce as nr

    return nr.reduce_noise(y=np.asarray(y, dtype=np.float32), sr=sr, stationary=stationary,
                           prop_decrease=prop_decrease).astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, noise: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Ajoute un bruit (blanc par défaut) au SNR demandé — démos et augmentation."""
    y = np.asarray(y, dtype=np.float64)
    if noise is None:
        noise = np.random.default_rng(seed).standard_normal(len(y))
    else:
        noise = np.resize(np.asarray(noise, dtype=np.float64), len(y))
    p_signal = np.mean(y**2)
    p_noise = np.mean(noise**2) + _EPS
    noise = noise * np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    return (y + noise).astype(np.float32)


def preprocess(y: np.ndarray, sr: int, config: PreprocessConfig | None = None
               ) -> tuple[list[np.ndarray] | np.ndarray, PreprocessReport]:
    """Pipeline complet. Retourne un signal (sans VAD) ou une liste de segments (avec VAD) + un rapport."""
    from .audio import resample

    cfg = config or PreprocessConfig()
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:  # (canaux, n) → mono
        y = y.mean(axis=0)
    dur_in = len(y) / sr
    clipping = detect_clipping(y)
    snr_before = estimate_snr(y, sr)
    if sr != cfg.target_sr:
        y, sr = resample(y, sr, cfg.target_sr), cfg.target_sr
    y = remove_dc(y)
    if cfg.denoise:
        y = denoise(y, sr)
    if cfg.normalization == "peak":
        y = normalize_peak(y, cfg.peak_dbfs)
    elif cfg.normalization == "rms":
        y = normalize_rms(y, cfg.rms_dbfs)
    elif cfg.normalization == "lufs":
        y = normalize_loudness(y, sr, cfg.lufs)
    segments: list[tuple[float, float]] = []
    if cfg.vad:
        segments = vad_segments(y, sr, method=cfg.vad, min_speech_s=cfg.min_segment_s)
        chunks = split_on_segments(y, segments, sr, max_segment_s=cfg.max_segment_s)
        chunks = [c for c in chunks if len(c) >= cfg.min_segment_s * sr]
        out_dur = sum(len(c) for c in chunks) / sr
        report = PreprocessReport(dur_in, out_dur, snr_before,
                                  estimate_snr(np.concatenate(chunks), sr) if chunks else None, clipping, segments)
        return chunks, report
    y, _ = trim_silence(y, cfg.trim_top_db)
    if cfg.pad_s > 0:
        y = np.pad(y, int(cfg.pad_s * sr)).astype(np.float32)
    return y, PreprocessReport(dur_in, len(y) / sr, snr_before, estimate_snr(y, sr), clipping, segments)
