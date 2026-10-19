"""Visualisations pédagogiques.

Toutes les fonctions acceptent ``ax=None`` (crée une figure) et renvoient l'``Axes`` utilisé,
pour pouvoir composer des figures multi-panneaux. Axes toujours étiquetés avec unités.
"""
from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE


def _ax(ax, figsize=(10, 3.5)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE, ax=None, title: str | None = None, segments=None):
    """Forme d'onde (temps en s). ``segments`` : liste de (début_s, fin_s) surlignés (résultat de VAD)."""
    ax = _ax(ax)
    t = np.arange(len(y)) / sr
    ax.plot(t, y, lw=0.6, color="#1f3a5f")
    for start, end in segments or []:
        ax.axvspan(start, end, color="#e07a1f", alpha=0.25)
    ax.set(xlabel="Temps (s)", ylabel="Amplitude", title=title or "Forme d'onde", xlim=(0, t[-1] if len(t) else 1))
    return ax


def plot_spectrum(y: np.ndarray, sr: int = SAMPLE_RATE, ax=None, max_freq: float | None = 5000, title: str | None = None):
    """Spectre d'amplitude (FFT) en dB d'un segment court (ex. une voyelle)."""
    ax = _ax(ax)
    y = np.asarray(y, dtype=np.float32) * np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    ax.plot(freqs, 20 * np.log10(spec + 1e-10), lw=0.8, color="#1f3a5f")
    ax.set(xlabel="Fréquence (Hz)", ylabel="Amplitude (dB)", title=title or "Spectre", xlim=(0, max_freq or sr / 2))
    return ax


def plot_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = 512, hop_length: int = 160,
                     db: bool = True, max_freq: float | None = 8000, ax=None, title: str | None = None):
    """Spectrogramme STFT (linéaire ou dB) avec barre de couleur."""
    import librosa.display
    import matplotlib.pyplot as plt

    from .features import stft_magnitude

    ax = _ax(ax)
    S = stft_magnitude(y, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(S, ref=np.max) if db else S
    img = librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set(ylim=(0, max_freq or sr / 2), xlabel="Temps (s)", ylabel="Fréquence (Hz)",
           title=title or f"Spectrogramme (n_fft={n_fft}, hop={hop_length})")
    plt.colorbar(img, ax=ax, format="%+2.0f dB" if db else "%.2f")
    return ax


def plot_mel(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 80, ax=None, title: str | None = None):
    """Spectrogramme log-Mel (paramètres Whisper par défaut : 25 ms / 10 ms, 80 bandes)."""
    import librosa.display
    import matplotlib.pyplot as plt

    from .features import log_mel

    ax = _ax(ax)
    M = log_mel(y, sr=sr, n_mels=n_mels)
    img = librosa.display.specshow(M, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
    ax.set(xlabel="Temps (s)", ylabel="Fréquence (échelle Mel)", title=title or f"Log-Mel ({n_mels} bandes)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    return ax


def plot_f0(y: np.ndarray, sr: int = SAMPLE_RATE, ax=None, title: str | None = None, overlay_spectrogram: bool = True,
            fmin: float = 75, fmax: float = 400):
    """Contour de F0 (Parselmouth) superposé au spectrogramme ; en Hz, points non voisés masqués."""
    from .features import f0_contour

    ax = _ax(ax)
    t, f0 = f0_contour(y, sr, fmin=fmin, fmax=fmax)
    if overlay_spectrogram:
        import librosa.display

        from .features import stft_magnitude

        S = stft_magnitude(y, n_fft=1024, hop_length=160)
        import librosa

        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=160, x_axis="time",
                                 y_axis="hz", ax=ax, cmap="Greys")
        ax.set_ylim(0, max(600, fmax * 1.5))
    ax.plot(t, f0, "o-", ms=3, lw=2, color="#e07a1f", label="F0")
    ax.set(xlabel="Temps (s)", ylabel="Fréquence (Hz)", title=title or "Contour de F0")
    ax.legend(loc="upper right")
    return ax


def compare_f0(items: list[tuple[str, np.ndarray]], sr: int = SAMPLE_RATE, normalize_time: bool = True, ax=None,
               semitones: bool = False, fmin: float = 75, fmax: float = 400):
    """Superpose les contours de F0 de plusieurs énoncés (paires minimales tonales).

    ``items`` : liste de (étiquette, signal). ``normalize_time`` : abscisse en % de la durée.
    ``semitones`` : F0 en demi-tons relatifs à la médiane de chaque énoncé (compare des locuteurs différents).
    """
    from .features import f0_contour, f0_semitones

    ax = _ax(ax)
    for label, y in items:
        t, f0 = f0_contour(y, sr, fmin=fmin, fmax=fmax)
        x = 100 * t / (len(y) / sr) if normalize_time else t
        ax.plot(x, f0_semitones(f0) if semitones else f0, "o-", ms=3, lw=2, label=label)
    ax.set(xlabel="Durée (%)" if normalize_time else "Temps (s)",
           ylabel="F0 (demi-tons / médiane)" if semitones else "F0 (Hz)", title="Comparaison des contours de F0")
    ax.legend()
    return ax


def plot_vowel_space(formants, ax=None):
    """Triangle vocalique F1/F2 (axes inversés, convention phonétique). Colonnes : label, F1, F2."""
    ax = _ax(ax, figsize=(6, 5))
    ax.scatter(formants["F2"], formants["F1"], s=60, color="#1f3a5f")
    for _, row in formants.iterrows():
        ax.annotate(str(row["label"]), (row["F2"], row["F1"]), fontsize=14, xytext=(5, 5), textcoords="offset points")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set(xlabel="F2 (Hz) ← antérieure", ylabel="F1 (Hz) ← fermée", title="Espace vocalique")
    return ax


def plot_emissions(log_probs: np.ndarray, vocab: list[str], frame_s: float = 0.02, top_k: int | None = 20,
                   path: list[int] | None = None, ax=None, title: str | None = None):
    """Matrice d'émission CTC (temps × tokens) en carte de chaleur ; ``path`` : meilleur chemin superposé.

    Seuls les ``top_k`` tokens les plus probables sur l'énoncé sont affichés (lisibilité).
    """
    import matplotlib.pyplot as plt

    ax = _ax(ax, figsize=(12, 5))
    probs = np.exp(log_probs)
    keep = np.argsort(-probs.max(axis=0))[: top_k or probs.shape[1]]
    keep = np.sort(keep)
    img = ax.imshow(probs[:, keep].T, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1,
                    extent=(0, probs.shape[0] * frame_s, -0.5, len(keep) - 0.5))
    labels = [repr(vocab[i]) if vocab[i].strip() else "␣" for i in keep]
    ax.set_yticks(range(len(keep)), labels, fontsize=9)
    if path is not None:
        pos = {tok: i for i, tok in enumerate(keep)}
        pts = [(t * frame_s + frame_s / 2, pos[tok]) for t, tok in enumerate(path) if tok in pos]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=8, color="#e07a1f", label="chemin glouton")
            ax.legend(loc="upper right")
    ax.set(xlabel="Temps (s)", ylabel="Token", title=title or "Matrice d'émission CTC (probabilités)")
    plt.colorbar(img, ax=ax, label="probabilité")
    return ax
