"""Entrées/sorties audio.

Convention du module : signaux ``numpy.ndarray`` float32 mono dans [-1, 1], 16 kHz par défaut.

Choix de conception : on décode avec ``soundfile`` (libsndfile) et, pour les formats compressés
(mp3, m4a, opus…), avec ``librosa``/``audioread`` (FFmpeg). On évite volontairement de dépendre de
``torchcodec`` (sensible aux versions de torch/CUDA/FFmpeg) : le module reste robuste sur Colab et en local.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE

_SOUNDFILE_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def _to_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    return y.astype(np.float32, copy=False)


def decode_bytes(data: bytes, sr: int | None = SAMPLE_RATE, mono: bool = True) -> tuple[np.ndarray, int]:
    """Décode un fichier audio présent en mémoire (ex. colonne ``audio`` d'un Parquet Hugging Face)."""
    import soundfile as sf

    y, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = y.T  # (canaux, n)
    if mono:
        y = y.mean(axis=0)
    if sr is not None and sr != file_sr:
        y = resample(y, file_sr, sr)
        file_sr = sr
    return _to_float32(y), int(file_sr)


def load_audio(path: str | Path, sr: int | None = SAMPLE_RATE, mono: bool = True) -> tuple[np.ndarray, int]:
    """Charge un fichier audio (wav, flac, ogg, mp3, m4a…).

    Args:
        path: chemin du fichier.
        sr: fréquence cible ; ``None`` conserve la fréquence native.
        mono: moyenne des canaux si True.

    Returns:
        (signal float32 de forme (n,) si mono sinon (canaux, n), fréquence d'échantillonnage).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier audio introuvable : {path}")
    if path.suffix.lower() in _SOUNDFILE_EXT:
        import soundfile as sf

        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.T
    else:  # formats compressés : décodage via FFmpeg (audioread)
        import librosa

        y, file_sr = librosa.load(str(path), sr=None, mono=False)
        y = np.atleast_2d(y)
    if mono:
        y = y.mean(axis=0)
    if sr is not None and sr != file_sr:
        y = resample(y, int(file_sr), sr)
        file_sr = sr
    return _to_float32(y), int(file_sr)


def save_audio(path: str | Path, y: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Écrit un WAV PCM 16 bits ; crée les dossiers parents. Retourne le chemin écrit."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(_to_float32(y), -1.0, 1.0)
    sf.write(str(path), y.T if y.ndim == 2 else y, sr, subtype="PCM_16")
    return path


def audio_info(path: str | Path) -> dict:
    """Métadonnées sans décoder tout le fichier : ``sr``, ``channels``, ``duration_s``, ``frames``,
    ``format``, ``subtype`` (ex. PCM_16), ``bit_depth`` (int ou None)."""
    path = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "PCM_U8": 8, "PCM_S8": 8, "FLOAT": 32, "DOUBLE": 64}
        return {
            "path": str(path),
            "sr": int(info.samplerate),
            "channels": int(info.channels),
            "duration_s": float(info.frames / info.samplerate),
            "frames": int(info.frames),
            "format": info.format,
            "subtype": info.subtype,
            "bit_depth": bits.get(info.subtype),
        }
    except Exception:  # format compressé non lu par libsndfile
        y, sr = load_audio(path, sr=None, mono=False)
        y = np.atleast_2d(y)
        return {
            "path": str(path), "sr": sr, "channels": y.shape[0], "duration_s": y.shape[1] / sr,
            "frames": y.shape[1], "format": path.suffix.lstrip(".").upper(), "subtype": "compressé",
            "bit_depth": None,
        }


def resample(y: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Rééchantillonne avec filtre anti-repliement (librosa, ``res_type='soxr_hq'``)."""
    import librosa

    if orig_sr == target_sr:
        return _to_float32(y)
    return _to_float32(librosa.resample(_to_float32(y), orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq"))


def naive_downsample(y: np.ndarray, factor: int) -> np.ndarray:
    """Décimation **sans** filtre anti-repliement (``y[::factor]``) — sert à démontrer le repliement (S2)."""
    return _to_float32(np.asarray(y)[..., ::factor])


def play(y: np.ndarray | str | Path, sr: int = SAMPLE_RATE, label: str | None = None, normalize: bool = True):
    """Lecteur audio IPython (accepte un signal ou un chemin). Affiche ``label`` au-dessus s'il est fourni.

    ``normalize=False`` conserve le niveau réel (indispensable pour comparer des volumes).
    """
    from IPython.display import HTML, Audio, display

    if isinstance(y, (str, Path)):
        y, sr = load_audio(y, sr=None)
    if label:
        display(HTML(f"<b>{label}</b>"))
    player = Audio(np.asarray(y), rate=sr, normalize=normalize)
    display(player)
    return None


def synth_tone(freq: float, duration_s: float = 1.0, sr: int = SAMPLE_RATE, amplitude: float = 0.5) -> np.ndarray:
    """Sinusoïde pure (démos d'échantillonnage / repliement)."""
    t = np.arange(int(round(duration_s * sr))) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def quantize(y: np.ndarray, bits: int) -> np.ndarray:
    """Requantifie un signal sur ``bits`` bits (démonstration du bruit de quantification, S2)."""
    levels = 2 ** (bits - 1)
    return (np.round(np.clip(y, -1, 1) * (levels - 1)) / (levels - 1)).astype(np.float32)


def record(seconds: float = 3.0, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Enregistre depuis le micro du navigateur **sur Google Colab** (renvoie signal, sr).

    Hors Colab, lève une erreur explicite : utiliser un fichier enregistré au smartphone.
    """
    try:
        from google.colab.output import eval_js  # type: ignore
    except ImportError as exc:
        raise RuntimeError("record() ne fonctionne que sur Google Colab. Enregistrez un fichier et utilisez load_audio().") from exc
    import base64

    from IPython.display import Javascript, display

    js = Javascript(f"""
    async function recordAudio() {{
      const stream = await navigator.mediaDevices.getUserMedia({{audio: true}});
      const recorder = new MediaRecorder(stream); const chunks = [];
      recorder.ondataavailable = e => chunks.push(e.data);
      recorder.start(); await new Promise(r => setTimeout(r, {int(seconds * 1000)})); recorder.stop();
      await new Promise(r => recorder.onstop = r);
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks); const buf = await blob.arrayBuffer();
      let binary = ''; const bytes = new Uint8Array(buf);
      for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    }}""")
    display(js)
    print(f"🎙️ Enregistrement pendant {seconds:.0f} s… parlez !")
    data = base64.b64decode(eval_js("recordAudio()"))
    tmp = Path("/tmp/m1_record.webm")
    tmp.write_bytes(data)
    return load_audio(tmp, sr=sr)
