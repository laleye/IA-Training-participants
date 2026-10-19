"""Accès aux jeux de données du cours (kit de données, ticket M1-G09).

Les packs sont **construits à la demande** à partir de sources ouvertes vérifiées (voir ``data/sources.yaml``) :
- SimbaBench (UBC-NLP, CC-BY-4.0) qui redistribue les tests ALFFA fon/wol (MIT) et Kallaama wol (CC-BY-4.0) ;
- MInDS-14 (PolyAI, CC-BY-4.0) pour quelques énoncés téléphoniques en français.
Ils sont écrits dans ``data_dir()/<pack>/`` (WAV 16 kHz mono + ``manifest.csv`` + ``LICENSE.txt``). Idempotent.

Dossier : variable d'environnement ``M1_DATA_DIR`` sinon ``/content/m1_data`` (Colab) ou ``~/m1_data``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SIMBA_REPO = "UBC-NLP/SimbaBench_dataset"
SIMBA_FILES = {
    "fon_alffa": "asr_test/HF_test-fon*Alffa_Public.parquet",
    "wol_alffa": "asr_test/HF_test-wol*Alffa_Public.parquet",
    "wol_kallaama": "asr_test/HF_test-wol*Kallaama.parquet",
}
SEED = 2026
RESOURCES = Path(__file__).resolve().parent / "resources"

CITATIONS = """Sources (merci de citer en cas de réutilisation) :
- SimbaBench — Elmadany et al. (2025). Voice of a Continent: Mapping Africa's Speech Technology Frontier. EMNLP 2025.
  https://aclanthology.org/2025.emnlp-main.559/ — licence CC-BY-4.0.
- ALFFA (fon) — Laleye et al. (2016), FedCSIS, DOI 10.15439/2016F153 ; ALFFA (wolof) — Gauthier et al. (2016), LREC.
  https://github.com/getalp/ALFFA_PUBLIC — licence MIT.
- Kallaama (wolof, parole spontanée agricole) — Gauthier, Ndiaye, Guissé (2024), arXiv:2404.01991 ; financé par Lacuna Fund.
  https://github.com/gauthelo/kallaama-speech-dataset — licence CC-BY-4.0.
- MInDS-14 (français) — Gerz et al. (2021). https://huggingface.co/datasets/PolyAI/minds14 — licence CC-BY-4.0.
"""

PACKS = {
    "demo_s1": {"desc": "4 fon + 4 wol (ALFFA, lu) + 2 wol (Kallaama, spontané) + 2 fra (MInDS-14, téléphone)", "used_in": "S0, S1"},
    "testset_s3": {"desc": "Jeu de test commun : 20 fon + 20 wol lus (ALFFA) + 10 wol spontanés (Kallaama)", "used_in": "S3, S4, TP3"},
    "long_recording_s2": {"desc": "Enregistrement long réel (Kallaama, radio/entretien, ≈ 2–3 min) à segmenter", "used_in": "S2, TP2"},
    "minimal_pairs": {"desc": "Paires minimales (tons fon ; longueur/gémination wol). Voix SYNTHÉTIQUE en attendant "
                               "les enregistrements de locuteurs natifs", "used_in": "S1, S2, TD2, TP1"},
}


def data_dir() -> Path:
    """Dossier racine des données (variable d'environnement ``M1_DATA_DIR`` sinon défaut)."""
    if os.environ.get("M1_DATA_DIR"):
        d = Path(os.environ["M1_DATA_DIR"])
    elif Path("/content").exists():
        d = Path("/content/m1_data")
    else:
        d = Path.home() / "m1_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_packs():
    """Liste des packs disponibles : name, description, used_in, ready (déjà construit ?)."""
    import pandas as pd

    return pd.DataFrame([{"name": k, "description": v["desc"], "used_in": v["used_in"],
                          "ready": (data_dir() / k / "manifest.csv").exists()} for k, v in PACKS.items()])


# ------------------------------------------------------------------------------------------- sources
def _simba(source: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SIMBA_REPO, SIMBA_FILES[source], repo_type="dataset")
    df = pd.read_parquet(path)
    df["text"] = df["text"].str.strip()
    return df.sort_values("benchmark_id").reset_index(drop=True)


def _select(df, n: int, min_s: float, max_s: float, seed: int = SEED, exclude_markers: bool = True):
    ok = df[(df.duration_s >= min_s) & (df.duration_s <= max_s)]
    if exclude_markers:  # énoncés réduits à des hésitations ou majoritairement inintelligibles
        ok = ok[~ok.text.str.fullmatch(r"(\s*(%\w+|\*)\s*)+") & (ok.text.str.split().str.len() >= 3)]
    return ok.sample(n=min(n, len(ok)), random_state=seed).sort_values("benchmark_id")


def _write_rows(rows, out: Path, lang: str, source: str, license: str) -> list[dict]:
    from .audio import decode_bytes, save_audio

    records = []
    for _, r in rows.iterrows():
        y, sr = decode_bytes(r["audio"]["bytes"], sr=16_000)
        uid = r["benchmark_id"].replace("*", "_")
        p = save_audio(out / "audio" / f"{uid}.wav", y, sr)
        records.append({"id": uid, "path": str(p.relative_to(out)), "text": r["text"], "lang": lang,
                        "speaker_id": "inconnu", "duration_s": round(len(y) / sr, 3), "sr": sr,
                        "source": source, "license": license})
    return records


def _minds14_fr(n: int, out: Path) -> list[dict]:
    """Quelques énoncés français de MInDS-14 (PolyAI, CC-BY-4.0) : parole téléphonique réelle à 8 kHz
    (requêtes bancaires) — utile aussi pour comparer téléphonie (8 kHz) et ASR (16 kHz) en S2."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from .audio import decode_bytes, save_audio

    path = hf_hub_download("PolyAI/minds14", "fr-FR/train/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet")
    df = pd.read_parquet(path)
    df = df[df.transcription.str.split().str.len().between(6, 20)].sample(n=n, random_state=SEED)
    records = []
    for k, (_, r) in enumerate(df.iterrows()):
        y8, sr8 = decode_bytes(r["audio"]["bytes"], sr=None)
        uid = f"minds14_fr_{k:02d}"
        save_audio(out / "audio" / f"{uid}_8k.wav", y8, sr8)  # original téléphonique conservé
        y, sr = decode_bytes(r["audio"]["bytes"], sr=16_000)
        p = save_audio(out / "audio" / f"{uid}.wav", y, sr)
        records.append({"id": uid, "path": str(p.relative_to(out)), "text": r["transcription"], "lang": "fra",
                        "speaker_id": "inconnu", "duration_s": round(len(y) / sr, 3), "sr": sr,
                        "source": f"MInDS-14 fr-FR (téléphone {sr8} Hz, intention bancaire)", "license": "CC-BY-4.0"})
    return records


# ------------------------------------------------------------------------------------------- builders
def _build_demo_s1(out: Path) -> list[dict]:
    rec = _write_rows(_select(_simba("fon_alffa"), 4, 1.8, 5), out, "fon", "ALFFA (via SimbaBench)", "MIT")
    rec += _write_rows(_select(_simba("wol_alffa"), 4, 2.5, 6), out, "wol", "ALFFA (via SimbaBench)", "MIT")
    rec += _write_rows(_select(_simba("wol_kallaama"), 2, 4, 9), out, "wol", "Kallaama (via SimbaBench)", "CC-BY-4.0")
    try:
        rec += _minds14_fr(2, out)
    except Exception as exc:  # réseau / quota : le pack reste utilisable sans le français
        print(f"⚠️ Énoncés français MInDS-14 non récupérés ({type(exc).__name__}) : pack construit sans français.")
    return rec


def _build_testset_s3(out: Path) -> list[dict]:
    rec = _write_rows(_select(_simba("fon_alffa"), 20, 2, 7, seed=SEED + 1), out, "fon", "ALFFA (via SimbaBench)", "MIT")
    rec += _write_rows(_select(_simba("wol_alffa"), 20, 2.5, 7, seed=SEED + 1), out, "wol", "ALFFA (via SimbaBench)", "MIT")
    rec += _write_rows(_select(_simba("wol_kallaama"), 10, 3, 12, seed=SEED + 1), out, "wol",
                       "Kallaama (via SimbaBench)", "CC-BY-4.0")
    for r in rec:
        r["domain"] = "spontané" if "Kallaama" in r["source"] else "lu"
    return rec


def _build_long_recording(out: Path) -> list[dict]:
    df = _simba("wol_kallaama")
    long = df[(df.duration_s >= 90) & (df.duration_s <= 200)].sort_values("duration_s", ascending=False)
    if long.empty:
        long = df.sort_values("duration_s", ascending=False)
    rec = _write_rows(long.head(1), out, "wol", "Kallaama (via SimbaBench)", "CC-BY-4.0")
    # second fichier : la même parole avec un bruit de fond réaliste ajouté (ventilateur/rue simulé) à 10 dB
    from .audio import load_audio, save_audio
    from .preprocess import add_noise

    y, sr = load_audio(out / rec[0]["path"])
    rng = np.random.default_rng(SEED)
    brown = np.cumsum(rng.standard_normal(len(y)))  # bruit brun (grave), plus réaliste qu'un bruit blanc
    brown = (brown - np.convolve(brown, np.ones(2001) / 2001, mode="same")).astype(np.float32)
    noisy = add_noise(y, snr_db=10, noise=brown)
    p = save_audio(out / "audio" / f"{rec[0]['id']}_bruit10dB.wav", noisy, sr)
    rec.append({**rec[0], "id": p.stem, "path": str(p.relative_to(out)), "source": rec[0]["source"] + " + bruit synthétique"})
    return rec


def _build_minimal_pairs(out: Path) -> list[dict]:
    """Synthèse (TTS) des paires minimales de ``resources/minimal_pairs.csv`` en attendant des voix natives."""
    import pandas as pd

    from .asr import load_tts, unload
    from .audio import resample, save_audio

    pairs = pd.read_csv(RESOURCES / "minimal_pairs.csv", dtype=str).fillna("")
    records = []
    for lang, key in (("fon", "mms-tts-fon"), ("wol", "speecht5-wol")):
        sub = pairs[pairs.lang == lang]
        if sub.empty:
            continue
        tts = load_tts(key)
        for _, r in sub.iterrows():
            y, sr = tts(r["texte"], seed=0)
            y = resample(y, sr, 16_000) if sr != 16_000 else y
            p = save_audio(out / "audio" / f"{r['id']}.wav", y, 16_000)
            records.append({"id": r["id"], "path": str(p.relative_to(out)), "text": r["texte"], "lang": lang,
                            "speaker_id": f"tts_{key}", "duration_s": round(len(y) / 16_000, 3), "sr": 16_000,
                            "pair": r["paire"], "contrast": r["contraste"], "api": r["api"], "gloss_fr": r["traduction_fr"],
                            "source": f"SYNTHÉTIQUE ({key}) — à remplacer par des locuteurs natifs", "license": "CC-BY-NC-4.0"})
        unload(tts)
    return records


_BUILDERS = {"demo_s1": _build_demo_s1, "testset_s3": _build_testset_s3,
             "long_recording_s2": _build_long_recording, "minimal_pairs": _build_minimal_pairs}


def get_pack(name: str, force: bool = False) -> Path:
    """Construit (si besoin) un pack et renvoie son dossier. Idempotent : le second appel est instantané.

    Le dossier contient ``manifest.csv`` (chemins relatifs au dossier), ``audio/*.wav``, ``LICENSE.txt``.
    """
    import pandas as pd

    if name not in _BUILDERS:
        raise KeyError(f"Pack inconnu : {name!r}. Packs disponibles : {', '.join(_BUILDERS)}")
    out = data_dir() / name
    manifest = out / "manifest.csv"
    if manifest.exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"📦 Construction du pack {name!r} dans {out} …")
    records = _BUILDERS[name](out)
    pd.DataFrame(records).to_csv(manifest, index=False)
    (out / "LICENSE.txt").write_text(CITATIONS, encoding="utf-8")
    print(f"✅ {len(records)} fichiers prêts.")
    return out


def load_manifest(name: str):
    """Manifest d'un pack (DataFrame) avec une colonne ``abspath`` (chemin absolu des fichiers audio)."""
    import pandas as pd

    root = get_pack(name)
    df = pd.read_csv(root / "manifest.csv")
    df["abspath"] = df["path"].map(lambda p: str(root / p))
    return df


def load_cached_predictions(model: str, lang: str | None = None):
    """Prédictions pré-calculées sur ``testset_s3`` (plan B si GPU/Hub indisponible) : id, lang, hyp, latency_s."""
    import pandas as pd

    path = RESOURCES / "predictions" / f"{model}.csv"
    if not path.exists():
        available = sorted(p.stem for p in (RESOURCES / "predictions").glob("*.csv"))
        raise FileNotFoundError(f"Pas de prédictions en cache pour {model!r}. Disponibles : {available}")
    df = pd.read_csv(path)
    return df[df.lang == lang].reset_index(drop=True) if lang else df


def load_cached_logits(utt_id: str, model: str = "simba-h") -> tuple[np.ndarray, list[str], int]:
    """Logits CTC pré-calculés (plan B de l'atelier S3-NB01) : (log_probs, vocab, blank_id)."""
    path = RESOURCES / "logits" / f"{model}__{utt_id}.npz"
    data = np.load(path, allow_pickle=False)
    return data["log_probs"].astype(np.float32), [str(v) for v in data["vocab"]], int(data["blank_id"])
