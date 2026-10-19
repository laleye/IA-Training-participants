"""Construction et contrôle de corpus de parole.

Format de manifest du module (CSV, UTF-8, une ligne par énoncé) :
    id, path, text, lang, speaker_id, gender, age_range, dialect, duration_s, sr, snr_db,
    recording_device, environment, consent_id, license, split

Règle d'or : **aucune donnée directement identifiante** (nom, téléphone, adresse) dans le manifest.
``speaker_id`` est un pseudonyme (ex. ``fon_spk01``) ; la table de correspondance reste hors ligne.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MANIFEST_COLUMNS = [
    "id", "path", "text", "lang", "speaker_id", "gender", "age_range", "dialect", "duration_s", "sr",
    "snr_db", "recording_device", "environment", "consent_id", "license", "split",
]
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}


def _read_transcripts(transcripts) -> dict[str, str]:
    import pandas as pd

    if isinstance(transcripts, dict):
        return {str(k): str(v) for k, v in transcripts.items()}
    df = pd.read_csv(transcripts, dtype=str).fillna("")
    id_col = "id" if "id" in df.columns else df.columns[0]
    text_col = "text" if "text" in df.columns else ("texte" if "texte" in df.columns else df.columns[1])
    return dict(zip(df[id_col], df[text_col]))


def build_manifest(audio_dir: str | Path, transcripts: str | Path | dict, lang: str,
                   speakers: str | Path | None = None, compute_snr: bool = True):
    """Assemble le manifest à partir d'un dossier d'audios, des transcriptions (CSV id,text ou dict)
    et d'une table optionnelle de métadonnées locuteurs ; calcule duration_s, sr, snr_db.

    L'``id`` d'un énoncé est le nom de fichier sans extension. Convention du cours :
    ``{langue}_{speaker_id}_{id_item}.wav`` → le ``speaker_id`` est déduit du nom si absent des métadonnées.
    """
    import pandas as pd

    from .audio import audio_info, load_audio
    from .preprocess import estimate_snr

    audio_dir = Path(audio_dir)
    texts = _read_transcripts(transcripts)
    rows = []
    for p in sorted(f for f in audio_dir.rglob("*") if f.suffix.lower() in AUDIO_EXT):
        info = audio_info(p)
        parts = p.stem.split("_")
        row = {c: None for c in MANIFEST_COLUMNS}
        row.update({"id": p.stem, "path": str(p), "text": texts.get(p.stem, ""), "lang": lang,
                    "speaker_id": "_".join(parts[:2]) if len(parts) >= 3 else None,
                    "duration_s": round(info["duration_s"], 3), "sr": info["sr"]})
        if compute_snr:
            y, sr = load_audio(p)
            snr = estimate_snr(y, sr)
            row["snr_db"] = None if np.isnan(snr) else round(snr, 1)
        rows.append(row)
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if speakers is not None:
        spk = pd.read_csv(speakers, dtype=str)
        meta_cols = [c for c in spk.columns if c in MANIFEST_COLUMNS and c != "speaker_id"]
        df = df.drop(columns=meta_cols).merge(spk[["speaker_id", *meta_cols]], on="speaker_id", how="left")
        df = df[MANIFEST_COLUMNS]
    return df


def quality_checks(df, lang: str, min_s: float = 0.5, max_s: float = 20.0, min_snr_db: float = 10.0):
    """Ajoute des colonnes booléennes de problèmes : too_short, too_long, low_snr, clipping,
    empty_text, out_of_alphabet, duplicate_text, chars_per_second_outlier ; + colonne ``ok``."""
    from .audio import load_audio
    from .preprocess import detect_clipping
    from .text import normalize_text, out_of_alphabet

    df = df.copy()
    norm = df["text"].fillna("").map(lambda t: normalize_text(t, lang))
    df["too_short"] = df["duration_s"] < min_s
    df["too_long"] = df["duration_s"] > max_s
    df["low_snr"] = df["snr_db"].astype(float) < min_snr_db
    if "path" in df and df["path"].notna().all():
        df["clipping"] = [detect_clipping(load_audio(p)[0]) > 0.001 for p in df["path"]]
    else:
        df["clipping"] = False
    df["empty_text"] = norm.str.len() == 0
    df["out_of_alphabet"] = df["text"].fillna("").map(lambda t: sorted(out_of_alphabet(normalize_text(t, lang), lang)))
    df["out_of_alphabet"] = df["out_of_alphabet"].map(lambda s: "".join(s))
    df["duplicate_text"] = norm.duplicated(keep=False) & ~df["empty_text"]
    cps = norm.str.replace(" ", "").str.len() / df["duration_s"].clip(lower=0.1)
    q1, q3 = cps.quantile(0.25), cps.quantile(0.75)
    iqr = q3 - q1
    df["chars_per_s"] = cps.round(1)
    df["chars_per_second_outlier"] = (cps < q1 - 1.5 * iqr) | (cps > q3 + 1.5 * iqr)
    problems = ["too_short", "too_long", "low_snr", "clipping", "empty_text", "duplicate_text", "chars_per_second_outlier"]
    df["ok"] = ~df[problems].any(axis=1) & (df["out_of_alphabet"] == "")
    return df


def corpus_stats(df) -> dict:
    """Statistiques : durée totale, nb énoncés, nb locuteurs, répartition genre/âge/dialecte, vocabulaire."""
    words = " ".join(df["text"].fillna("")).split()
    stats = {
        "n_utterances": int(len(df)),
        "total_min": round(float(df["duration_s"].sum()) / 60, 2),
        "mean_duration_s": round(float(df["duration_s"].mean()), 2),
        "n_speakers": int(df["speaker_id"].nunique()),
        "n_words": len(words),
        "vocabulary_size": len(set(words)),
    }
    for col in ("gender", "age_range", "dialect", "split"):
        if col in df and df[col].notna().any():
            stats[f"by_{col}_min"] = (df.groupby(col)["duration_s"].sum() / 60).round(2).to_dict()
    return stats


def speaker_split(df, test_size: float = 0.2, dev_size: float = 0.1, seed: int = 42):
    """Affecte ``split`` (train/dev/test) **par locuteur** (aucun locuteur dans deux splits).

    Les proportions sont visées en **durée**. Avec 2 locuteurs seulement : un en train, un en test, pas de dev.
    """
    import warnings

    df = df.copy()
    dur = df.groupby("speaker_id")["duration_s"].sum() if "duration_s" in df else df.groupby("speaker_id").size()
    speakers = list(dur.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    if len(speakers) < 2:
        raise ValueError("Impossible de séparer train/test avec un seul locuteur.")
    if len(speakers) == 2:
        warnings.warn("2 locuteurs seulement : un en train, un en test (pas de dev). Résultats très fragiles.")
        mapping = {speakers[0]: "train", speakers[1]: "test"}
    else:
        total = float(dur.sum())
        cibles = {"test": test_size * total, "dev": dev_size * total}
        cumul = {"test": 0.0, "dev": 0.0}
        mapping = {}
        for spk in speakers:
            d = float(dur[spk])
            for split in ("test", "dev"):
                vide = split not in mapping.values() and cibles[split] > 0
                # on ajoute le locuteur si le split est vide, ou si cela RAPPROCHE de la cible
                if vide or abs(cumul[split] + d - cibles[split]) < abs(cumul[split] - cibles[split]):
                    mapping[spk], cumul[split] = split, cumul[split] + d
                    break
            else:
                mapping[spk] = "train"
        if "train" not in mapping.values():  # garantir un train non vide
            mapping[speakers[-1]] = "train"
    df["split"] = df["speaker_id"].map(mapping)
    return df


def to_hf_dataset(df, audio_column: str = "path", sr: int = 16_000, decode: bool = True):
    """Convertit en ``datasets.DatasetDict`` (un split par valeur de ``split``) avec colonne ``audio`` typée Audio.

    Les octets audio sont embarqués dans le dataset (``{"bytes", "path"}``) : le jeu publié est autonome.
    ``decode=False`` évite de dépendre de ``torchcodec`` pour la lecture : on décode avec ``m1utils.audio.decode_bytes``.
    """
    from datasets import Audio, Dataset, DatasetDict

    from .audio import load_audio, save_audio

    records = df.copy()
    import io

    def to_bytes(p):
        y, _ = load_audio(p, sr=sr)
        buf = io.BytesIO()
        import soundfile as sf

        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        return {"bytes": buf.getvalue(), "path": Path(p).name}

    records["audio"] = records[audio_column].map(to_bytes)
    records = records.drop(columns=[audio_column]).astype({c: str for c in ("gender", "age_range", "dialect",
                                                                            "recording_device", "environment",
                                                                            "consent_id", "license") if c in records})
    splits = {}
    for name, part in records.groupby(records["split"].fillna("train")):
        ds = Dataset.from_pandas(part.reset_index(drop=True), preserve_index=False)
        splits[name] = ds.cast_column("audio", Audio(sampling_rate=sr, decode=decode))
    del save_audio
    return DatasetDict(splits)


def _size_category(n: int) -> str:
    for bound, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < bound:
            return label
    return "100K<n<1M"


def render_dataset_card(df, name: str, lang: str, license: str, description: str,
                        template: str | Path | None = None, **fields) -> str:
    """Génère une dataset card (README.md du Hub) à partir du modèle du cours et des statistiques.

    Les champs non calculables (protocole, consentement…) se passent en ``**fields`` ; ceux qui manquent
    sont remplis par « À compléter » pour rester visibles.
    """
    import re

    if template is None:
        template = Path(__file__).resolve().parents[1] / "module1" / "commun" / "modeles" / "dataset_card_template.md"
        if not template.exists():  # installation pip : modèle embarqué dans le paquet
            template = Path(__file__).resolve().parent / "resources" / "dataset_card_template.md"
    text = Path(template).read_text(encoding="utf-8")
    stats = corpus_stats(df)
    values = {
        "pretty_name": name, "lang": lang, "license": license, "description": description,
        "size_category": _size_category(stats["n_utterances"]),
        "n_utterances": stats["n_utterances"], "total_min": stats["total_min"],
        "n_speakers": stats["n_speakers"], "mean_duration_s": stats["mean_duration_s"],
        "vocabulary_size": stats["vocabulary_size"],
        "splits": ", ".join(f"{k} : {v} min" for k, v in stats.get("by_split_min", {}).items()) or "non découpé",
        "gender": ", ".join(f"{k} : {v} min" for k, v in stats.get("by_gender_min", {}).items()) or "non renseigné",
        "age_range": ", ".join(f"{k} : {v} min" for k, v in stats.get("by_age_range_min", {}).items()) or "non renseigné",
        "dialect": ", ".join(f"{k} : {v} min" for k, v in stats.get("by_dialect_min", {}).items()) or "non renseigné",
    }
    values.update({k: str(v) for k, v in fields.items()})
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), "À compléter")), text)
