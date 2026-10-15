"""Chargement et utilisation de modèles ASR / TTS / LID pré-entraînés.

Le registre ``MODEL_REGISTRY`` est la **seule source** des identifiants de modèles utilisés dans le cours.
Chaque entrée a été vérifiée sur le Hub (existence, licence, architecture) — voir
``module1/commun/registre_modeles.md`` pour les usages, mesures et limites.
"""
from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE

MODEL_REGISTRY: dict[str, dict] = {
    # --- ASR généralistes (démonstration des limites sur fon/wol) ------------------------------------
    "whisper-small": {"id": "openai/whisper-small", "family": "whisper", "task": "asr", "langs": [],
                      "license": "apache-2.0", "params_m": 242,
                      "notes": "Whisper multilingue (99 langues) — ni le fon ni le wolof : hallucinations attendues."},
    "whisper-turbo": {"id": "openai/whisper-large-v3-turbo", "family": "whisper", "task": "asr", "langs": [],
                      "license": "mit", "params_m": 809, "notes": "Whisper large-v3 à décodeur réduit (4 couches)."},
    # --- Massivement multilingue ------------------------------------------------------------------------
    "mms-1b-all": {"id": "facebook/mms-1b-all", "family": "ctc", "task": "asr", "langs": ["fon", "wol"],
                   "adapter": "lang", "license": "cc-by-nc-4.0", "params_m": 965,
                   "notes": "MMS (1 162 langues), un adaptateur par langue ; données majoritairement religieuses lues."},
    # --- Spécialisés Afrique (Simba, EMNLP 2025) ----------------------------------------------------------
    "simba-h": {"id": "UBC-NLP/Simba-H", "family": "ctc", "task": "asr", "langs": ["fon", "wol"],
                "license": "cc-by-4.0", "params_m": 94,
                "notes": "AfriHuBERT fine-tuné CTC sur SimbaBench, 43 langues africaines. Léger et rapide."},
    "simba-m": {"id": "UBC-NLP/Simba-M", "family": "ctc", "task": "asr", "langs": ["fon", "wol"],
                "adapter": "multilingual_african", "license": "cc-by-4.0", "params_m": 965,
                "notes": "MMS-1b-all fine-tuné sur SimbaBench (adaptateur 'multilingual_african')."},
    "simba-s": {"id": "UBC-NLP/Simba-S", "family": "seamless", "task": "asr", "langs": ["fon", "wol"],
                "license": "cc-by-4.0", "params_m": 1502,
                "notes": "SeamlessM4T-v2 fine-tuné ; meilleur modèle Simba en moyenne. ≈ 3 Go en fp16."},
    "simba-w": {"id": "UBC-NLP/Simba-W", "family": "whisper", "task": "asr", "langs": ["fon", "wol"],
                "license": "cc-by-4.0", "params_m": 1543,
                "notes": "Whisper-large-v3 fine-tuné sur SimbaBench. Lourd : GPU ≥ 12 Go recommandé."},
    # --- Communautaires (fine-tuning sur une langue) ------------------------------------------------------
    "fon-xlsr": {"id": "chrisjay/fonxlsr", "family": "ctc", "task": "asr", "langs": ["fon"],
                 "license": "apache-2.0", "params_m": 315,
                 "notes": "XLSR-53 fine-tuné sur le Fon Dataset (laleye/pyFongbe) : même domaine qu'ALFFA."},
    "wolof-hubert": {"id": "soynade-research/Wolof-HuBERT-CTC", "family": "ctc", "task": "asr", "langs": ["wol"],
                     "license": "agpl-3.0", "params_m": 94,
                     "notes": "HuBERT CTC wolof ; données d'entraînement non documentées (WER de validation annoncé 0,36)."},
    # --- Encodeurs auto-supervisés (pas d'ASR direct ; ateliers d'architecture S3-NB04) --------------------
    "wav2vec2-base": {"id": "facebook/wav2vec2-base", "family": "ssl", "task": "ssl", "langs": [],
                      "license": "apache-2.0", "params_m": 95, "pretraining": True,
                      "notes": "wav2vec 2.0 BASE pré-entraîné sur LibriSpeech (anglais lu), AVEC son quantificateur : "
                               "permet de rejouer la tâche contrastive."},
    "afrihubert": {"id": "ajesujoba/AfriHuBERT", "family": "ssl", "task": "ssl", "langs": [],
                   "license": "cc", "params_m": 95, "notes": "HuBERT pré-entraîné sur des langues africaines (Alabi et al.)."},
    # --- Synthèse vocale --------------------------------------------------------------------------------
    "mms-tts-fon": {"id": "facebook/mms-tts-fon", "family": "vits", "task": "tts", "langs": ["fon"],
                    "license": "cc-by-nc-4.0", "params_m": 36, "notes": "VITS de bout en bout ; vocabulaire avec tons."},
    "speecht5-wol": {"id": "bilalfaye/speecht5_tts-wolof-v0.2", "family": "speecht5", "task": "tts",
                     "langs": ["wol", "fra"], "vocoder": "microsoft/speecht5_hifigan", "license": "mit",
                     "params_m": 144, "notes": "SpeechT5 (texte → Mel) + vocodeur HiFi-GAN (Mel → onde)."},
    # --- Identification de langue -----------------------------------------------------------------------
    "simba-slid": {"id": "UBC-NLP/Simba-SLID-49", "family": "lid", "task": "lid", "langs": ["fon", "wol"],
                   "license": "cc-by-4.0", "params_m": 95, "notes": "49 langues africaines dont fon et wol."},
    "mms-lid-126": {"id": "facebook/mms-lid-126", "family": "lid", "task": "lid", "langs": ["wol", "fra"],
                    "license": "cc-by-nc-4.0", "params_m": 966, "notes": "126 langues ; le fon n'en fait PAS partie."},
}

_SPECIAL_RE = re.compile(r"^[<\[].*[>\]]$")


@dataclass
class Transcription:
    text: str
    model: str
    lang: str | None
    duration_s: float
    latency_s: float
    chunks: list[dict] | None = None  # horodatages (mots/caractères) si disponibles

    @property
    def rtf(self) -> float:
        """Facteur temps réel : temps de calcul / durée audio (< 1 = plus rapide que le temps réel)."""
        return self.latency_s / self.duration_s if self.duration_s else float("nan")


def _resolve(name: str) -> tuple[str, dict]:
    if name in MODEL_REGISTRY:
        return name, MODEL_REGISTRY[name]
    for key, info in MODEL_REGISTRY.items():
        if info["id"] == name:
            return key, info
    raise KeyError(f"Modèle {name!r} absent du registre du cours. Clés disponibles : {', '.join(MODEL_REGISTRY)}")


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_array(item, sr: int = SAMPLE_RATE) -> np.ndarray:
    from .audio import load_audio, resample

    if isinstance(item, (str, Path)):
        return load_audio(item, sr=sr)[0]
    if isinstance(item, dict):  # {"array", "sampling_rate"} ou {"bytes"}
        if "bytes" in item and item["bytes"] is not None:
            from .audio import decode_bytes

            return decode_bytes(item["bytes"], sr=sr)[0]
        y, isr = np.asarray(item["array"], dtype=np.float32), item.get("sampling_rate", sr)
        return resample(y, isr, sr) if isr != sr else y
    return np.asarray(item, dtype=np.float32)


class ASRModel:
    """Modèle ASR prêt à l'emploi : ``model(audio) -> Transcription``.

    ``audio`` : tableau NumPy 16 kHz, chemin de fichier ou dict ``{"array", "sampling_rate"}``.
    Les modèles CTC exposent aussi ``model.logits(audio)`` et ``model.vocab`` (atelier S3-NB01).
    """

    def __init__(self, name: str, lang: str | None = None, device: str | None = None, dtype: str = "auto",
                 max_new_tokens: int = 128):
        import torch

        self.key, self.info = _resolve(name)
        if self.info["task"] != "asr":
            raise ValueError(f"{self.key} n'est pas un modèle ASR (tâche : {self.info['task']}).")
        self.id, self.family, self.lang = self.info["id"], self.info["family"], lang
        self.device = _device(device)
        if dtype == "auto":
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, dtype)
        self.max_new_tokens = max_new_tokens
        if self.family == "ctc":
            from transformers import AutoModelForCTC, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.id)
            self.model = AutoModelForCTC.from_pretrained(self.id, dtype=self.dtype).to(self.device).eval()
            adapter = self.info.get("adapter")
            if adapter == "lang":
                if not lang:
                    raise ValueError(f"{self.key} nécessite une langue : load_asr('{self.key}', lang='fon').")
                self.processor.tokenizer.set_target_lang(lang)
                self.model.load_adapter(lang)
            elif adapter:
                self.model.load_adapter(adapter)
        else:  # whisper / seamless : pipeline Hugging Face (encodeur-décodeur)
            from transformers import pipeline

            self.pipe = pipeline("automatic-speech-recognition", model=self.id, dtype=self.dtype, device=self.device)
            self.model = self.pipe.model
            self.processor = None

    # -- CTC ---------------------------------------------------------------------------------------------
    @property
    def vocab(self) -> list[str]:
        """Vocabulaire ordonné (index → token) d'un modèle CTC."""
        if self.processor is None:
            raise AttributeError("Seuls les modèles CTC exposent un vocabulaire de sortie par trame.")
        v = self.processor.tokenizer.get_vocab()
        out = [""] * (max(v.values()) + 1)
        for tok, i in v.items():
            out[i] = tok
        # Le tokenizer peut déclarer plus de tokens que la couche de sortie du modèle (tokens « ajoutés »
        # sans sortie, cas de Simba-H : 430 vs 428). On s'aligne sur la taille réelle des logits.
        n_out = getattr(self.model.config, "vocab_size", len(out))
        return out[:n_out]

    @property
    def blank_id(self) -> int:
        return int(self.processor.tokenizer.pad_token_id)

    def logits(self, audio) -> np.ndarray:
        """Log-probabilités CTC (trames × vocabulaire), float32."""
        import torch

        y = _as_array(audio)
        x = self.processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device, self.dtype)
        with torch.no_grad():
            out = self.model(x).logits[0].float()
        return torch.log_softmax(out, dim=-1).cpu().numpy()

    # -- Transcription ----------------------------------------------------------------------------------
    def __call__(self, audio) -> Transcription:
        y = _as_array(audio)
        start = time.perf_counter()
        if self.family == "ctc":
            lp = self.logits(y)
            text = self.processor.decode(lp.argmax(-1))
        else:
            kwargs = {}
            if self.family == "whisper":
                kwargs["generate_kwargs"] = {"task": "transcribe", "max_new_tokens": self.max_new_tokens}
            out = self.pipe({"raw": y, "sampling_rate": SAMPLE_RATE}, **kwargs)
            text = out["text"]
        latency = time.perf_counter() - start
        return Transcription(text=text.strip(), model=self.key, lang=self.lang, duration_s=len(y) / SAMPLE_RATE,
                             latency_s=latency)

    def __repr__(self) -> str:
        return f"ASRModel({self.key!r}, id={self.id!r}, famille={self.family!r}, langue={self.lang!r}, device={self.device!r})"


def whisper_detect_language(model: ASRModel, audio, top_k: int = 3) -> list[tuple[str, float]]:
    """Langues que Whisper « croit » entendre : [(code, probabilité)]. Révèle pourquoi Whisper échoue sur fon/wol."""
    import torch

    if model.family != "whisper":
        raise ValueError("whisper_detect_language ne s'applique qu'aux modèles Whisper.")
    y = _as_array(audio)
    fe = model.pipe.feature_extractor
    feats = fe(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(model.device, model.dtype)
    with torch.no_grad():
        enc = model.model.get_encoder()(feats)
        dec_in = torch.tensor([[model.model.generation_config.decoder_start_token_id]], device=model.device)
        logits = model.model(encoder_outputs=enc, decoder_input_ids=dec_in).logits[0, -1].float()
    lang_ids = {i: t.strip("<|>") for t, i in model.model.generation_config.lang_to_id.items()}
    ids = torch.tensor(list(lang_ids))
    probs = torch.softmax(logits[ids], -1).cpu().numpy()
    order = np.argsort(-probs)[:top_k]
    return [(lang_ids[int(ids[i])], float(probs[i])) for i in order]


def load_asr(name: str, lang: str | None = None, device: str | None = None, dtype: str = "auto",
             max_new_tokens: int = 128) -> ASRModel:
    """Charge un modèle ASR du registre (clé ou id Hub). Pour MMS, active l'adaptateur ``lang``."""
    return ASRModel(name, lang=lang, device=device, dtype=dtype, max_new_tokens=max_new_tokens)


def transcribe_batch(model: ASRModel, items: list, batch_size: int = 8, show_progress: bool = True) -> list[Transcription]:
    """Transcrit une liste de signaux/chemins ; mesure la latence par énoncé.

    Pour les modèles encodeur-décodeur, les énoncés sont regroupés par lots (``batch_size``) ; la latence
    rapportée est alors la latence du lot répartie au prorata des durées.
    """
    from tqdm.auto import tqdm

    arrays = [_as_array(it) for it in items]
    if model.family == "ctc":
        it = tqdm(arrays, desc=model.key) if show_progress else arrays
        return [model(a) for a in it]
    results: list[Transcription] = []
    kwargs = {}
    if model.family == "whisper":
        kwargs["generate_kwargs"] = {"task": "transcribe", "max_new_tokens": model.max_new_tokens}
    rng = range(0, len(arrays), batch_size)
    for i in tqdm(rng, desc=model.key) if show_progress else rng:
        batch = arrays[i: i + batch_size]
        start = time.perf_counter()
        outs = model.pipe([{"raw": a, "sampling_rate": SAMPLE_RATE} for a in batch], batch_size=len(batch), **kwargs)
        total = time.perf_counter() - start
        durs = [len(a) / SAMPLE_RATE for a in batch]
        for o, d in zip(outs, durs):
            results.append(Transcription(o["text"].strip(), model.key, model.lang, d, total * d / sum(durs)))
    return results


def ctc_logits(name: str | ASRModel, y: np.ndarray, lang: str | None = None, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, list[str]]:
    """Log-probabilités CTC (trames × vocabulaire) et vocabulaire ordonné (index → token)."""
    from .audio import resample

    model = name if isinstance(name, ASRModel) else load_asr(name, lang=lang)
    if model.family != "ctc":
        raise ValueError(f"{model.key} n'est pas un modèle CTC.")
    if sr != SAMPLE_RATE:
        y = resample(y, sr, SAMPLE_RATE)
    return model.logits(y), model.vocab


def ctc_greedy_decode(ids: list[int] | np.ndarray, vocab: list[str], blank_id: int = 0,
                      word_delimiter: str = "|") -> str:
    """Décodage glouton CTC de référence : fusion des répétitions → retrait des blanks → mots.

    Les autres tokens spéciaux (``<s>``, ``[UNK]``…) sont ignorés.
    """
    ids = [int(i) for i in np.asarray(ids).ravel()]
    collapsed = [i for k, i in enumerate(ids) if k == 0 or i != ids[k - 1]]  # 1. fusion des répétitions
    tokens = [vocab[i] for i in collapsed if i != blank_id]                  # 2. retrait des blanks
    tokens = [t for t in tokens if not (_SPECIAL_RE.match(t) and t != word_delimiter)]
    text = "".join(" " if t == word_delimiter else t for t in tokens)        # 3. délimiteur → espace
    return re.sub(r"\s+", " ", text).strip()


def build_beam_decoder(vocab: list[str], blank_id: int, word_delimiter: str = "|", hotwords: list[str] | None = None):
    """Décodeur CTC par recherche en faisceau (pyctcdecode), sans modèle de langue.

    Renvoie une fonction ``decode(log_probs, beam_width=50) -> str``. ``hotwords`` : mots à favoriser
    (ex. noms propres, termes d'un domaine).
    """
    import logging

    logging.getLogger("pyctcdecode").setLevel(logging.ERROR)  # avertissements kenlm / alphabet non pertinents ici
    from pyctcdecode import build_ctcdecoder

    labels = []
    for i, t in enumerate(vocab):
        if i == blank_id:
            labels.append("")
        elif t == word_delimiter:
            labels.append(" ")
        elif _SPECIAL_RE.match(t) or len(t) > 1 or t in labels:
            labels.append(f"\uE000{i}")  # token spécial ou doublon : étiquette unique, retirée après décodage
        else:
            labels.append(t)
    decoder = build_ctcdecoder(labels)

    def decode(log_probs: np.ndarray, beam_width: int = 50) -> str:
        text = decoder.decode(log_probs, beam_width=beam_width, hotwords=hotwords or None)
        return re.sub(r"\s+", " ", re.sub("\uE000\\d+", "", text)).strip()

    return decode


def ctc_char_alignment(log_probs: np.ndarray, vocab: list[str], blank_id: int, frame_s: float = 0.02,
                       word_delimiter: str = "|") -> list[dict]:
    """Alignement glouton : pour chaque caractère émis, (char, début_s, fin_s, prob moyenne)."""
    ids = log_probs.argmax(-1)
    probs = np.exp(log_probs.max(-1))
    out: list[dict] = []
    k = 0
    while k < len(ids):
        j = k
        while j + 1 < len(ids) and ids[j + 1] == ids[k]:
            j += 1
        if ids[k] != blank_id:
            tok = vocab[ids[k]]
            out.append({"char": " " if tok == word_delimiter else tok, "start_s": round(k * frame_s, 3),
                        "end_s": round((j + 1) * frame_s, 3), "prob": float(probs[k: j + 1].mean())})
        k = j + 1
    return out


def words_from_alignment(chars: list[dict]) -> list[dict]:
    """Regroupe un alignement caractère par caractère en mots (texte, début_s, fin_s)."""
    words, cur = [], []
    for c in chars + [{"char": " "}]:
        if c["char"] == " ":
            if cur:
                words.append({"word": "".join(x["char"] for x in cur), "start_s": cur[0]["start_s"],
                              "end_s": cur[-1]["end_s"], "prob": float(np.mean([x["prob"] for x in cur]))})
            cur = []
        else:
            cur.append(c)
    return words


# ------------------------------------------------------------------------------------------------------- TTS
@dataclass
class TTSModel:
    key: str
    model: object
    tokenizer: object
    sampling_rate: int
    vocoder: object = None
    speaker_embedding: object = None
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def __call__(self, text: str, seed: int = 0) -> tuple[np.ndarray, int]:
        """Synthétise ``text`` → (signal float32, fréquence d'échantillonnage)."""
        import torch

        torch.manual_seed(seed)  # VITS est stochastique (durées) : graine pour la reproductibilité
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.vocoder is None:
                wav = self.model(**inputs).waveform[0]
            else:
                mel = self.model.generate_speech(inputs["input_ids"], self.speaker_embedding)
                wav = self.vocoder(mel)
        return wav.float().cpu().numpy(), self.sampling_rate

    def mel(self, text: str) -> np.ndarray:
        """(SpeechT5 uniquement) Spectrogramme Mel prédit par le modèle acoustique, avant le vocodeur."""
        import torch

        if self.vocoder is None:
            raise AttributeError("Ce modèle (VITS) génère directement la forme d'onde : pas de Mel intermédiaire.")
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.generate_speech(inputs["input_ids"], self.speaker_embedding).float().cpu().numpy()


def load_tts(name: str, device: str | None = None, speaker_index: int = 7306) -> TTSModel:
    """Charge un modèle TTS du registre ; renvoie ``tts(text) -> (signal, sr)``."""
    key, info = _resolve(name)
    dev = _device(device)
    if info["family"] == "vits":
        from transformers import AutoTokenizer, VitsModel

        model = VitsModel.from_pretrained(info["id"]).to(dev).eval()
        return TTSModel(key, model, AutoTokenizer.from_pretrained(info["id"]), model.config.sampling_rate, device=dev)
    if info["family"] == "speecht5":
        import torch
        from datasets import load_dataset
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

        proc = SpeechT5Processor.from_pretrained(info["id"])
        model = SpeechT5ForTextToSpeech.from_pretrained(info["id"]).to(dev).eval()
        vocoder = SpeechT5HifiGan.from_pretrained(info["vocoder"]).to(dev).eval()
        # x-vectors (empreintes de locuteurs) CMU Arctic — version Parquet (les scripts de chargement ne sont plus supportés)
        xv = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", revision="refs/convert/parquet")
        spk = torch.tensor(xv[speaker_index]["xvector"]).unsqueeze(0).to(dev)
        return TTSModel(key, model, proc, 16_000, vocoder=vocoder, speaker_embedding=spk, device=dev)
    raise ValueError(f"{key} n'est pas un modèle TTS.")


# ------------------------------------------------------------------------------------------------------- LID
_LID_CACHE: dict = {}


def identify_language(y: np.ndarray, sr: int = SAMPLE_RATE, top_k: int = 3, model: str = "simba-slid",
                      device: str | None = None) -> list[tuple[str, float]]:
    """Identification de langue : [(code de langue, probabilité)] triés par probabilité décroissante."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    from .audio import resample

    key, info = _resolve(model)
    dev = _device(device)
    if key not in _LID_CACHE:
        _LID_CACHE[key] = (AutoFeatureExtractor.from_pretrained(info["id"]),
                           AutoModelForAudioClassification.from_pretrained(info["id"]).to(dev).eval())
    fe, mdl = _LID_CACHE[key]
    if sr != SAMPLE_RATE:
        y = resample(y, sr, SAMPLE_RATE)
    x = fe(np.asarray(y, dtype=np.float32), sampling_rate=SAMPLE_RATE, return_tensors="pt").to(dev)
    with torch.no_grad():
        probs = torch.softmax(mdl(**x).logits[0].float(), -1).cpu().numpy()
    best = np.argsort(-probs)[:top_k]
    return [(mdl.config.id2label[int(i)], float(probs[i])) for i in best]


def unload(*models) -> None:
    """Libère la mémoire (GPU) occupée par des modèles : à appeler avant d'en charger un autre."""
    import torch

    for m in models:
        for attr in ("model", "pipe", "vocoder"):
            if hasattr(m, attr):
                try:
                    setattr(m, attr, None)
                except Exception:
                    pass
    _LID_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
