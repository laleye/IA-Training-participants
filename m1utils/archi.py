"""Outils des ateliers d'architecture (séances 3·1 et 3·2 : S3-NB03, S3-NB04).

Les composants à comprendre (attention, bloc Transformer, module de convolution, perte contrastive)
sont codés **dans les notebooks**. Ce module ne contient que la logistique : charger un encodeur
auto-supervisé du registre, extraire les sorties de chaque couche, afficher une carte d'attention,
étiqueter des trames et calculer un champ réceptif.
"""
from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .asr import _as_array, _device, _resolve


def attention_reference(Q, K, V, mask=None):
    """Attention par produit scalaire de référence (secours des ateliers si l'exercice n'est pas terminé).

    ``Q`` (…, T_q, d_k), ``K`` (…, T_k, d_k), ``V`` (…, T_k, d_v) : tenseurs PyTorch. ``mask`` : booléen
    (…, T_q, T_k), ``True`` = position AUTORISÉE. Renvoie (sortie, poids).
    """
    import math

    import torch

    scores = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    poids = torch.softmax(scores, dim=-1)
    return poids @ V, poids


def load_encoder(name: str, device: str | None = None):
    """Charge un encodeur auto-supervisé du registre (tâche ``ssl``) en mode évaluation, en float32.

    Returns:
        (model, feature_extractor). Pour ``wav2vec2-base``, le modèle est un ``Wav2Vec2ForPreTraining``
        (quantificateur inclus) ; sinon un ``AutoModel`` (HubertModel, Wav2Vec2Model…).
    """
    from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2ForPreTraining

    key, info = _resolve(name)
    if info["task"] != "ssl":
        raise ValueError(f"{key} n'est pas un encodeur auto-supervisé (tâche : {info['task']}). "
                         "Pour un modèle ASR, utilisez load_asr(...).model.")
    cls = Wav2Vec2ForPreTraining if info.get("pretraining") else AutoModel
    model = cls.from_pretrained(info["id"]).to(_device(device)).eval()
    return model, AutoFeatureExtractor.from_pretrained(info["id"])


def hidden_states(model, feature_extractor, audio) -> list[np.ndarray]:
    """Sorties de toutes les couches d'un encodeur pour un énoncé : liste de L+1 tableaux (trames × dim).

    L'élément 0 est l'entrée de la première couche Transformer (après l'encodeur convolutif et
    l'encodage positionnel), l'élément i la sortie de la couche i.
    """
    import torch

    y = _as_array(audio)
    p = next(model.parameters())
    x = feature_extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(p.device, p.dtype)
    with torch.no_grad():
        out = model(x, output_hidden_states=True)
    return [h[0].float().cpu().numpy() for h in out.hidden_states]


def conv_receptive_field(kernels: list[int], strides: list[int]) -> tuple[int, int]:
    """Champ réceptif et pas (en échantillons) d'une pile de convolutions 1-D.

    Récurrence : r ← r + (k − 1) × j, puis j ← j × s (j = pas cumulé). Exemple, encodeur de wav2vec 2.0 :
    noyaux (10, 3, 3, 3, 3, 2, 2), pas (5, 2, 2, 2, 2, 2, 2) → (400, 320), soit 25 ms et 20 ms à 16 kHz.
    """
    r, j = 1, 1
    for k, s in zip(kernels, strides):
        r += (k - 1) * j
        j *= s
    return r, j


def ctc_frame_labels(log_probs: np.ndarray, blank_id: int, ignore: tuple[int, ...] = ()) -> np.ndarray:
    """Étiquette chaque trame par le symbole CTC le plus probable ; −1 pour le blank et les symboles ignorés.

    Sert de « pseudo-annotation phonétique » pour sonder les couches d'un encodeur (S3-NB04).
    """
    ids = np.asarray(log_probs).argmax(-1)
    lab = ids.copy()
    lab[(ids == blank_id) | np.isin(ids, list(ignore))] = -1
    return lab


def stack_frames(features: list[list[np.ndarray]], labels: list[np.ndarray], utterances, layer: int):
    """Empile les trames étiquetées (label ≥ 0) des énoncés choisis, pour une couche : (X, y)."""
    xs, ys = [], []
    for i in utterances:
        f, lab = features[i][layer], labels[i]
        n = min(len(f), len(lab))
        keep = lab[:n] >= 0
        xs.append(f[:n][keep])
        ys.append(lab[:n][keep])
    return np.concatenate(xs), np.concatenate(ys)


def plot_attention(weights: np.ndarray, frame_s: float | None = None, ax=None, title: str | None = None,
                   query_labels: list[str] | None = None, key_labels: list[str] | None = None, cmap: str = "magma"):
    """Carte d'attention (requêtes en lignes, clés en colonnes). ``frame_s`` : axes en secondes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    w = np.asarray(weights)
    extent = None
    if frame_s:
        extent = (0, w.shape[1] * frame_s, w.shape[0] * frame_s, 0)
    img = ax.imshow(w, aspect="auto", cmap=cmap, extent=extent, interpolation="nearest")
    if query_labels is not None:
        ax.set_yticks(range(len(query_labels)), query_labels, fontsize=9)
    if key_labels is not None:
        ax.set_xticks(range(len(key_labels)), key_labels, fontsize=9)
    unite = " (s)" if frame_s else ""
    ax.set(xlabel=f"clé : position regardée{unite}", ylabel=f"requête : position qui regarde{unite}",
           title=title or "Poids d'attention")
    ax.grid(False)
    plt.colorbar(img, ax=ax, fraction=0.046)
    return ax
