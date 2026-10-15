"""m1utils — boîte à outils partagée des notebooks du Module 1 (IA vocale & langues locales).

Objectif : garder les notebooks courts et lisibles. Toute logique réutilisée dans ≥ 2 notebooks
vit ici, documentée et testée (``tests/``). Les fonctions sont volontairement simples et
pédagogiques : leur code source peut être montré aux participants.

Modules :
    audio       chargement, sauvegarde, écoute, rééchantillonnage           (ticket M1-G04)
    viz         formes d'onde, spectrogrammes, F0, matrices d'émission CTC  (ticket M1-G04)
    features    STFT, log-Mel, MFCC, F0, formants                           (ticket M1-G04)
    preprocess  normalisation, VAD, segmentation, débruitage, SNR           (ticket M1-G05)
    text        normalisation orthographique Fongbe / Wolof                  (ticket M1-G06)
    asr         chargement de modèles ASR/TTS, décodage CTC                  (ticket M1-G07)
    metrics     WER / CER, rapports par énoncé                               (ticket M1-G07)
    corpus      manifest, contrôles qualité, split par locuteur, HF Dataset  (ticket M1-G08)
    data        téléchargement des jeux de données du cours                  (ticket M1-G09)
    archi       encodeurs auto-supervisés, cartes d'attention, sondage    (ticket M1-G17)
"""
from __future__ import annotations

import os
import random

__version__ = "0.1.0"
SAMPLE_RATE = 16_000  # fréquence d'échantillonnage de référence de tout le module


def setup_notebook(seed: int = 42, quiet: bool = False) -> dict:
    """Prépare un notebook : graines aléatoires, style des figures, détection du matériel.

    Returns:
        dict avec les clés ``device`` ("cuda" | "mps" | "cpu"), ``colab`` (bool), ``seed``.
    """
    import warnings

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    device = "cpu"
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.figsize": (10, 3.5), "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
    except ImportError:
        pass
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:  # sorties lisibles : pas de barres « Loading weights » ni d'avertissements internes
        import transformers

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except ImportError:
        pass
    colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    info = {"device": device, "colab": colab, "seed": seed}
    if not quiet:
        print(f"m1utils {__version__} · matériel : {device} · Colab : {colab} · graine : {seed}")
        if device == "cpu":
            print("⚠️  Pas de GPU détecté. Sur Colab : Exécution → Modifier le type d'exécution → GPU T4.")
    return info
