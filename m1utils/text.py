"""Normalisation orthographique Fongbe / Wolof (relecture linguistique : ticket M1-Q03).

Pourquoi : le WER/CER et le vocabulaire des modèles CTC sont **très sensibles** à la forme Unicode
(précomposé vs combinant), aux homoglyphes (ε grec vs ɛ latin), aux diacritiques tonals, à la casse,
à la ponctuation et aux conventions d'annotation (``%hum``, ``*``…). Une même phrase peut avoir
plusieurs encodages visuellement identiques.

Constats sur données réelles (SimbaBench, tests ALFFA/Kallaama) qui motivent ce module :
- les transcriptions fon d'ALFFA utilisent « ε » GREC (U+03B5) au lieu de « ɛ » LATIN (U+025B) ;
- les tons n'y sont pas notés, alors que MMS (fon) produit des tons (ex. « yǎ ») ;
- le corpus wolof Kallaama (parole spontanée) note les hésitations ``%hum``/``%e``, l'inintelligible ``*``,
  les mots tronqués ``bo()`` et contient des mots français (code-switching).

Décisions de normalisation du cours (voir ``module1/commun/orthographes_fon_wol.md``) :
NFC → homoglyphes unifiés → apostrophes unifiées → [marqueurs d'annotation retirés] → minuscules →
[tons retirés (fon)] → [ponctuation, apostrophes et traits d'union remplacés par une espace] → espaces réduits.
"""
from __future__ import annotations

import re
import unicodedata

# Codes de langue utilisés dans tout le module (ISO 639-3, comme MMS)
LANGS = {"fon": "Fongbe", "wol": "Wolof", "fra": "Français"}

# Diacritiques combinants tonals (Unicode) : aigu (haut), grave (bas), circonflexe (descendant),
# caron (montant), macron (moyen, rare en fon)
TONE_MARKS = {"́", "̀", "̂", "̌", "̄"}

# Alphabets de référence (lettres de base, minuscules, NFC).
# Fongbe : alphabet des langues nationales du Bénin (digrammes gb, kp, xw, hw, ny écrits avec ces lettres ;
# nasalité notée par « n » après la voyelle). Source : alphabet des langues nationales du Bénin (CENALA)
# [À VÉRIFIER : référence exacte et lettres rares (q, r en emprunts)].
FON_ALPHABET: set[str] = set("abcdɖeɛfghijklmnŋoɔprstuvwxyz")
# Wolof : orthographe officielle du Sénégal (décret relatif à l'orthographe et à la séparation des mots en wolof)
# [À VÉRIFIER : numéro du décret]. Voyelles : a à e é ë i o ó u ; consonnes sans h, v, z (emprunts).
WOL_ALPHABET: set[str] = set("aàbcdeéëfgijklmnñŋoópqrstuwxy")
FRA_ALPHABET: set[str] = set("aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz")
ALPHABETS = {"fon": FON_ALPHABET, "wol": WOL_ALPHABET, "fra": FRA_ALPHABET}

# Homoglyphes : caractères d'autres écritures qui « ressemblent » aux lettres africaines de référence
HOMOGLYPHS = {
    "ε": "ɛ",  # ε grec → ɛ latin
    "Ε": "Ɛ",  # Ε grec majuscule → Ɛ latin
    "đ": "ɖ",  # đ (d barré, vietnamien/croate) → ɖ
    "Đ": "Ɖ",  # Đ → Ɖ
    "έ": "ɛ́",  # έ grec → ɛ́
    "ὲ": "ɛ̀",  # ὲ grec → ɛ̀
    "ɔɔ": "ɔɔ",  # (identité, rappel : ɔ est déjà latin)
}
APOSTROPHES = {"’": "'", "‘": "'", "ʼ": "'", "ʻ": "'", "`": "'", "´": "'"}

# Conventions d'annotation (Kallaama et corpus de parole spontanée)
_ANNOTATION_RE = re.compile(r"(?:^|\s)(?:%\w+|\*+|\S*\(\))(?=\s|$)")


def nfc(text: str) -> str:
    """Forme Unicode composée (NFC)."""
    return unicodedata.normalize("NFC", text)


def nfd(text: str) -> str:
    """Forme Unicode décomposée (NFD) : lettres de base + diacritiques combinants."""
    return unicodedata.normalize("NFD", text)


def fix_homoglyphs(text: str) -> str:
    """Remplace les homoglyphes (ε grec, đ…) par les lettres latines de référence."""
    for bad, good in HOMOGLYPHS.items():
        text = text.replace(bad, good)
    return text


def strip_tones(text: str, lang: str = "fon") -> str:
    """Retire les diacritiques tonals sans toucher aux lettres spéciales (ɛ, ɔ, ɖ, ŋ, ë…).

    En wolof, les accents notent la **qualité vocalique** (é, ë, ó, à) et non des tons : rien n'est retiré.
    """
    if lang != "fon":
        return nfc(text)
    return nfc("".join(ch for ch in nfd(text) if ch not in TONE_MARKS))


def strip_annotations(text: str) -> str:
    """Retire les marqueurs d'annotation : hésitations ``%hum``/``%e``, inintelligible ``*``, mots tronqués ``bo()``."""
    prev = None
    while prev != text:
        prev, text = text, _ANNOTATION_RE.sub(" ", text)
    return text


def normalize_text(text: str, lang: str = "fon", lowercase: bool = True, keep_tones: bool = True,
                   strip_punct: bool = True, unify_apostrophes: bool = True, remove_annotations: bool = True) -> str:
    """Normalisation de référence pour l'évaluation ASR et la préparation de corpus.

    Étapes : NFC → homoglyphes → apostrophes unifiées → [marqueurs d'annotation retirés] → minuscules →
    [retrait des tons] → [ponctuation remplacée par une espace] → espaces multiples réduits → NFC.

    Ponctuation : apostrophes et traits d'union sont remplacés par une espace (« l'impôt » → « l impôt »,
    « xam-xam » → « xam xam ») : la règle est la même pour la référence et l'hypothèse, donc l'évaluation
    reste juste, et on évite les variantes « l' impôt » / « l'impôt ».
    """
    text = fix_homoglyphs(nfc(text))
    if unify_apostrophes:
        for bad, good in APOSTROPHES.items():
            text = text.replace(bad, good)
    if remove_annotations:
        text = strip_annotations(text)
    if lowercase:
        text = text.lower()
    if not keep_tones:
        text = strip_tones(text, lang)
    if strip_punct:
        # on garde lettres (et leurs diacritiques combinants), chiffres et espaces
        text = "".join(ch if (unicodedata.category(ch)[0] in "LMN" or ch.isspace()) else " " for ch in nfd(text))
    text = re.sub(r"\s+", " ", text).strip()
    return nfc(text)


def char_inventory(texts: list[str], form: str = "NFC", lang: str | None = None):
    """Inventaire des caractères : colonnes char, codepoint (U+XXXX), name (unicodedata), category, count,
    in_alphabet (si ``lang`` est donné). Trié par fréquence décroissante."""
    import collections

    import pandas as pd

    counts = collections.Counter(unicodedata.normalize(form, "".join(texts)))
    alphabet = ALPHABETS.get(lang or "", None)
    rows = []
    for ch, n in counts.most_common():
        rows.append({
            "char": ch if not unicodedata.category(ch).startswith("M") else f"◌{ch}",
            "codepoint": f"U+{ord(ch):04X}",
            "name": unicodedata.name(ch, "?"),
            "category": unicodedata.category(ch),
            "count": n,
            "in_alphabet": None if alphabet is None else (ch.isspace() or _letter_ok(ch, lang)),
        })
    return pd.DataFrame(rows)


def _letter_ok(ch: str, lang: str) -> bool:
    alphabet = ALPHABETS[lang]
    if ch.lower() in alphabet:
        return True
    decomposed = nfd(ch.lower())
    base, marks = decomposed[0], set(decomposed[1:])
    if unicodedata.category(ch).startswith("M"):  # diacritique isolé (forme NFD)
        return lang == "fon" and ch in TONE_MARKS
    return lang == "fon" and base in alphabet and marks <= TONE_MARKS


def out_of_alphabet(text: str, lang: str) -> set[str]:
    """Caractères (hors espace/ponctuation) absents de l'alphabet de référence de ``lang``."""
    return {ch for ch in nfc(text) if unicodedata.category(ch)[0] in "LM" and not _letter_ok(ch, lang)}
