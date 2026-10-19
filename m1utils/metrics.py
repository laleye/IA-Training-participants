"""Métriques d'évaluation ASR : WER, CER, détail des opérations, tableaux par énoncé et par modèle.

Rappel : WER = (S + D + I) / N, où N est le nombre de mots de la **référence**. Le WER peut dépasser 100 %
(beaucoup d'insertions, typique des hallucinations). Sur un corpus, on agrège les **opérations** (somme des
erreurs / somme des N) — et non la moyenne des WER par énoncé, qui sur-pondère les énoncés courts.
"""
from __future__ import annotations

ERROR_CATEGORIES = {
    "ton_diacritique": "Ton ou diacritique erroné/absent (ɛ/e, ó/o, ǎ/a…)",
    "segmentation": "Découpage en mots différent (mots collés ou coupés)",
    "confusion_phonetique": "Son proche mal reconnu (consonne/voyelle voisine, longueur)",
    "mot_francais": "Mot ou expression française (code-switching) mal transcrit",
    "hallucination_langue": "Texte inventé, répété, ou dans une autre langue",
    "omission_troncature": "Passage manquant (début/fin coupés, parole faible)",
}


def _jiwer():
    import jiwer

    return jiwer


def _as_list(x) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def _safe(hyps: list[str]) -> list[str]:
    # jiwer refuse les chaînes vides côté hypothèse : on les remplace par un mot impossible (= suppression totale)
    return [h if h.strip() else "∅" for h in hyps]


def wer(reference: str | list[str], hypothesis: str | list[str]) -> float:
    """Word Error Rate = (S + D + I) / N (jiwer). Accepte un énoncé ou une liste (agrégé sur le corpus)."""
    return float(_jiwer().wer(_as_list(reference), _safe(_as_list(hypothesis))))


def cer(reference: str | list[str], hypothesis: str | list[str]) -> float:
    """Character Error Rate (mêmes conventions que ``wer``)."""
    return float(_jiwer().cer(_as_list(reference), _safe(_as_list(hypothesis))))


def edit_ops(reference: str, hypothesis: str, unit: str = "word") -> dict:
    """Détail des opérations : {'S', 'D', 'I', 'N', 'H' (correct), 'alignment' (texte lisible)}."""
    jw = _jiwer()
    out = jw.process_words(reference, _safe([hypothesis])[0]) if unit == "word" else \
        jw.process_characters(reference, _safe([hypothesis])[0])
    n = out.hits + out.substitutions + out.deletions
    return {"S": out.substitutions, "D": out.deletions, "I": out.insertions, "N": n, "H": out.hits,
            "alignment": jw.visualize_alignment(out, show_measures=False)}


def evaluate(refs: list[str], hyps: list[str], lang: str = "fon", normalize: bool = True,
             keep_tones: bool = True, ids: list[str] | None = None):
    """Tableau par énoncé (id, ref, hyp, wer, cer, S, D, I, N) + ligne « CORPUS » agrégée.

    Applique ``m1utils.text.normalize_text`` aux deux côtés si ``normalize``.
    """
    import pandas as pd

    from .text import normalize_text

    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} références pour {len(hyps)} hypothèses.")
    prep = (lambda t: normalize_text(t, lang, keep_tones=keep_tones)) if normalize else (lambda t: t.strip())
    rows = []
    for k, (r, h) in enumerate(zip(refs, hyps)):
        r_, h_ = prep(r), prep(h)
        if not r_:
            continue  # référence vide (ex. uniquement des hésitations) : non évaluable
        ops = edit_ops(r_, h_)
        rows.append({"id": ids[k] if ids else k, "ref": r_, "hyp": h_, "wer": wer(r_, h_), "cer": cer(r_, h_),
                     "S": ops["S"], "D": ops["D"], "I": ops["I"], "N": ops["N"]})
    df = pd.DataFrame(rows)
    corpus = {"id": "CORPUS", "ref": "", "hyp": "",
              "wer": (df.S.sum() + df.D.sum() + df.I.sum()) / max(df.N.sum(), 1),
              "cer": cer(df.ref.tolist(), df.hyp.tolist()),
              "S": df.S.sum(), "D": df.D.sum(), "I": df.I.sum(), "N": df.N.sum()}
    return pd.concat([df, pd.DataFrame([corpus])], ignore_index=True)


def compare_models(results: dict[str, list[str]], refs: list[str], lang: str):
    """Tableau comparatif : une ligne par modèle (WER/CER bruts, normalisés, normalisés sans tons)."""
    import pandas as pd

    rows = []
    for name, hyps in results.items():
        raw = evaluate(refs, hyps, lang, normalize=False).iloc[-1]
        norm = evaluate(refs, hyps, lang, normalize=True).iloc[-1]
        notone = evaluate(refs, hyps, lang, normalize=True, keep_tones=False).iloc[-1]
        rows.append({"modèle": name, "WER brut": raw.wer, "CER brut": raw.cer, "WER norm.": norm.wer,
                     "CER norm.": norm.cer, "WER sans tons": notone.wer, "CER sans tons": notone.cer})
    return pd.DataFrame(rows).set_index("modèle").round(3)


_FRENCH_HINTS = {"de", "la", "le", "les", "des", "est", "et", "que", "qui", "pour", "par", "avec", "projet", "parce",
                 "comme", "donc", "toujours", "un", "une", "mais", "ou", "en"}


def suggest_error_category(reference: str, hypothesis: str, lang: str = "fon") -> str:
    """**Suggère** une catégorie d'erreur (taxonomie ``ERROR_CATEGORIES``) pour un couple (référence, hypothèse).

    Heuristique pédagogique : elle sert de point de départ, l'analyste humain valide ou corrige.
    """
    import unicodedata

    from .text import normalize_text, strip_tones

    ref, hyp = normalize_text(reference, lang), normalize_text(hypothesis, lang)
    if ref == hyp:
        return "aucune"
    if not hyp or len(hyp.replace(" ", "")) < 0.5 * len(ref.replace(" ", "")):
        return "omission_troncature"
    latin = sum(1 for c in hyp if unicodedata.name(c, "").startswith("LATIN") or c == " ") / max(len(hyp), 1)
    mots = hyp.split()
    repetitif = len(mots) >= 6 and len(set(mots)) / len(mots) < 0.4
    if latin < 0.8 or repetitif or len(hyp) > 2 * len(ref) + 10:
        return "hallucination_langue"
    def sans_diacritiques(t):
        return "".join(c for c in unicodedata.normalize("NFD", strip_tones(t, "fon")) if not unicodedata.combining(c))
    if sans_diacritiques(ref) == sans_diacritiques(hyp):
        return "ton_diacritique"
    if ref.replace(" ", "") == hyp.replace(" ", ""):
        return "segmentation"
    if set(reference.lower().split()) & _FRENCH_HINTS:
        return "mot_francais"
    return "confusion_phonetique"
