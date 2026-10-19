---
pretty_name: "{{pretty_name}}"
language:
  - {{lang}}
license: {{license}}
task_categories:
  - automatic-speech-recognition
size_categories:
  - {{size_category}}
tags:
  - speech
  - low-resource
  - ia-training-module1
---

# {{pretty_name}}

## 1. Résumé

{{description}}

| Indicateur | Valeur |
|---|---|
| Énoncés | {{n_utterances}} |
| Durée totale | {{total_min}} min |
| Durée moyenne | {{mean_duration_s}} s |
| Locuteurs | {{n_speakers}} |
| Taille du vocabulaire | {{vocabulary_size}} mots distincts |
| Découpage | {{splits}} |

## 2. Langues et variétés

- Langue (ISO 639-3) : `{{lang}}`
- Variété(s) / région(s) : {{dialect}}
- Orthographe : {{orthography}}

## 3. Structure des données

Champs : `id`, `audio` (WAV 16 kHz mono), `text`, `lang`, `speaker_id` (pseudonyme), `gender`, `age_range`,
`dialect`, `duration_s`, `sr`, `snr_db`, `recording_device`, `environment`, `consent_id`, `license`, `split`.
Le découpage train/dev/test est fait **par locuteur** (aucun locuteur dans deux splits).

## 4. Collecte

- Protocole : {{protocol}}
- Matériel : {{device}}
- Lieux et conditions : {{environment}}
- Période : {{period}}

## 5. Annotation et normalisation

- Transcripteurs : {{annotators}}
- Conventions : {{conventions}}
- Normalisation appliquée : NFC, homoglyphes unifiés (ε → ɛ), minuscules, ponctuation retirée (`m1utils.text.normalize_text`).
- Contrôles qualité : {{quality_checks}}

## 6. Locuteurs (statistiques agrégées uniquement)

- Répartition par genre (durée) : {{gender}}
- Répartition par tranche d'âge (durée) : {{age_range}}
- Aucune information individuelle identifiante n'est publiée.

## 7. Consentement et droits

- Forme du consentement : {{consent}}
- Licence : `{{license}}`
- Retrait : toute personne enregistrée peut demander le retrait de sa voix en écrivant à {{contact}} ; les données
  seront supprimées de la version suivante du jeu.

## 8. Biais et limites connus

{{limitations}}

## 9. Usages prévus et usages déconseillés

- Prévus : {{intended_uses}}
- Déconseillés : identification ou surveillance de personnes, clonage de voix, toute utilisation contraire au consentement donné.

## 10. Citation

{{citation}}

## 11. Contact

{{contact}}
