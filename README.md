# Mesure du risque de réidentification dans des bases médicales anonymisées

> **Projet Fil Rouge — M2 MIAS 2025–2026**
> Aurélien Loison · Hugo Kazzi · Édouard Lansiaux
> Encadré par Pr E. Chazard (PU-PH) et Pr S. Hammadi (CRIStAL)

## Structure du dépôt

```
reidentification_risk/
├── run_all.py                  # Point d'entrée principal
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration : QI, paliers, chemins, seuils
│   ├── data_loader.py          # Chargement des données (KB + fichiers transformés)
│   ├── individualization.py    # 7 méthodes d'attaque par individualisation
│   ├── inference.py            # 3 méthodes d'inférence (A/B/C) × 2 codes
│   └── visualization.py       # Toutes les fonctions de tracé
├── projet_donnees/             # ← Placer les données ici
│   ├── connaissances_externes.txt
│   ├── out_direct_0.txt
│   ├── out_direct_5.txt
│   ├── ...
│   ├── out_direct_100.txt
│   ├── out_sample_0.txt
│   ├── ...
│   └── out_sample_100.txt
└── outputs/                    # Résultats générés automatiquement
    ├── figures/                # PNG des graphiques
    └── tables/                 # CSV des indicateurs
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancer toutes les expériences

```bash
python run_all.py
```

**Durée estimée** : 30–60 min (1000 patients × 21 paliers × 2 modes × 3 niveaux).

### Mode rapide (test)

```bash
python run_all.py --fast
```

6 paliers seulement, 200 patients → ~5 min.

### Individualisation seule

```bash
python run_all.py --only indiv
```

### Inférence seule

```bash
python run_all.py --only inference
```

## Données attendues

Les fichiers doivent être dans `projet_donnees/` au format TSV (tab-separated).

| Fichier | Description |
|---------|-------------|
| `connaissances_externes.txt` | Base de connaissances (vérité terrain avec tous les QI) |
| `out_direct_XX.txt` | Données directes avec XX % des cellules mélangées |
| `out_sample_XX.txt` | Données échantillonnées avec remise + XX % de mélange |

XX ∈ {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100}

### Colonnes requises

| Catégorie | Colonnes |
|-----------|----------|
| ID | `id_sejour` |
| Démographie | `age`, `age5`, `age10`, `sexe` |
| Séjour | `entree_mode`, `sortie_mode`, `specialite`, `chirurgie` |
| Dates | `entree_date_y`, `entree_date_ym`, `entree_date_ymd`, `sortie_date_y`, `sortie_date_ym`, `sortie_date_ymd` |
| Comorbidités | `diabete`, `insuffisance_renale`, `demence` |
| Sensibles | `liste_diag` (codes CIM-10, séparés par `;`), `liste_acte` (codes CCAM, séparés par `;`) |

## Méthodes implémentées

### Individualisation (7 méthodes)

| # | Méthode | Fonction | Description |
|---|---------|----------|-------------|
| 1 | Match exact 1/N | `match_exact_1_over_n()` | Probabilité de réidentification par correspondance parfaite |
| 2 | Match strict N=1 | `match_exact_strict()` | Réidentification uniquement si un seul suspect ET correct |
| 3 | Matrice de confiance | `confidence_matrix()` | 4 quadrants : succès critique / leurre / noyé / échec |
| 4 | Fiabilité hacker | `hacker_accuracy()` | Précision sous 3 règles (unicité, groupe restreint, incertitude) |
| 5 | Score rareté | `weighted_rarity_score()` | Pondération par information propre + gap top1/top2 |
| 6 | Monte-Carlo LOO | `monte_carlo_stability()` | Leave-one-out sur les QI avec vote par stabilité |
| 7 | Score net | `risk_score_net()` | Calibration : +confiance si juste, −confiance si faux |

### Inférence (3 méthodes × 2 types de codes)

| Méthode | Scoring | Intuition |
|---------|---------|-----------|
| A — Vote majoritaire | P(c\|CE) | Code le plus fréquent dans la classe d'équivalence |
| B — Lift bayésien | P(c\|CE) / P(c\|global) | Pénalise les codes banals (esprit t-closeness) |
| C — Rareté pondérée | P(c\|CE) × (1 − P(c\|global)) | Favorise les codes rares localement dominants |

Appliquées sur :
- **CIM-10** : diagnostics tronqués à 3 caractères
- **CCAM** : actes tronqués à 4 caractères

### Corrélation

Approche bibliographique (pas de code — une seule base disponible).
Voir le rapport §6 pour la discussion sur Join Potential, Jensen-Shannon, CCA, et linkage simulé.

## Sorties

### Tables (CSV)

| Fichier | Contenu |
|---------|---------|
| `indiv_1_match_exact_1N.csv` | Taux 1/N par palier × mode × niveau |
| `indiv_2_match_strict.csv` | Taux strict par palier × mode × niveau |
| `indiv_4_hacker_accuracy.csv` | Fiabilité M1/M2/M3 |
| `indiv_5_rarity_score.csv` | Accuracy + confiance win/fail |
| `indiv_6_monte_carlo.csv` | Données individuelles Monte-Carlo |
| `indiv_7_risk_score_net.csv` | Score net [-100, 100] |
| `inference_individuel.csv` | Données individuelles d'inférence |
| `inference_indicateurs.csv` | Indicateurs agrégés d'inférence |

### Figures (PNG)

| Préfixe | Contenu |
|---------|---------|
| `A1_*` | Match exact 1/N |
| `A2_*` | Match strict |
| `A3_*` | Matrice de confiance (barres empilées) |
| `A8_*` | Monte-Carlo + rareté |
| `B1_*` | Taux d'inférence réel (grille 2×3) |
| `B2_*` | Certitude & risque d'inférence |
| `B4_*` | Indice de désorientation Δ |
| `B5_*` | Matrice de risque inférence (4 quadrants) |
| `B7_*` | Distinction vrai/faux (capacité de filtrage) |

## Niveaux de connaissance attaquant

| Niveau | QI | Granularité |
|--------|----|-------------|
| N1 (Faible) | 6 | Âge par tranche de 10, année seule, mode entrée/sortie |
| N2 (Moyen) | 8 | Âge par tranche de 5, année+mois, + spécialité/chirurgie |
| N3 (Élevé) | 11 | Âge exact, date complète, + diabète/IR/démence |

## Licence

Projet académique — UFR3S / École Centrale de Lille / Université de Lille.
