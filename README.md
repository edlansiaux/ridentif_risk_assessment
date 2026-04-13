# Measuring Re-identification Risk in Anonymized Medical Databases

## Repository Structure

```
reidentification_risk/
├── run_all.py                  # Main entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration: QIs, thresholds, paths, risk bands
│   ├── data_loader.py          # Data loading (KB + transformed files)
│   ├── individualization.py    # 7 individualization attack methods
│   ├── inference.py            # 3 inference methods (A/B/C) × 2 code types
│   └── visualization.py       # All plotting functions
├── project_data/               # ← Place data files here
│   ├── external_knowledge.txt
│   ├── out_direct_0.txt
│   ├── out_direct_5.txt
│   ├── ...
│   ├── out_direct_100.txt
│   ├── out_sample_0.txt
│   ├── ...
│   └── out_sample_100.txt
└── outputs/                    # Auto-generated results
    ├── figures/                # PNG charts
    └── tables/                 # CSV indicators
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments

```bash
python run_all.py
```

**Estimated duration**: 30–60 min (1000 patients × 21 thresholds × 2 modes × 3 levels).

### Fast mode (test)

```bash
python run_all.py --fast
```

6 thresholds only, 200 patients → ~5 min.

### Individualization only

```bash
python run_all.py --only indiv
```

### Inference only

```bash
python run_all.py --only inference
```

## Expected Data

Files must be placed in `project_data/` in TSV (tab-separated) format.

| File | Description |
|------|-------------|
| `external_knowledge.txt` | Knowledge base (ground truth with all QIs) |
| `out_direct_XX.txt` | Direct data with XX % of cells shuffled |
| `out_sample_XX.txt` | Sampled data with replacement + XX % shuffling |

XX ∈ {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100}

### Required Columns

| Category | Columns |
|----------|---------|
| ID | `id_sejour` |
| Demographics | `age`, `age5`, `age10`, `sexe` |
| Stay | `entree_mode`, `sortie_mode`, `specialite`, `chirurgie` |
| Dates | `entree_date_y`, `entree_date_ym`, `entree_date_ymd`, `sortie_date_y`, `sortie_date_ym`, `sortie_date_ymd` |
| Comorbidities | `diabete`, `insuffisance_renale`, `demence` |
| Sensitive | `liste_diag` (ICD-10 codes, `;`-separated), `liste_acte` (CCAM codes, `;`-separated) |

## Implemented Methods

### Individualization (7 methods)

| # | Method | Function | Description |
|---|--------|----------|-------------|
| 1 | Exact match 1/N | `match_exact_1_over_n()` | Re-identification probability via perfect match |
| 2 | Strict match N=1 | `match_exact_strict()` | Re-identification only if single suspect AND correct |
| 3 | Confidence matrix | `confidence_matrix()` | 4 quadrants: critical success / decoy / submerged / failure |
| 4 | Hacker accuracy | `hacker_accuracy()` | Precision under 3 rules (uniqueness, restricted group, uncertainty) |
| 5 | Rarity score | `weighted_rarity_score()` | Weighting by self-information + top1/top2 gap |
| 6 | Monte-Carlo LOO | `monte_carlo_stability()` | Leave-one-out over QIs with stability vote |
| 7 | Net score | `risk_score_net()` | Calibration: +confidence if correct, −confidence if wrong |

### Inference (3 methods × 2 code types)

| Method | Scoring | Intuition |
|--------|---------|-----------|
| A — Majority vote | P(c\|EC) | Most frequent code within the equivalence class |
| B — Bayesian lift | P(c\|EC) / P(c\|global) | Penalizes common codes (t-closeness spirit) |
| C — Weighted rarity | P(c\|EC) × (1 − P(c\|global)) | Favors locally dominant rare codes |

Applied to:
- **ICD-10**: diagnoses truncated to 3 characters
- **CCAM**: procedures truncated to 4 characters

### Correlation

Bibliographic approach (no code — only one database available).
See report §6 for discussion on Join Potential, Jensen-Shannon, CCA, and simulated linkage.

## Outputs

### Tables (CSV)

| File | Content |
|------|---------|
| `indiv_1_match_exact_1N.csv` | 1/N rate by threshold × mode × level |
| `indiv_2_match_strict.csv` | Strict rate by threshold × mode × level |
| `indiv_4_hacker_accuracy.csv` | M1/M2/M3 accuracy |
| `indiv_5_rarity_score.csv` | Accuracy + win/fail confidence |
| `indiv_6_monte_carlo.csv` | Monte-Carlo individual data |
| `indiv_7_risk_score_net.csv` | Net score [-100, 100] |
| `inference_individuel.csv` | Individual inference data |
| `inference_indicateurs.csv` | Aggregated inference indicators |

### Figures (PNG)

| Prefix | Content |
|--------|---------|
| `A1_*` | Exact match 1/N |
| `A2_*` | Strict match |
| `A3_*` | Confidence matrix (stacked bars) |
| `A8_*` | Monte-Carlo + rarity |
| `B1_*` | Real inference rate (2×3 grid) |
| `B2_*` | Certainty & inference risk |
| `B4_*` | Disorientation index Δ |
| `B5_*` | Inference risk matrix (4 quadrants) |
| `B7_*` | True/false distinction (filtering capacity) |

## Attacker Knowledge Levels

| Level | QIs | Granularity |
|-------|-----|-------------|
| N1 (Low) | 6 | Age in 10-year bands, year only, admission/discharge mode |
| N2 (Medium) | 8 | Age in 5-year bands, year+month, + specialty/surgery |
| N3 (High) | 11 | Exact age, full date, + diabetes/renal failure/dementia |

## [License](https://github.com/edlansiaux/ridentif_risk_assessment/blob/main/LICENSE)
