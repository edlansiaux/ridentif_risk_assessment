"""
Central configuration — quasi-identifiers, thresholds, paths.
"""

# ═══════════════════════════════════════════════════════════════
# Quasi-identifiers by attacker knowledge level
# ═══════════════════════════════════════════════════════════════

QI_N1 = [
    "age10", "sexe",
    "entree_mode", "sortie_mode",
    "entree_date_y", "sortie_date_y",
]

QI_N2 = [
    "age5", "sexe",
    "entree_mode", "sortie_mode",
    "entree_date_ym", "sortie_date_ym",
    "specialite", "chirurgie",
]

QI_N3 = [
    "age", "sexe",
    "entree_mode", "sortie_mode",
    "entree_date_ymd", "sortie_date_ymd",
    "specialite", "chirurgie",
    "diabete", "insuffisance_renale", "demence",
]

LEVELS = {
    "n1": {"qi": QI_N1, "label": "N1 (Low)",    "color": "#2ecc71"},
    "n2": {"qi": QI_N2, "label": "N2 (Medium)", "color": "#3498db"},
    "n3": {"qi": QI_N3, "label": "N3 (High)",   "color": "#e74c3c"},
}

# ═══════════════════════════════════════════════════════════════
# Shuffling thresholds tested
# ═══════════════════════════════════════════════════════════════

PERCENTAGES = list(range(0, 105, 5))           # 0, 5, 10, …, 100
PERCENTAGES_COARSE = list(range(0, 110, 10))   # 0, 10, 20, …, 100
MODES = ["direct", "sample"]

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

DATA_DIR         = "project_data"
KB_FILE          = f"{DATA_DIR}/external_knowledge.txt"
OUTPUT_DIR       = "outputs"
FIGURES_DIR      = f"{OUTPUT_DIR}/figures"
TABLES_DIR       = f"{OUTPUT_DIR}/tables"

# ═══════════════════════════════════════════════════════════════
# Experimental hyperparameters
# ═══════════════════════════════════════════════════════════════

N_SAMPLE          = 1000    # Patients tested per configuration
N_MONTE_CARLO     = 20      # Monte-Carlo iterations (leave-one-out)
BATCH_SIZE        = 500     # Batch size for weighted scoring
CONFIDENCE_THRESH = 50.0    # Confidence threshold (%) for the matrix
SEED              = 42

# ═══════════════════════════════════════════════════════════════
# Risk thresholds (for colored bands)
# ═══════════════════════════════════════════════════════════════

RISK_BANDS = [
    (0,   5,  "#32CD32", 0.15, "Very safe (0–5 %)"),
    (5,  10,  "#006400", 0.15, "Safe (5–10 %)"),
    (10, 20,  "#FFD700", 0.25, "Caution (10–20 %)"),
    (20, 35,  "#FF8C00", 0.25, "High risk (20–35 %)"),
    (35, 50,  "#FF0000", 0.15, "Danger (35–50 %)"),
    (50, 100, "#000000", 0.35, "Critical (>50 %)"),
]
