"""
Configuration centrale — quasi-identifiants, paliers, chemins.
"""

# ═══════════════════════════════════════════════════════════════
# Quasi-identifiants par niveau de connaissance attaquant
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
    "n1": {"qi": QI_N1, "label": "N1 (Faible)", "color": "#2ecc71"},
    "n2": {"qi": QI_N2, "label": "N2 (Moyen)",  "color": "#3498db"},
    "n3": {"qi": QI_N3, "label": "N3 (Élevé)",  "color": "#e74c3c"},
}

# ═══════════════════════════════════════════════════════════════
# Paliers de mélange testés
# ═══════════════════════════════════════════════════════════════

PERCENTAGES = list(range(0, 105, 5))           # 0, 5, 10, …, 100
PERCENTAGES_COARSE = list(range(0, 110, 10))   # 0, 10, 20, …, 100
MODES = ["direct", "sample"]

# ═══════════════════════════════════════════════════════════════
# Chemins
# ═══════════════════════════════════════════════════════════════

DATA_DIR         = "projet_donnees"
KB_FILE          = f"{DATA_DIR}/connaissances_externes.txt"
OUTPUT_DIR       = "outputs"
FIGURES_DIR      = f"{OUTPUT_DIR}/figures"
TABLES_DIR       = f"{OUTPUT_DIR}/tables"

# ═══════════════════════════════════════════════════════════════
# Hyper-paramètres expérimentaux
# ═══════════════════════════════════════════════════════════════

N_SAMPLE          = 1000    # Patients testés par configuration
N_MONTE_CARLO     = 20      # Itérations Monte-Carlo (leave-one-out)
BATCH_SIZE        = 500     # Taille de batch pour le scoring pondéré
CONFIDENCE_THRESH = 50.0    # Seuil de confiance (%) pour la matrice
SEED              = 42

# ═══════════════════════════════════════════════════════════════
# Seuils de risque (pour les bandes colorées)
# ═══════════════════════════════════════════════════════════════

RISK_BANDS = [
    (0,   5,  "#32CD32", 0.15, "Très sûr (0–5 %)"),
    (5,  10,  "#006400", 0.15, "Sûr (5–10 %)"),
    (10, 20,  "#FFD700", 0.25, "Vigilance (10–20 %)"),
    (20, 35,  "#FF8C00", 0.25, "Risque élevé (20–35 %)"),
    (35, 50,  "#FF0000", 0.15, "Danger (35–50 %)"),
    (50, 100, "#000000", 0.35, "Critique (>50 %)"),
]
