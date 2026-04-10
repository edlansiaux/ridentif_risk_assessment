"""
Visualisations — fonctions de tracé pour l'individualisation et l'inférence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .config import LEVELS, RISK_BANDS, FIGURES_DIR
import os

os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {k: v["color"] for k, v in LEVELS.items()}
COLORS_UP = {k.upper(): v for k, v in COLORS.items()}


def _add_risk_bands(ax, ymax=100):
    """Ajoute les bandes colorées de seuils de risque."""
    for lo, hi, col, alpha, _ in RISK_BANDS:
        if lo < ymax:
            ax.axhspan(lo, min(hi, ymax), facecolor=col, alpha=alpha, zorder=-10)


def _risk_legend():
    """Retourne les patches pour la légende des seuils."""
    return [mpatches.Patch(color=c, alpha=a, label=lab)
            for _, _, c, a, lab in RISK_BANDS]


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════
# INDIVIDUALISATION
# ═══════════════════════════════════════════════════════════════

def plot_reid_rate(df, col_prefix="direct", title_suffix="", filename="A1_reid_rate"):
    """Courbes de taux de réidentification (match exact 1/N)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    _add_risk_bands(ax)
    ax.grid(True, ls="--", alpha=0.4)

    for lvl in ["n1", "n2", "n3"]:
        for mode in ["direct", "sample"]:
            col = f"{mode}_{lvl}"
            if col in df.columns:
                ls = "-" if mode == "direct" else "--"
                ax.plot(df["mélange_%"], df[col],
                        color=COLORS[lvl], ls=ls, marker="o" if mode == "direct" else None,
                        lw=2, label=f"{lvl.upper()} {mode.capitalize()}")

    ax.set_xlabel("Taux de mélange (%)")
    ax.set_ylabel("Probabilité de succès du hacker (%)")
    ax.set_title(f"Risque de réidentification {title_suffix}", fontweight="bold")
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper right")
    _save(fig, filename)


def plot_confidence_matrix(results_matrix, percentages, filename="A2_conf_matrix"):
    """Barres empilées de la matrice de confiance."""
    labels = ["Succès critique", "Leurre (mélange)", "Noyé dans la masse", "Échec connu"]
    colors_mat = ["#d62728", "#3498db", "#f1c40f", "#2ecc71"]

    for mode in results_matrix:
        for lvl in results_matrix[mode]:
            data = np.array(results_matrix[mode][lvl])
            if len(data) == 0:
                continue
            fig, ax = plt.subplots(figsize=(14, 8))
            bottom = np.zeros(len(percentages))
            for k in range(4):
                ax.bar(percentages, data[:, k], bottom=bottom, width=8,
                       color=colors_mat[k], label=labels[k])
                bottom += data[:, k]

            ax.set_title(f"Matrice de risque — {lvl} — {mode}", fontweight="bold")
            ax.set_xlabel("Taux de mélange (%)")
            ax.set_ylabel("Pourcentage (%)")
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
            fig.tight_layout()
            _save(fig, f"{filename}_{mode}_{lvl}")


def plot_monte_carlo(df_individuel, filename="A8_monte_carlo"):
    """Taux de succès Monte-Carlo par niveau/mode avec bandes de risque."""
    fig, ax = plt.subplots(figsize=(14, 8))
    _add_risk_bands(ax)
    ax.grid(True, ls=":", alpha=0.3)

    summary = (df_individuel.groupby(["mélange_%", "mode", "niveau"])["is_correct"]
               .mean().reset_index())
    summary["is_correct"] *= 100

    for lvl in ["n1", "n2", "n3"]:
        for mode in ["direct", "sample"]:
            sub = summary[(summary["niveau"] == lvl) & (summary["mode"] == mode)]
            ls = "-" if mode == "direct" else "--"
            mk = "o" if mode == "direct" else "x"
            ax.plot(sub["mélange_%"], sub["is_correct"],
                    color=COLORS[lvl], ls=ls, marker=mk, lw=2,
                    label=f"{lvl.upper()} ({mode})")

    ax.set_xlabel("Taux de mélange (%)")
    ax.set_ylabel("Taux de réidentification (%)")
    ax.set_title("Monte-Carlo + rareté — Risque de réidentification", fontweight="bold")
    ax.legend()
    _save(fig, filename)


# ═══════════════════════════════════════════════════════════════
# INFÉRENCE
# ═══════════════════════════════════════════════════════════════

def plot_inference_success(df_agg, filename="B1_inference_success"):
    """Grille 2×3 : taux de succès réel par code_type × méthode."""
    code_types = [("diag", "CIM-10"), ("acte", "CCAM")]
    methods = ["A", "B", "C"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)

    for row, (ct, ct_label) in enumerate(code_types):
        for col, meth in enumerate(methods):
            ax = axes[row, col]
            _add_risk_bands(ax)
            ax.grid(True, ls=":", alpha=0.3)
            sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)]

            for lvl in ["n1", "n2", "n3"]:
                for mode in ["direct", "sample"]:
                    d = sub[(sub["niveau"] == lvl) & (sub["mode"] == mode)]
                    ls = "-" if mode == "direct" else "--"
                    mk = "o" if mode == "direct" else "x"
                    ax.plot(d["mélange_%"], d["acc"],
                            color=COLORS[lvl], ls=ls, marker=mk, ms=4, lw=1.5,
                            label=f"{lvl.upper()} {mode}")

            ax.set_title(f"{ct_label} — Méthode {meth}", fontweight="bold")
            if col == 0:
                ax.set_ylabel("Taux de succès (%)")
            if row == 1:
                ax.set_xlabel("Taux de mélange (%)")

    axes[0, 2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Taux d'inférence réel (agrégation par individu)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)


def plot_inference_certainty(df_agg, filename="B2_inference_certainty"):
    """Grille 2×3 : superposition certitude + succès réel."""
    code_types = [("diag", "CIM-10"), ("acte", "CCAM")]
    methods = ["A", "B", "C"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)

    for row, (ct, ct_label) in enumerate(code_types):
        for col, meth in enumerate(methods):
            ax = axes[row, col]
            _add_risk_bands(ax)
            sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)
                         & (df_agg["mode"] == "direct")]

            for lvl in ["n1", "n2", "n3"]:
                d = sub[sub["niveau"] == lvl]
                ax.plot(d["mélange_%"], d["acc"], color=COLORS[lvl],
                        lw=2, marker="o", ms=4, label=f"Succès {lvl.upper()}")
                ax.plot(d["mélange_%"], d["certitude"], color=COLORS[lvl],
                        lw=1.5, ls="--", label=f"Certitude {lvl.upper()}")

            ax.set_title(f"{ct_label} — Méthode {meth}")
            if col == 0:
                ax.set_ylabel("(%)")
            if row == 1:
                ax.set_xlabel("Taux de mélange (%)")

    axes[0, 2].legend(fontsize=7)
    fig.suptitle("Taux de certitude & risque d'inférence", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)


def plot_inference_disorientation(df_agg, filename="B4_disorientation"):
    """Grille 2×3 : indice de désorientation Δ."""
    code_types = [("diag", "CIM-10"), ("acte", "CCAM")]
    methods = ["A", "B", "C"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)

    for row, (ct, ct_label) in enumerate(code_types):
        for col, meth in enumerate(methods):
            ax = axes[row, col]
            ax.axhline(0, color="black", ls="--", lw=1)
            ax.grid(True, ls=":", alpha=0.3)
            sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)
                         & (df_agg["mode"] == "direct")]

            for lvl in ["n1", "n2", "n3"]:
                d = sub[sub["niveau"] == lvl]
                ax.plot(d["mélange_%"], d["delta_diso"],
                        color=COLORS[lvl], lw=2, marker="o", ms=4,
                        label=f"Δ {lvl.upper()}")

            ax.set_title(f"{ct_label} — Méthode {meth}")
            if col == 0:
                ax.set_ylabel("Écart de perception Δ (%)")
            if row == 1:
                ax.set_xlabel("Taux de mélange (%)")

    axes[0, 2].legend()
    fig.suptitle("Indice de désorientation (Δ = Certitude − Succès réel)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)


def plot_inference_matrix(df_agg, filename="B5_inference_matrix"):
    """Barres empilées de la matrice 4 quadrants pour l'inférence."""
    labels = ["Succès critique", "Leurre", "Noyé dans la masse", "Échec connu"]
    colors_mat = ["#d62728", "#3498db", "#f1c40f", "#2ecc71"]
    mat_cols = ["mat_critique", "mat_leurre", "mat_noye", "mat_echec"]

    for ct, ct_label in [("diag", "CIM-10"), ("acte", "CCAM")]:
        for mode in ["direct", "sample"]:
            fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
            for col, meth in enumerate(["A", "B", "C"]):
                ax = axes[col]
                sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)
                             & (df_agg["mode"] == mode)]

                pcts = sorted(sub["mélange_%"].unique())
                width = 2.2

                for ip, p in enumerate(pcts):
                    for jl, lvl in enumerate(["n1", "n2", "n3"]):
                        row = sub[(sub["mélange_%"] == p) & (sub["niveau"] == lvl)]
                        if row.empty:
                            continue
                        pos_x = p + (jl - 1) * (width + 0.3)
                        bottom = 0
                        for k, mc in enumerate(mat_cols):
                            val = row[mc].values[0]
                            lbl = labels[k] if (ip == 0 and jl == 0) else None
                            ax.bar(pos_x, val, bottom=bottom, width=width,
                                   color=colors_mat[k], label=lbl, edgecolor="black", lw=0.2)
                            bottom += val

                ax.set_title(f"Méthode {meth}")
                ax.set_xlabel("Taux de mélange (%)")
                if col == 0:
                    ax.set_ylabel("(%)")

            axes[0].legend(fontsize=7, loc="upper left")
            fig.suptitle(f"Matrice de risque — Inférence {ct_label} — {mode}",
                         fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            _save(fig, f"{filename}_{ct}_{mode}")


def plot_inference_vrai_faux(df_agg, filename="B7_vrai_faux"):
    """Grille 2×3 : confiance quand raison vs tort (capacité de filtrage)."""
    code_types = [("diag", "CIM-10"), ("acte", "CCAM")]
    methods = ["A", "B", "C"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)

    for row, (ct, ct_label) in enumerate(code_types):
        for col, meth in enumerate(methods):
            ax = axes[row, col]
            ax.grid(True, ls=":", alpha=0.3)
            sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)
                         & (df_agg["mode"] == "direct")]

            for lvl in ["n1", "n2", "n3"]:
                d = sub[sub["niveau"] == lvl]
                ax.plot(d["mélange_%"], d["conf_win"], color=COLORS[lvl],
                        lw=2, marker="o", ms=4, label=f"Raison {lvl.upper()}")
                ax.plot(d["mélange_%"], d["conf_fail"], color=COLORS[lvl],
                        lw=1.5, ls="--", label=f"Tort {lvl.upper()}")

            ax.set_title(f"{ct_label} — Méthode {meth}")
            if col == 0:
                ax.set_ylabel("Indice de confiance (0–100)")
            if row == 1:
                ax.set_xlabel("Taux de mélange (%)")

    axes[0, 2].legend(fontsize=7)
    fig.suptitle("Fiabilité du hacker : Capacité à discerner le vrai du faux",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)
