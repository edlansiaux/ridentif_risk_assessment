"""
Visualizations — plotting functions for individualization and inference.
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
    """Add colored risk threshold bands to an axis."""
    for lo, hi, col, alpha, _ in RISK_BANDS:
        if lo < ymax:
            ax.axhspan(lo, min(hi, ymax), facecolor=col, alpha=alpha, zorder=-10)


def _risk_legend():
    """Return patches for the risk threshold legend."""
    return [mpatches.Patch(color=c, alpha=a, label=lab)
            for _, _, c, a, lab in RISK_BANDS]


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════
# INDIVIDUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_reid_rate(df, col_prefix="direct", title_suffix="", filename="A1_reid_rate"):
    """Re-identification rate curves (exact match 1/N)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    _add_risk_bands(ax)
    ax.grid(True, ls="--", alpha=0.4)

    for lvl in ["n1", "n2", "n3"]:
        for mode in ["direct", "sample"]:
            col = f"{mode}_{lvl}"
            if col in df.columns:
                ls = "-" if mode == "direct" else "--"
                ax.plot(df["shuffling_%"], df[col],
                        color=COLORS[lvl], ls=ls, marker="o" if mode == "direct" else None,
                        lw=2, label=f"{lvl.upper()} {mode.capitalize()}")

    ax.set_xlabel("Shuffling rate (%)")
    ax.set_ylabel("Attacker success probability (%)")
    ax.set_title(f"Re-identification risk {title_suffix}", fontweight="bold")
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper right")
    _save(fig, filename)


def plot_confidence_matrix(results_matrix, percentages, filename="A2_conf_matrix"):
    """Stacked bars for the confidence matrix."""
    labels = ["Critical success", "Decoy (shuffled)", "Submerged in crowd", "Known failure"]
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

            ax.set_title(f"Risk matrix — {lvl} — {mode}", fontweight="bold")
            ax.set_xlabel("Shuffling rate (%)")
            ax.set_ylabel("Percentage (%)")
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
            fig.tight_layout()
            _save(fig, f"{filename}_{mode}_{lvl}")


def plot_monte_carlo(df_individual, filename="A8_monte_carlo"):
    """Monte-Carlo success rate by level/mode with risk bands."""
    fig, ax = plt.subplots(figsize=(14, 8))
    _add_risk_bands(ax)
    ax.grid(True, ls=":", alpha=0.3)

    summary = (df_individual.groupby(["shuffling_%", "mode", "level"])["is_correct"]
               .mean().reset_index())
    summary["is_correct"] *= 100

    for lvl in ["n1", "n2", "n3"]:
        for mode in ["direct", "sample"]:
            sub = summary[(summary["level"] == lvl) & (summary["mode"] == mode)]
            ls = "-" if mode == "direct" else "--"
            mk = "o" if mode == "direct" else "x"
            ax.plot(sub["shuffling_%"], sub["is_correct"],
                    color=COLORS[lvl], ls=ls, marker=mk, lw=2,
                    label=f"{lvl.upper()} ({mode})")

    ax.set_xlabel("Shuffling rate (%)")
    ax.set_ylabel("Re-identification rate (%)")
    ax.set_title("Monte-Carlo + rarity — Re-identification risk", fontweight="bold")
    ax.legend()
    _save(fig, filename)


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

def plot_inference_success(df_agg, filename="B1_inference_success"):
    """2×3 grid: real success rate by code_type × method."""
    code_types = [("diag", "ICD-10"), ("acte", "CCAM")]
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
                    d = sub[(sub["level"] == lvl) & (sub["mode"] == mode)]
                    ls = "-" if mode == "direct" else "--"
                    mk = "o" if mode == "direct" else "x"
                    ax.plot(d["shuffling_%"], d["acc"],
                            color=COLORS[lvl], ls=ls, marker=mk, ms=4, lw=1.5,
                            label=f"{lvl.upper()} {mode}")

            ax.set_title(f"{ct_label} — Method {meth}", fontweight="bold")
            if col == 0:
                ax.set_ylabel("Success rate (%)")
            if row == 1:
                ax.set_xlabel("Shuffling rate (%)")

    axes[0, 2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Real inference rate (aggregated per individual)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)


def plot_inference_certainty(df_agg, filename="B2_inference_certainty"):
    """2×3 grid: overlaid certainty + real success rate."""
    code_types = [("diag", "ICD-10"), ("acte", "CCAM")]
    methods = ["A", "B", "C"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)

    for row, (ct, ct_label) in enumerate(code_types):
        for col, meth in enumerate(methods):
            ax = axes[row, col]
            _add_risk_bands(ax)
            sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)
                         & (df_agg["mode"] == "direct")]

            for lvl in ["n1", "n2", "n3"]:
                d = sub[sub["level"] == lvl]
                ax.plot(d["shuffling_%"], d["acc"], color=COLORS[lvl],
                        lw=2, marker="o", ms=4, label=f"Success {lvl.upper()}")
                ax.plot(d["shuffling_%"], d["certainty"], color=COLORS[lvl],
                        lw=1.5, ls="--", label=f"Certainty {lvl.upper()}")

            ax.set_title(f"{ct_label} — Method {meth}")
            if col == 0:
                ax.set_ylabel("(%)")
            if row == 1:
                ax.set_xlabel("Shuffling rate (%)")

    axes[0, 2].legend(fontsize=7)
    fig.suptitle("Certainty rate & inference risk", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)


def plot_inference_disorientation(df_agg, filename="B4_disorientation"):
    """2×3 grid: disorientation index Δ."""
    code_types = [("diag", "ICD-10"), ("acte", "CCAM")]
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
                d = sub[sub["level"] == lvl]
                ax.plot(d["shuffling_%"], d["delta_diso"],
                        color=COLORS[lvl], lw=2, marker="o", ms=4,
                        label=f"Δ {lvl.upper()}")

            ax.set_title(f"{ct_label} — Method {meth}")
            if col == 0:
                ax.set_ylabel("Perception gap Δ (%)")
            if row == 1:
                ax.set_xlabel("Shuffling rate (%)")

    axes[0, 2].legend()
    fig.suptitle("Disorientation index (Δ = Certainty − Real success)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)


def plot_inference_matrix(df_agg, filename="B5_inference_matrix"):
    """Stacked bars of the 4-quadrant matrix for inference."""
    labels = ["Critical success", "Decoy", "Submerged in crowd", "Known failure"]
    colors_mat = ["#d62728", "#3498db", "#f1c40f", "#2ecc71"]
    mat_cols = ["mat_critical", "mat_decoy", "mat_submerged", "mat_failure"]

    for ct, ct_label in [("diag", "ICD-10"), ("acte", "CCAM")]:
        for mode in ["direct", "sample"]:
            fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
            for col, meth in enumerate(["A", "B", "C"]):
                ax = axes[col]
                sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)
                             & (df_agg["mode"] == mode)]

                pcts = sorted(sub["shuffling_%"].unique())
                width = 2.2

                for ip, p in enumerate(pcts):
                    for jl, lvl in enumerate(["n1", "n2", "n3"]):
                        row = sub[(sub["shuffling_%"] == p) & (sub["level"] == lvl)]
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

                ax.set_title(f"Method {meth}")
                ax.set_xlabel("Shuffling rate (%)")
                if col == 0:
                    ax.set_ylabel("(%)")

            axes[0].legend(fontsize=7, loc="upper left")
            fig.suptitle(f"Risk matrix — Inference {ct_label} — {mode}",
                         fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            _save(fig, f"{filename}_{ct}_{mode}")


def plot_inference_vrai_faux(df_agg, filename="B7_vrai_faux"):
    """2×3 grid: confidence when right vs wrong (filtering capacity)."""
    code_types = [("diag", "ICD-10"), ("acte", "CCAM")]
    methods = ["A", "B", "C"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)

    for row, (ct, ct_label) in enumerate(code_types):
        for col, meth in enumerate(methods):
            ax = axes[row, col]
            ax.grid(True, ls=":", alpha=0.3)
            sub = df_agg[(df_agg["code_type"] == ct) & (df_agg["method"] == meth)
                         & (df_agg["mode"] == "direct")]

            for lvl in ["n1", "n2", "n3"]:
                d = sub[sub["level"] == lvl]
                ax.plot(d["shuffling_%"], d["conf_win"], color=COLORS[lvl],
                        lw=2, marker="o", ms=4, label=f"Correct {lvl.upper()}")
                ax.plot(d["shuffling_%"], d["conf_fail"], color=COLORS[lvl],
                        lw=1.5, ls="--", label=f"Wrong {lvl.upper()}")

            ax.set_title(f"{ct_label} — Method {meth}")
            if col == 0:
                ax.set_ylabel("Confidence index (0–100)")
            if row == 1:
                ax.set_xlabel("Shuffling rate (%)")

    axes[0, 2].legend(fontsize=7)
    fig.suptitle("Hacker reliability: Capacity to distinguish correct from incorrect",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, filename)
