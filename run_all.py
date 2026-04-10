#!/usr/bin/env python3
"""
run_all.py — Exécute l'intégralité des expériences du rapport.

Usage :
    python run_all.py                     # tout
    python run_all.py --only indiv        # individualisation seule
    python run_all.py --only inference    # inférence seule
    python run_all.py --fast              # paliers 0,20,40,60,80,100 + n_sample=200
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np

# Ajout du répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    PERCENTAGES, PERCENTAGES_COARSE, MODES, N_SAMPLE, FIGURES_DIR, TABLES_DIR,
)
from src.data_loader import load_kb, build_kb_views, load_transformed, build_configs
from src.individualization import (
    match_exact_1_over_n,
    match_exact_strict,
    confidence_matrix,
    hacker_accuracy,
    weighted_rarity_score,
    monte_carlo_stability,
    risk_score_net,
    weighted_risk_matrix,
)
from src.inference import (
    build_ground_truth,
    run_inference_scenario,
    aggregate_inference,
)
from src.visualization import (
    plot_reid_rate,
    plot_confidence_matrix,
    plot_monte_carlo,
    plot_inference_success,
    plot_inference_certainty,
    plot_inference_disorientation,
    plot_inference_matrix,
    plot_inference_vrai_faux,
)


def parse_args():
    p = argparse.ArgumentParser(description="Expériences de réidentification")
    p.add_argument("--only", choices=["indiv", "inference"], default=None,
                   help="N'exécuter qu'une partie")
    p.add_argument("--fast", action="store_true",
                   help="Mode rapide (6 paliers, n_sample=200)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# INDIVIDUALISATION
# ═══════════════════════════════════════════════════════════════

def run_individualization(configs, percentages, modes, n_sample):
    print("\n" + "=" * 70)
    print("  PARTIE I — INDIVIDUALISATION")
    print("=" * 70)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- 1. Match exact 1/N ---
    print("\n[1/7] Match exact (1/N)…")
    rows_1n = []
    for pct in percentages:
        res = {"mélange_%": pct}
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
                for name, qi, df_kb in configs:
                    res[f"{mode}_{name}"] = match_exact_1_over_n(df_kb, df_p, qi)
            except FileNotFoundError:
                pass
        rows_1n.append(res)
        print(f"  {pct}%", end=" ", flush=True)
    df_1n = pd.DataFrame(rows_1n)
    df_1n.to_csv(f"{TABLES_DIR}/indiv_1_match_exact_1N.csv", index=False)
    plot_reid_rate(df_1n, title_suffix="(1/N)", filename="A1_match_exact_1N")
    print()

    # --- 2. Match exact strict (N=1) ---
    print("[2/7] Match exact strict (N=1)…")
    rows_strict = []
    for pct in percentages:
        res = {"mélange_%": pct}
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
                for name, qi, df_kb in configs:
                    res[f"{mode}_{name}"] = match_exact_strict(df_kb, df_p, qi)
            except FileNotFoundError:
                pass
        rows_strict.append(res)
        print(f"  {pct}%", end=" ", flush=True)
    df_strict = pd.DataFrame(rows_strict)
    df_strict.to_csv(f"{TABLES_DIR}/indiv_2_match_strict.csv", index=False)
    plot_reid_rate(df_strict, title_suffix="stricte (N=1)", filename="A2_match_strict")
    print()

    # --- 3. Matrice de confiance ---
    print("[3/7] Matrice de confiance…")
    pcts_coarse = [p for p in percentages if p % 10 == 0]
    results_matrix = {m: {l: [] for l in ["N1", "N2", "N3"]} for m in modes}
    for pct in pcts_coarse:
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
                for name, qi, df_kb in configs:
                    cm = confidence_matrix(df_kb, df_p, qi)
                    results_matrix[mode][name.upper()].append(
                        [cm["critique"], cm["leurre"], cm["noye"], cm["echec"]]
                    )
            except FileNotFoundError:
                for lvl in ["N1", "N2", "N3"]:
                    results_matrix[mode][lvl].append([0, 0, 0, 100])
        print(f"  {pct}%", end=" ", flush=True)
    plot_confidence_matrix(results_matrix, pcts_coarse, filename="A3_conf_matrix")
    print()

    # --- 4. Fiabilité du hacker ---
    print("[4/7] Fiabilité du hacker…")
    rows_acc = []
    for pct in percentages:
        res = {"mélange_%": pct}
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
                for name, qi, df_kb in configs:
                    ha = hacker_accuracy(df_kb, df_p, qi)
                    for mk, mv in ha.items():
                        res[f"{mode}_{name}_{mk}"] = mv
            except FileNotFoundError:
                pass
        rows_acc.append(res)
        print(f"  {pct}%", end=" ", flush=True)
    df_acc = pd.DataFrame(rows_acc)
    df_acc.to_csv(f"{TABLES_DIR}/indiv_4_hacker_accuracy.csv", index=False)
    print()

    # --- 5. Score pondéré par rareté ---
    print("[5/7] Score pondéré par rareté + confiance…")
    rows_rar = []
    for pct in percentages:
        res = {"mélange_%": pct}
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
                for name, qi, df_kb in configs:
                    wr = weighted_rarity_score(df_kb, df_p, qi, n_sample=n_sample)
                    for k, v in wr.items():
                        res[f"{mode}_{name}_{k}"] = v
            except FileNotFoundError:
                pass
        rows_rar.append(res)
        print(f"  {pct}%", end=" ", flush=True)
    df_rar = pd.DataFrame(rows_rar)
    df_rar.to_csv(f"{TABLES_DIR}/indiv_5_rarity_score.csv", index=False)
    print()

    # --- 6. Monte-Carlo leave-one-out ---
    print("[6/7] Monte-Carlo leave-one-out…")
    all_mc_rows = []
    for pct in percentages:
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
                for name, qi, df_kb in configs:
                    indiv = monte_carlo_stability(df_kb, df_p, qi, n_sample=n_sample)
                    for r in indiv:
                        r["mélange_%"] = pct
                        r["mode"] = mode
                        r["niveau"] = name
                        all_mc_rows.append(r)
            except FileNotFoundError:
                pass
        print(f"  {pct}%", end=" ", flush=True)
    df_mc = pd.DataFrame(all_mc_rows)
    df_mc.to_csv(f"{TABLES_DIR}/indiv_6_monte_carlo.csv", index=False)
    if not df_mc.empty:
        plot_monte_carlo(df_mc, filename="A8_monte_carlo")
    print()

    # --- 7. Score net de l'attaquant ---
    print("[7/7] Score net de l'attaquant…")
    rows_net = []
    for pct in percentages:
        res = {"mélange_%": pct}
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
                for name, qi, df_kb in configs:
                    res[f"{mode}_{name}"] = risk_score_net(
                        df_kb, df_p, qi, n_sample=min(500, n_sample)
                    )
            except FileNotFoundError:
                pass
        rows_net.append(res)
        print(f"  {pct}%", end=" ", flush=True)
    df_net = pd.DataFrame(rows_net)
    df_net.to_csv(f"{TABLES_DIR}/indiv_7_risk_score_net.csv", index=False)
    print()

    print("\n✓ Individualisation terminée.")
    return df_1n, df_strict, df_mc


# ═══════════════════════════════════════════════════════════════
# INFÉRENCE
# ═══════════════════════════════════════════════════════════════

def run_inference(configs, percentages, modes, n_sample):
    print("\n" + "=" * 70)
    print("  PARTIE II — INFÉRENCE")
    print("=" * 70)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Vérité terrain
    print("\nConstruction de la vérité terrain…")
    df_0 = load_transformed("direct", 0)
    gt = build_ground_truth(df_0)
    n_diag = len({c for s in gt["diag"].values() for c in s})
    n_acte = len({c for s in gt["acte"].values() for c in s})
    print(f"  CIM-10 : {n_diag} codes uniques | CCAM : {n_acte} codes uniques")

    # Boucle principale
    print("\nBoucle principale (palier × mode × niveau × code_type × 3 méthodes)…")
    all_rows = []
    for pct in percentages:
        for mode in modes:
            try:
                df_p = load_transformed(mode, pct)
            except FileNotFoundError:
                continue
            for name, qi, df_kb in configs:
                for code_type in ["diag", "acte"]:
                    rows = run_inference_scenario(
                        df_kb, df_p, qi, gt,
                        code_type=code_type, n_sample=n_sample,
                    )
                    for r in rows:
                        r["mélange_%"] = pct
                        r["mode"] = mode
                        r["niveau"] = name
                        r["code_type"] = code_type
                        all_rows.append(r)
        print(f"  {pct}%", end=" ", flush=True)
    print()

    df_ind = pd.DataFrame(all_rows)
    df_ind.to_csv(f"{TABLES_DIR}/inference_individuel.csv", index=False)
    print(f"  → {len(df_ind)} lignes individuelles sauvegardées.")

    # Agrégation
    print("Agrégation des indicateurs…")
    df_agg = aggregate_inference(df_ind)
    df_agg.to_csv(f"{TABLES_DIR}/inference_indicateurs.csv", index=False)
    print(f"  → {len(df_agg)} lignes agrégées.")

    # Visualisations
    print("Génération des figures d'inférence…")
    plot_inference_success(df_agg)
    plot_inference_certainty(df_agg)
    plot_inference_disorientation(df_agg)
    plot_inference_matrix(df_agg)
    plot_inference_vrai_faux(df_agg)

    print("\n✓ Inférence terminée.")
    return df_ind, df_agg


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()

    # Configuration
    if args.fast:
        percentages = [0, 20, 40, 60, 80, 100]
        n_sample = 200
        print("⚡ Mode rapide activé (6 paliers, n_sample=200)")
    else:
        percentages = PERCENTAGES
        n_sample = N_SAMPLE

    modes = MODES

    # Chargement
    print("Chargement des données…")
    df_kb = load_kb()
    kb_views = build_kb_views(df_kb)
    configs = build_configs(kb_views)
    print(f"  Base de connaissances : {len(df_kb)} patients")
    for name, qi, _ in configs:
        print(f"  {name.upper()} : {len(qi)} QI")

    # Exécution
    if args.only != "inference":
        run_individualization(configs, percentages, modes, n_sample)

    if args.only != "indiv":
        run_inference(configs, percentages, modes, n_sample)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  TERMINÉ en {elapsed / 60:.1f} minutes.")
    print(f"  Figures → {FIGURES_DIR}/")
    print(f"  Tables  → {TABLES_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
