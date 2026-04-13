"""
Inference indicators — three methods × two code types.

Method A : Bayesian majority vote   P(c|EC)
Method B : Bayesian lift            P(c|EC) / P(c|global)
Method C : Weighted rarity          P(c|EC) × (1 - P(c|global))

Codes: ICD-10 (3 chars) and CCAM (4 chars).
"""

import numpy as np
import pandas as pd
from collections import Counter
from .config import N_SAMPLE, SEED


# ═══════════════════════════════════════════════════════════════
# Code parsers
# ═══════════════════════════════════════════════════════════════

def truncate_diag(s) -> frozenset:
    """ICD-10 → first 3 characters."""
    if pd.isna(s) or str(s).strip() == "":
        return frozenset()
    return frozenset(c.strip()[:3] for c in str(s).split(";") if len(c.strip()) >= 3)


def truncate_acte(s) -> frozenset:
    """CCAM → first 4 characters."""
    if pd.isna(s) or str(s).strip() == "":
        return frozenset()
    return frozenset(c.strip()[:4] for c in str(s).split(";") if len(c.strip()) >= 4)


# ═══════════════════════════════════════════════════════════════
# Ground truth construction
# ═══════════════════════════════════════════════════════════════

def build_ground_truth(df_original: pd.DataFrame) -> dict:
    """
    Builds ground truth dictionaries from the 0% baseline.
    Returns {"diag": {id: frozenset(codes)}, "acte": {id: frozenset(codes)}}.
    """
    gt_diag = df_original.set_index("id_sejour")["liste_diag"].apply(truncate_diag).to_dict()
    gt_acte = df_original.set_index("id_sejour")["liste_acte"].apply(truncate_acte).to_dict()
    return {"diag": gt_diag, "acte": gt_acte}


# ═══════════════════════════════════════════════════════════════
# Core function — one pass = 3 methods
# ═══════════════════════════════════════════════════════════════

def run_inference_scenario(
    df_kb: pd.DataFrame,
    df_anon: pd.DataFrame,
    qi_cols: list[str],
    ground_truth: dict,
    code_type: str = "diag",
    n_sample: int = N_SAMPLE,
    seed: int = SEED,
) -> list[dict]:
    """
    For each KB patient, computes all 3 inference methods.
    Returns 3 rows per patient (one per method A/B/C).

    Each row contains:
      id_sejour, method, is_correct, confidence, top1_proba,
      n_distinct_codes, ec_size.
    """
    df_anon = df_anon.copy().reset_index(drop=True)
    df_anon.columns = df_anon.columns.str.strip()
    valid = [c for c in qi_cols if c in df_anon.columns]
    if not valid:
        return []

    trunc = truncate_diag if code_type == "diag" else truncate_acte
    col_src = "liste_diag" if code_type == "diag" else "liste_acte"
    gt = ground_truth[code_type]

    anon_sets = df_anon[col_src].apply(trunc).values

    # Global distribution
    all_codes = Counter()
    for s in anon_sets:
        all_codes.update(s)
    n_total = len(df_anon)
    p_global = {c: v / n_total for c, v in all_codes.items()}

    # Pre-compute equivalence classes
    ec_map = df_anon.groupby(valid).indices

    df_kb_s = df_kb.sample(min(n_sample, len(df_kb)), random_state=seed).reset_index(drop=True)
    kb_vals = df_kb_s[valid].values
    kb_ids = df_kb_s["id_sejour"].values

    rows = []
    for i in range(len(df_kb_s)):
        pid = kb_ids[i]
        key = tuple(kb_vals[i]) if len(valid) > 1 else kb_vals[i][0]
        true_codes = gt.get(pid, frozenset())
        if not true_codes:
            continue

        if key not in ec_map:
            for m in ["A", "B", "C"]:
                rows.append({
                    "id_sejour": pid, "method": m,
                    "is_correct": False, "confidence": 0.0,
                    "top1_proba": 0.0, "n_distinct_codes": 0,
                    "ec_size": 0,
                })
            continue

        ec_idx = ec_map[key]
        ec_size = len(ec_idx)
        ec_sets = [anon_sets[j] for j in ec_idx]

        code_counts = Counter()
        for s in ec_sets:
            code_counts.update(s)
        n_distincts = len(code_counts)

        if not code_counts:
            for m in ["A", "B", "C"]:
                rows.append({
                    "id_sejour": pid, "method": m,
                    "is_correct": False, "confidence": 0.0,
                    "top1_proba": 0.0, "n_distinct_codes": 0,
                    "ec_size": ec_size,
                })
            continue

        # Three scoring methods
        scores_A = {c: n / ec_size for c, n in code_counts.items()}
        scores_B = {
            c: (n / ec_size) / p_global.get(c, 1.0 / n_total)
            for c, n in code_counts.items()
        }
        scores_C = {
            c: (n / ec_size) * (1.0 - p_global.get(c, 0.0))
            for c, n in code_counts.items()
        }

        for mname, scores in [("A", scores_A), ("B", scores_B), ("C", scores_C)]:
            sorted_c = sorted(scores.items(), key=lambda x: -x[1])
            s1_code, s1_val = sorted_c[0]
            s2_val = sorted_c[1][1] if len(sorted_c) >= 2 else 0.0
            conf = (1.0 - s2_val / s1_val) if s1_val > 0 else 0.0

            rows.append({
                "id_sejour":       pid,
                "method":          mname,
                "is_correct":      s1_code in true_codes,
                "confidence":      conf * 100,
                "top1_proba":      (code_counts[s1_code] / ec_size) * 100,
                "n_distinct_codes": n_distincts,
                "ec_size":         ec_size,
            })

    return rows


# ═══════════════════════════════════════════════════════════════
# Aggregation of individual results
# ═══════════════════════════════════════════════════════════════

def aggregate_inference(df_ind: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Aggregates individual inference data into an indicators DataFrame by group:
      (shuffling_%, mode, level, method, code_type).

    Computed metrics:
      acc, certainty, conf_mean, conf_win, conf_fail,
      delta_diso, m1_uniqueness, m2_restricted, m3_uncertain,
      mat_critical, mat_decoy, mat_submerged, mat_failure.
    """
    grp = df_ind.groupby(["shuffling_%", "mode", "level", "method", "code_type"])
    out = []

    for keys, sub in grp:
        n = len(sub)
        if n == 0:
            continue
        ic = sub["is_correct"].astype(bool).values
        cf = sub["confidence"].values
        nd = sub["n_distinct_codes"].values

        acc = ic.mean() * 100
        certainty = (cf >= threshold).mean() * 100
        conf_mean = cf.mean()
        conf_win = cf[ic].mean() if ic.any() else 0.0
        conf_fail = cf[~ic].mean() if (~ic).any() else 0.0
        delta_diso = certainty - acc

        # M1/M2/M3 rules
        m1 = nd == 1
        m1_acc = ic[m1].mean() * 100 if m1.any() else 0.0

        m2 = (nd >= 1) & (nd <= 2)
        m2_acc = (ic[m2].astype(float) / nd[m2]).mean() * 100 if m2.any() else 0.0

        m3 = (nd >= 1) & (nd <= 5)
        m3_acc = (ic[m3].astype(float) / nd[m3]).mean() * 100 if m3.any() else 0.0

        # 4-quadrant matrix
        high = cf >= threshold
        mat_crit = (high & ic).sum()  / n * 100
        mat_decoy = (high & ~ic).sum() / n * 100
        mat_subm = (~high & ic).sum() / n * 100
        mat_fail = (~high & ~ic).sum() / n * 100

        out.append({
            "shuffling_%": keys[0], "mode": keys[1], "level": keys[2],
            "method": keys[3], "code_type": keys[4],
            "acc": acc, "certainty": certainty, "conf_mean": conf_mean,
            "conf_win": conf_win, "conf_fail": conf_fail,
            "delta_diso": delta_diso,
            "m1_uniqueness": m1_acc, "m2_restricted": m2_acc, "m3_uncertain": m3_acc,
            "mat_critical": mat_crit, "mat_decoy": mat_decoy,
            "mat_submerged": mat_subm, "mat_failure": mat_fail,
        })

    return pd.DataFrame(out)
