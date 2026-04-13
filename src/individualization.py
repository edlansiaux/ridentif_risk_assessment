"""
Individualization indicators — seven complementary approaches.

Method 1 : Exact match (1/N probability)
Method 2 : Strict exact match (N=1 unique & correct)
Method 3 : Confidence matrix (4 quadrants: perception/reality)
Method 4 : Hacker accuracy (uniqueness, restricted group, uncertainty)
Method 5 : Rarity-weighted score + confidence index (gap)
Method 6 : Monte-Carlo leave-one-out + rarity
Method 7 : Attacker net score (confidence calibration)
"""

import numpy as np
import pandas as pd
from .config import N_SAMPLE, N_MONTE_CARLO, BATCH_SIZE, SEED


# ═══════════════════════════════════════════════════════════════
# Utility: column validation
# ═══════════════════════════════════════════════════════════════

def _valid_qi(qi_cols: list[str], df_anon: pd.DataFrame) -> list[str]:
    df_anon.columns = df_anon.columns.str.strip()
    return [c for c in qi_cols if c in df_anon.columns]


# ═══════════════════════════════════════════════════════════════
# 1. Exact match — 1/N re-identification probability
# ═══════════════════════════════════════════════════════════════

def match_exact_1_over_n(df_kb, df_anon, qi_cols) -> float:
    """Returns the re-identification rate (%) via 1/N probability."""
    df_anon = df_anon.copy()
    valid = _valid_qi(qi_cols, df_anon)
    if not valid:
        return 0.0

    counts = df_anon.groupby(valid).size().reset_index(name="n_suspects")
    matches = pd.merge(
        df_kb[valid + ["id_sejour"]],
        df_anon[valid + ["id_sejour"]],
        on=valid, how="inner", suffixes=("_kb", "_anon"),
    )
    if matches.empty:
        return 0.0

    matches["is_correct"] = matches["id_sejour_kb"] == matches["id_sejour_anon"]
    matches = pd.merge(matches, counts, on=valid, how="left")
    matches["prob"] = matches["is_correct"] / matches["n_suspects"]

    total = matches.groupby("id_sejour_kb")["prob"].sum().sum()
    return (total / len(df_kb)) * 100


# ═══════════════════════════════════════════════════════════════
# 2. Strict exact match (N=1 unique & correct)
# ═══════════════════════════════════════════════════════════════

def match_exact_strict(df_kb, df_anon, qi_cols) -> float:
    """Returns the strict re-identification rate (%): single suspect AND correct."""
    df_anon = df_anon.copy()
    df_anon.columns = df_anon.columns.str.strip()
    valid = _valid_qi(qi_cols, df_anon)
    if not valid or df_anon.empty:
        return 0.0

    counts = df_anon.groupby(valid).size().reset_index(name="n_group_size")
    matches = pd.merge(
        df_kb[valid + ["id_sejour"]],
        df_anon[valid + ["id_sejour"]],
        on=valid, how="inner", suffixes=("_kb", "_anon"),
    )
    if matches.empty:
        return 0.0

    matches["is_correct"] = matches["id_sejour_kb"] == matches["id_sejour_anon"]
    matches = pd.merge(matches, counts, on=valid, how="left")

    stats = matches.groupby("id_sejour_kb").agg({
        "n_group_size": "sum",
        "is_correct": "any",
    })
    success = ((stats["n_group_size"] == 1) & stats["is_correct"]).sum()
    return (success / len(df_kb)) * 100


# ═══════════════════════════════════════════════════════════════
# 3. Confidence matrix (4 quadrants)
# ═══════════════════════════════════════════════════════════════

def confidence_matrix(df_kb, df_anon, qi_cols) -> dict:
    """
    Returns a dict with percentages:
      critical  : unique match & correct
      decoy     : unique match & incorrect
      submerged : multiple matches & correct
      failure   : everything else
    + delta     : disorientation index
    """
    df_anon = df_anon.copy()
    df_anon.columns = df_anon.columns.str.strip()
    valid = _valid_qi(qi_cols, df_anon)
    if not valid or df_anon.empty:
        return {"critical": 0, "decoy": 0, "submerged": 0, "failure": 100, "delta": 0}

    counts = df_anon.groupby(valid).size().reset_index(name="n_suspects")
    matches = pd.merge(
        df_kb[valid + ["id_sejour"]],
        df_anon[valid + ["id_sejour"]],
        on=valid, how="inner", suffixes=("_kb", "_anon"),
    )
    if matches.empty:
        return {"critical": 0, "decoy": 0, "submerged": 0, "failure": 100, "delta": 0}

    matches["is_correct"] = matches["id_sejour_kb"] == matches["id_sejour_anon"]
    matches = pd.merge(matches, counts, on=valid, how="left")

    n = len(df_kb)
    crit = ((matches["n_suspects"] == 1) & matches["is_correct"]).sum()
    decoy = ((matches["n_suspects"] == 1) & ~matches["is_correct"]).sum()
    submerged = ((matches["n_suspects"] > 1) & matches["is_correct"]).sum()
    failure = n - crit - decoy - submerged

    confidence = (matches["n_suspects"] == 1).sum() / n * 100
    success_rate = matches["is_correct"].sum() / n * 100

    return {
        "critical":  crit / n * 100,
        "decoy":     decoy / n * 100,
        "submerged": submerged / n * 100,
        "failure":   failure / n * 100,
        "delta":     confidence - success_rate,
    }


# ═══════════════════════════════════════════════════════════════
# 4. Hacker accuracy (3 rules)
# ═══════════════════════════════════════════════════════════════

def hacker_accuracy(df_kb, df_anon, qi_cols) -> dict:
    """
    Returns accuracy rates (%) under three rules:
      m1_uniqueness  : among N=1, % correct
      m2_restricted  : among N≤2, mean of (correct/N)
      m3_uncertain   : among N≤5, mean of (correct/N)
    """
    df_anon = df_anon.copy()
    df_anon.columns = df_anon.columns.str.strip()
    valid = _valid_qi(qi_cols, df_anon)
    if not valid or df_anon.empty:
        return {"m1": 0, "m2": 0, "m3": 0}

    counts = df_anon.groupby(valid).size().reset_index(name="n_suspects")
    matches = pd.merge(
        df_kb[valid + ["id_sejour"]],
        df_anon[valid + ["id_sejour"]],
        on=valid, how="inner", suffixes=("_kb", "_anon"),
    )
    if matches.empty:
        return {"m1": 0, "m2": 0, "m3": 0}

    matches["is_correct"] = matches["id_sejour_kb"] == matches["id_sejour_anon"]
    matches = pd.merge(matches, counts, on=valid, how="left")

    m1 = matches[matches["n_suspects"] == 1]
    m2 = matches[matches["n_suspects"] <= 2]
    m3 = matches[matches["n_suspects"] <= 5]

    return {
        "m1": m1["is_correct"].mean() * 100 if len(m1) else 0,
        "m2": (m2["is_correct"] / m2["n_suspects"]).mean() * 100 if len(m2) else 0,
        "m3": (m3["is_correct"] / m3["n_suspects"]).mean() * 100 if len(m3) else 0,
    }


# ═══════════════════════════════════════════════════════════════
# 5. Rarity-weighted score + confidence index (gap)
# ═══════════════════════════════════════════════════════════════

def weighted_rarity_score(df_kb, df_anon, qi_cols,
                          n_sample=N_SAMPLE, batch_size=BATCH_SIZE) -> dict:
    """
    Score(i,j) = Σ 1[QI_k(i)=QI_k(j)] × log(1/freq(QI_k(i)))
    Returns: acc, conf_mean, conf_win, conf_fail (all in %).
    """
    df_anon = df_anon.copy()
    df_anon.columns = df_anon.columns.str.strip()
    valid = _valid_qi(qi_cols, df_anon)
    if not valid:
        return {"acc": 0, "conf_win": 0, "conf_fail": 0}

    col_weights = {
        col: 1.0 - df_anon[col].value_counts(normalize=True)
        for col in valid
    }

    df_kb_s = df_kb.sample(min(n_sample, len(df_kb)), random_state=SEED)
    kb_ids = df_kb_s["id_sejour"].values
    anon_ids = df_anon["id_sejour"].values
    n_anon = len(df_anon)

    all_correct, all_conf = [], []

    for start in range(0, len(df_kb_s), batch_size):
        end = min(start + batch_size, len(df_kb_s))
        batch = df_kb_s.iloc[start:end]
        score_mat = np.zeros((len(batch), n_anon))

        for col in valid:
            b = batch[col].values
            a = df_anon[col].values
            m = b[:, None] == a[None, :]
            w = col_weights[col].reindex(b).fillna(0).values
            score_mat += m * w[:, None]

        if n_anon >= 2:
            top2 = np.argpartition(score_mat, -2, axis=1)[:, -2:]
            ridx = np.arange(len(batch))[:, None]
            t2s = score_mat[ridx, top2]
            srt = np.argsort(t2s, axis=1)
            s1 = t2s[np.arange(len(t2s)), srt[:, 1]]
            s2 = t2s[np.arange(len(t2s)), srt[:, 0]]
            best = top2[np.arange(len(top2)), srt[:, 1]]
        else:
            s1 = score_mat[:, 0]
            s2 = np.zeros_like(s1)
            best = np.zeros(len(s1), dtype=int)

        correct = anon_ids[best] == kb_ids[start:end]
        conf = np.where(s1 > 0, 1.0 - s2 / s1, 0.0)
        all_correct.extend(correct)
        all_conf.extend(conf)

    c = np.array(all_correct)
    f = np.array(all_conf)
    return {
        "acc":       c.mean() * 100,
        "conf_win":  f[c].mean() * 100 if c.any() else 0,
        "conf_fail": f[~c].mean() * 100 if (~c).any() else 0,
    }


# ═══════════════════════════════════════════════════════════════
# 6. Monte-Carlo leave-one-out + rarity (individual stability)
# ═══════════════════════════════════════════════════════════════

def monte_carlo_stability(df_kb, df_anon, qi_cols,
                          n_sample=N_SAMPLE) -> list[dict]:
    """
    For each individual, systematically removes each QI (leave-one-out)
    and votes on the most stable candidate.
    Returns a list of per-individual dicts: {id_sejour, confidence, is_correct}.
    """
    df_anon_c = df_anon.copy()
    df_anon_c.columns = df_anon_c.columns.str.strip()
    valid = [c for c in qi_cols if c in df_anon_c.columns]
    if not valid:
        return []

    df_kb_s = df_kb.sample(min(n_sample, len(df_kb)), random_state=SEED).copy()
    kb_ids = df_kb_s["id_sejour"].values
    anon_ids = df_anon_c["id_sejour"].values
    n_iter = len(valid)

    col_weights = {
        col: 1.0 - df_anon_c[col].value_counts(normalize=True)
        for col in valid
    }

    votes = {i: {} for i in range(len(df_kb_s))}

    for col_out in valid:
        cur = [c for c in valid if c != col_out]
        score_mat = np.zeros((len(df_kb_s), len(df_anon_c)))
        for col in cur:
            b = df_kb_s[col].values
            a = df_anon_c[col].values
            w = col_weights[col].reindex(b).fillna(0).values
            score_mat += (b[:, None] == a[None, :]) * w[:, None]

        best = np.argmax(score_mat, axis=1)
        for ki, ai in enumerate(best):
            pid = anon_ids[ai]
            votes[ki][pid] = votes[ki].get(pid, 0) + 1

    results = []
    for i in range(len(df_kb_s)):
        cand = max(votes[i], key=votes[i].get)
        results.append({
            "id_sejour":  kb_ids[i],
            "confidence": votes[i][cand] / n_iter * 100,
            "is_correct": int(cand == kb_ids[i]),
        })
    return results


# ═══════════════════════════════════════════════════════════════
# 7. Attacker net score (Monte-Carlo + calibration)
# ═══════════════════════════════════════════════════════════════

def risk_score_net(df_kb, df_anon, qi_cols,
                   n_sample=500, n_iterations=N_MONTE_CARLO) -> float:
    """
    Net score ∈ [-100, 100].
    Positive = profitable attack; negative = counterproductive attack.
    """
    df_kb_s = df_kb.sample(min(n_sample, len(df_kb)), random_state=SEED)
    anon_ids = df_anon["id_sejour"].values
    kb_ids = df_kb_s["id_sejour"].values

    valid = [c for c in qi_cols if c in df_anon.columns]
    col_weights = {
        col: 1.0 - df_anon[col].value_counts(normalize=True)
        for col in valid
    }

    votes = {i: {} for i in range(len(df_kb_s))}

    for _ in range(n_iterations):
        cur = list(valid)
        if len(cur) > 1:
            cur.remove(np.random.choice(cur))

        score_mat = np.zeros((len(df_kb_s), len(df_anon)))
        for col in cur:
            b = df_kb_s[col].values
            a = df_anon[col].values
            w = col_weights[col].reindex(b).fillna(0).values
            score_mat += (b[:, None] == a[None, :]) * w[:, None]

        best = np.argmax(score_mat, axis=1)
        for ki, ai in enumerate(best):
            pid = anon_ids[ai]
            votes[ki][pid] = votes[ki].get(pid, 0) + 1

    total = 0.0
    for i in range(len(df_kb_s)):
        cand = max(votes[i], key=votes[i].get)
        conf = votes[i][cand] / n_iterations
        total += conf if cand == kb_ids[i] else -conf

    return (total / len(df_kb_s)) * 100


# ═══════════════════════════════════════════════════════════════
# 5b. Rarity-weighted risk matrix (4 quadrants with threshold)
# ═══════════════════════════════════════════════════════════════

def weighted_risk_matrix(df_kb, df_anon, qi_cols,
                         threshold=0.5, n_sample=N_SAMPLE,
                         batch_size=BATCH_SIZE) -> dict:
    """
    Perception/reality matrix based on rarity score.
    Returns: {critical, decoy, submerged, failure} as percentages.
    """
    df_anon = df_anon.copy()
    df_anon.columns = df_anon.columns.str.strip()
    valid = _valid_qi(qi_cols, df_anon)
    if not valid or df_anon.empty:
        return {"critical": 0, "decoy": 0, "submerged": 0, "failure": 100}

    col_weights = {
        col: 1.0 - df_anon[col].value_counts(normalize=True) for col in valid
    }

    df_kb_s = df_kb.sample(min(n_sample, len(df_kb)), random_state=SEED)
    kb_ids = df_kb_s["id_sejour"].values
    anon_ids = df_anon["id_sejour"].values
    n_anon = len(df_anon)

    all_correct, all_conf = [], []

    for start in range(0, len(df_kb_s), batch_size):
        end = min(start + batch_size, len(df_kb_s))
        batch = df_kb_s.iloc[start:end]
        sm = np.zeros((len(batch), n_anon))

        for col in valid:
            b = batch[col].values
            a = df_anon[col].values
            m = b[:, None] == a[None, :]
            w = col_weights[col].reindex(b).fillna(0).values
            sm += m * w[:, None]

        if n_anon >= 2:
            top2 = np.argpartition(sm, -2, axis=1)[:, -2:]
            ridx = np.arange(len(batch))[:, None]
            t2s = sm[ridx, top2]
            srt = np.argsort(t2s, axis=1)
            s1 = t2s[np.arange(len(t2s)), srt[:, 1]]
            s2 = t2s[np.arange(len(t2s)), srt[:, 0]]
            best = top2[np.arange(len(top2)), srt[:, 1]]
        else:
            s1 = sm[:, 0]; s2 = np.zeros_like(s1)
            best = np.zeros(len(s1), dtype=int)

        correct = anon_ids[best] == kb_ids[start:end]
        conf = np.where(s1 > 0, 1.0 - s2 / s1, 0.0)
        all_correct.extend(correct)
        all_conf.extend(conf)

    c = np.array(all_correct)
    f = np.array(all_conf)
    n = len(c)

    return {
        "critical":  ((f >= threshold) & c).sum()  / n * 100,
        "decoy":     ((f >= threshold) & ~c).sum() / n * 100,
        "submerged": ((f < threshold)  & c).sum()  / n * 100,
        "failure":   ((f < threshold)  & ~c).sum() / n * 100,
    }
