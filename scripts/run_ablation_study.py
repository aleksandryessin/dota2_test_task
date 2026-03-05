#!/usr/bin/env python3
"""Ablation study: sequentially add feature groups to LightGBM V2.

Uses fixed best hyperparameters from Optuna HPO (trial #13, Acc@5=0.485).
Each step adds features ON TOP of the previous, so we track incremental gains.

Optimized: all feature blocks are pre-computed once, configs just select which to use.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from collections import Counter

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features_v2 import (
    parse_drafts, build_target_encoding, build_rolling_meta,
    build_context, build_ban_mask, build_series_features,
)

BEST_PARAMS = {
    "captain_alpha": 7.7382578018207955,
    "team_alpha": 40.08187695237324,
    "meta_window": 9000,
    "learning_rate": 0.04481053671661708,
    "num_leaves": 33,
    "max_depth": 6,
    "min_child_samples": 8,
    "subsample": 0.8445779114924311,
    "colsample_bytree": 0.5925101071718902,
    "reg_alpha": 0.05039156628791337,
    "reg_lambda": 2.080962230659839,
    "min_split_gain": 0.7006073144957847,
}

VAL_SIZE = 200
SEED = 42
MAX_BOOST_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 50
RESULTS_DIR = Path("ablation_results")

# 1=carry, 2=mid, 3=offlane, 4=soft_support, 5=hard_support
HERO_ROLES = {
    1: 1, 2: 3, 3: 5, 4: 1, 5: 5, 6: 1, 7: 4, 8: 1, 9: 4, 10: 1,
    11: 2, 12: 1, 13: 2, 14: 4, 15: 2, 16: 3, 17: 2, 18: 1, 19: 2, 20: 4,
    21: 3, 22: 2, 23: 2, 25: 2, 26: 5, 27: 5, 28: 3, 29: 3, 30: 5,
    31: 5, 32: 1, 33: 3, 34: 2, 35: 1, 36: 3, 37: 5, 38: 3, 39: 2, 40: 3,
    41: 1, 42: 1, 43: 2, 44: 1, 45: 2, 46: 2, 47: 2, 48: 1, 49: 2, 50: 5,
    51: 3, 52: 2, 53: 3, 54: 1, 55: 3, 56: 1, 57: 5, 58: 4, 59: 2, 60: 3,
    61: 3, 62: 4, 63: 1, 64: 5, 65: 3, 66: 5, 67: 1, 68: 5, 69: 3, 70: 1,
    71: 4, 72: 1, 73: 1, 74: 2, 75: 4, 76: 2, 77: 3, 78: 3, 79: 4, 80: 1,
    81: 1, 82: 2, 83: 5, 84: 5, 85: 4, 86: 4, 87: 5, 88: 4, 89: 1, 90: 5,
    91: 5, 92: 4, 93: 1, 94: 1, 95: 1, 96: 3, 97: 3, 98: 3, 99: 3, 100: 4,
    101: 4, 102: 4, 103: 4, 104: 3, 105: 4, 106: 2, 107: 4, 108: 3, 109: 1,
    110: 4, 111: 5, 112: 5, 113: 1, 114: 1, 119: 4, 120: 3, 121: 4, 123: 4,
    126: 2, 128: 4, 129: 3, 131: 4, 135: 3, 136: 4, 137: 3, 138: 1,
}


def accuracy_at_k(probs, val_heroes, idx2target, k=5, ban_mask_np=None):
    if ban_mask_np is not None:
        probs = probs.copy()
        probs[ban_mask_np > 0] = 0.0
    hits = 0
    for i, hero in enumerate(val_heroes):
        top_k = [idx2target[j] for j in np.argsort(probs[i])[-k:][::-1]]
        if hero in top_k:
            hits += 1
    return hits / len(val_heroes)


def precompute_priors(df, captain_enc, team_enc, meta_arr):
    """Vectorized computation of prior top-10, entropy."""
    default = captain_enc["__default__"]
    N = len(df)
    cap_top = np.zeros((N, 10), dtype=np.float32)
    team_top = np.zeros((N, 10), dtype=np.float32)
    meta_top = np.zeros((N, 10), dtype=np.float32)
    cap_entropy = np.zeros((N, 1), dtype=np.float32)
    team_entropy = np.zeros((N, 1), dtype=np.float32)

    fp_captains = df["fp_captain"].values
    fp_teams = df["fp_team_id"].values

    for i in range(N):
        cap_dist = captain_enc.get(fp_captains[i], default)
        tid = fp_teams[i]
        team_dist = team_enc.get(tid, default) if pd.notna(tid) else default

        cap_top[i] = np.sort(cap_dist)[::-1][:10]
        team_top[i] = np.sort(team_dist)[::-1][:10]
        cap_entropy[i, 0] = -np.sum(cap_dist * np.log(cap_dist + 1e-8))
        team_entropy[i, 0] = -np.sum(team_dist * np.log(team_dist + 1e-8))

        if meta_arr.ndim == 2:
            meta_top[i] = np.sort(meta_arr[i])[::-1][:10]
        else:
            meta_top[i] = np.sort(meta_arr)[::-1][:10]

    return cap_top, team_top, meta_top, cap_entropy, team_entropy


def precompute_opponent_priors(df, captain_enc, team_enc):
    """Top-10 sorted priors for opponent captain and team."""
    default = captain_enc["__default__"]
    N = len(df)
    fp_team_vals = df["first_pick_team"].values
    rad_cap = df["radiant_captain"].values
    dire_cap = df["dire_captain"].values
    rad_team = df["radiant_team_id"].values
    dire_team = df["dire_team_id"].values

    opp_cap_top = np.zeros((N, 10), dtype=np.float32)
    opp_team_top = np.zeros((N, 10), dtype=np.float32)

    for i in range(N):
        opp_cap_id = dire_cap[i] if fp_team_vals[i] == 0 else rad_cap[i]
        opp_team_id = dire_team[i] if fp_team_vals[i] == 0 else rad_team[i]

        cap_dist = captain_enc.get(opp_cap_id, default)
        opp_cap_top[i] = np.sort(cap_dist)[::-1][:10]

        team_dist = team_enc.get(opp_team_id, default) if pd.notna(opp_team_id) else default
        opp_team_top[i] = np.sort(team_dist)[::-1][:10]

    return opp_cap_top, opp_team_top


def precompute_ban_overlap(df, captain_enc, team_enc, meta_arr, target2idx):
    """Overlap between banned heroes and captain/team/meta priors."""
    default = captain_enc["__default__"]
    N = len(df)
    feats = np.zeros((N, 6), dtype=np.float32)
    fp_captains = df["fp_captain"].values
    fp_teams = df["fp_team_id"].values

    for i in range(N):
        bans = df["pre_fp_bans"].iloc[i]
        cap_dist = captain_enc.get(fp_captains[i], default)
        tid = fp_teams[i]
        team_dist = team_enc.get(tid, default) if pd.notna(tid) else default
        meta_dist = meta_arr[i] if meta_arr.ndim == 2 else meta_arr

        cap_top10 = set(np.argsort(cap_dist)[-10:])
        team_top10 = set(np.argsort(team_dist)[-10:])
        meta_top10 = set(np.argsort(meta_dist)[-10:])

        ban_ids = set()
        cp, tp, mp = 0.0, 0.0, 0.0
        for h in bans:
            if h in target2idx:
                idx = target2idx[h]
                ban_ids.add(idx)
                cp += cap_dist[idx]
                tp += team_dist[idx]
                mp += meta_dist[idx]

        feats[i, 0] = len(ban_ids & cap_top10)
        feats[i, 1] = cp
        feats[i, 2] = len(ban_ids & team_top10)
        feats[i, 3] = tp
        feats[i, 4] = len(ban_ids & meta_top10)
        feats[i, 5] = mp

    return feats


def precompute_ban_side(df):
    """Features from which side made which bans (8 features)."""
    N = len(df)
    feats = np.zeros((N, 8), dtype=np.float32)
    fp_teams = df["first_pick_team"].values

    for i in range(N):
        actions = json.loads(df["picks_bans"].iloc[i])
        actions_sorted = sorted(actions, key=lambda x: x["order"])
        fp_t = fp_teams[i]

        for a in actions_sorted:
            if a["is_pick"]:
                break
            h = a["hero_id"]
            role = HERO_ROLES.get(h, 0)
            if a["team"] == fp_t:
                if role in (1, 2):
                    feats[i, 6] += 1
                elif role in (3, 4, 5):
                    feats[i, 7] += 1
                feats[i, 0] += 1
            else:
                if role == 1:
                    feats[i, 2] += 1
                elif role == 2:
                    feats[i, 3] += 1
                elif role == 3:
                    feats[i, 4] += 1
                elif role in (4, 5):
                    feats[i, 5] += 1
                feats[i, 1] += 1
    return feats


def precompute_hero_role_bans(df):
    """Count of banned heroes per role (5 features)."""
    N = len(df)
    feats = np.zeros((N, 5), dtype=np.float32)
    for i in range(N):
        for h in df["pre_fp_bans"].iloc[i]:
            role = HERO_ROLES.get(h, 0)
            if 1 <= role <= 5:
                feats[i, role - 1] += 1
    return feats


def precompute_pair_top10(df, train_df, target2idx, alpha=50.0):
    """Captain×Team pair target encoding, top-10 sorted values."""
    num_classes = len(target2idx)
    global_counts = train_df["first_pick_hero"].value_counts()
    global_dist = np.zeros(num_classes, dtype=np.float32)
    for h, cnt in global_counts.items():
        if h in target2idx:
            global_dist[target2idx[h]] = cnt
    global_dist /= global_dist.sum() + 1e-8

    pair_enc = {}
    for (cap, team), group in train_df.groupby(["fp_captain", "fp_team_id"]):
        counts = group["first_pick_hero"].value_counts()
        n = len(group)
        empirical = np.zeros(num_classes, dtype=np.float32)
        for h, cnt in counts.items():
            if h in target2idx:
                empirical[target2idx[h]] = cnt
        empirical /= n + 1e-8
        smoothed = (n * empirical + alpha * global_dist) / (n + alpha)
        pair_enc[(cap, team)] = smoothed

    N = len(df)
    pair_top = np.zeros((N, 10), dtype=np.float32)
    fp_captains = df["fp_captain"].values
    fp_teams = df["fp_team_id"].values

    for i in range(N):
        key = (fp_captains[i], fp_teams[i])
        dist = pair_enc.get(key, global_dist)
        pair_top[i] = np.sort(dist)[::-1][:10]

    return pair_top


def precompute_experience(df, train_df):
    """Log-scaled captain and team match counts (2 features)."""
    cap_counts = Counter(train_df["fp_captain"])
    team_counts = Counter(train_df["fp_team_id"].dropna())
    N = len(df)
    feats = np.zeros((N, 2), dtype=np.float32)
    fp_captains = df["fp_captain"].values
    fp_teams = df["fp_team_id"].values

    for i in range(N):
        feats[i, 0] = np.log1p(cap_counts.get(fp_captains[i], 0))
        tid = fp_teams[i]
        if pd.notna(tid):
            feats[i, 1] = np.log1p(team_counts.get(tid, 0))
    return feats


def main():
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dota2-first-pick")

    print("Loading data...", flush=True)
    df, hero2idx, idx2hero = parse_drafts(pd.read_csv(cfg["data"]["path"]))

    train_df = df.iloc[:-VAL_SIZE].copy()
    val_df = df.iloc[-VAL_SIZE:].copy()

    target_heroes = sorted(train_df["first_pick_hero"].unique())
    target2idx = {h: i for i, h in enumerate(target_heroes)}
    idx2target = {i: h for h, i in target2idx.items()}
    num_heroes = len(target2idx)

    train_game_pos, _ = build_series_features(train_df, target2idx)
    val_game_pos, _ = build_series_features(val_df, target2idx)

    val_heroes = val_df["first_pick_hero"].tolist()
    val_ban_mask = build_ban_mask(val_df, target2idx)
    y_train = train_df["first_pick_hero"].map(target2idx).values
    y_val = val_df["first_pick_hero"].map(target2idx).values

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  "
          f"Classes: {num_heroes}", flush=True)

    # ── Build encodings ──────────────────────────────────
    captain_enc = build_target_encoding(
        train_df, "fp_captain", target2idx,
        alpha=BEST_PARAMS["captain_alpha"])
    team_enc = build_target_encoding(
        train_df, "fp_team_id", target2idx,
        alpha=BEST_PARAMS["team_alpha"])
    train_meta = build_rolling_meta(
        train_df, target2idx, window=BEST_PARAMS["meta_window"])
    val_meta = build_rolling_meta(
        pd.concat([train_df, val_df], ignore_index=True),
        target2idx, window=BEST_PARAMS["meta_window"])[-VAL_SIZE:]

    # ── Pre-compute ALL feature blocks ───────────────────
    print("Pre-computing all feature blocks...", flush=True)
    t0 = time.time()

    # Base features (always used)
    train_ctx = build_context(train_df, game_positions=train_game_pos)
    val_ctx = build_context(val_df, game_positions=val_game_pos)

    train_ban_mh = np.zeros((len(train_df), len(hero2idx)), dtype=np.float32)
    for i, bans in enumerate(train_df["pre_fp_bans"]):
        for h in bans:
            if h in hero2idx:
                train_ban_mh[i, hero2idx[h]] = 1.0
    val_ban_mh = np.zeros((len(val_df), len(hero2idx)), dtype=np.float32)
    for i, bans in enumerate(val_df["pre_fp_bans"]):
        for h in bans:
            if h in hero2idx:
                val_ban_mh[i, hero2idx[h]] = 1.0

    print("  Priors...", flush=True)
    tr_cap_top, tr_team_top, tr_meta_top, tr_cap_H, tr_team_H = \
        precompute_priors(train_df, captain_enc, team_enc, train_meta)
    va_cap_top, va_team_top, va_meta_top, va_cap_H, va_team_H = \
        precompute_priors(val_df, captain_enc, team_enc, val_meta)

    print("  Opponent priors...", flush=True)
    tr_opp_cap, tr_opp_team = precompute_opponent_priors(
        train_df, captain_enc, team_enc)
    va_opp_cap, va_opp_team = precompute_opponent_priors(
        val_df, captain_enc, team_enc)

    print("  Ban overlap...", flush=True)
    tr_ban_ovlp = precompute_ban_overlap(
        train_df, captain_enc, team_enc, train_meta, target2idx)
    va_ban_ovlp = precompute_ban_overlap(
        val_df, captain_enc, team_enc, val_meta, target2idx)

    print("  Ban side...", flush=True)
    tr_ban_side = precompute_ban_side(train_df)
    va_ban_side = precompute_ban_side(val_df)

    print("  Hero roles...", flush=True)
    tr_roles = precompute_hero_role_bans(train_df)
    va_roles = precompute_hero_role_bans(val_df)

    print("  Captain-team pair...", flush=True)
    tr_pair = precompute_pair_top10(train_df, train_df, target2idx)
    va_pair = precompute_pair_top10(val_df, train_df, target2idx)

    print("  Experience...", flush=True)
    tr_exp = precompute_experience(train_df, train_df)
    va_exp = precompute_experience(val_df, train_df)

    print(f"  All features pre-computed in {time.time()-t0:.0f}s\n", flush=True)

    # ── Define feature blocks ────────────────────────────
    # Named tuples: (train_block, val_block)
    BLOCKS = {
        "ctx":        (train_ctx, val_ctx),
        "ban_mh":     (train_ban_mh, val_ban_mh),
        "cap_top":    (tr_cap_top, va_cap_top),
        "team_top":   (tr_team_top, va_team_top),
        "meta_top":   (tr_meta_top, va_meta_top),
        "cap_H":      (tr_cap_H, va_cap_H),
        "team_H":     (tr_team_H, va_team_H),
        "opp_cap":    (tr_opp_cap, va_opp_cap),
        "opp_team":   (tr_opp_team, va_opp_team),
        "ban_ovlp":   (tr_ban_ovlp, va_ban_ovlp),
        "ban_side":   (tr_ban_side, va_ban_side),
        "roles":      (tr_roles, va_roles),
        "pair":       (tr_pair, va_pair),
        "exp":        (tr_exp, va_exp),
    }

    # ── Ablation configs: list of block names to include ─
    BASE = ["ctx", "ban_mh", "cap_top", "team_top", "meta_top"]

    configs = [
        ("0_baseline",
         BASE + ["cap_H", "team_H"]),
        ("1_no_entropy",
         BASE),
        ("2_+opponent",
         BASE + ["opp_cap", "opp_team"]),
        ("3_+ban_overlap",
         BASE + ["opp_cap", "opp_team", "ban_ovlp"]),
        ("4_+ban_side",
         BASE + ["opp_cap", "opp_team", "ban_ovlp", "ban_side"]),
        ("5_+hero_roles",
         BASE + ["opp_cap", "opp_team", "ban_ovlp", "ban_side", "roles"]),
        ("6_+cap_team_pair",
         BASE + ["opp_cap", "opp_team", "ban_ovlp", "ban_side", "roles", "pair"]),
        ("7_+experience",
         BASE + ["opp_cap", "opp_team", "ban_ovlp", "ban_side", "roles", "pair", "exp"]),
        ("8_all_with_entropy",
         BASE + ["cap_H", "team_H", "opp_cap", "opp_team", "ban_ovlp",
                  "ban_side", "roles", "pair", "exp"]),
    ]

    lgb_params = {
        "objective": "multiclass",
        "num_class": num_heroes,
        "metric": "multi_logloss",
        "learning_rate": BEST_PARAMS["learning_rate"],
        "num_leaves": BEST_PARAMS["num_leaves"],
        "max_depth": BEST_PARAMS["max_depth"],
        "min_child_samples": BEST_PARAMS["min_child_samples"],
        "subsample": BEST_PARAMS["subsample"],
        "colsample_bytree": BEST_PARAMS["colsample_bytree"],
        "reg_alpha": BEST_PARAMS["reg_alpha"],
        "reg_lambda": BEST_PARAMS["reg_lambda"],
        "min_split_gain": BEST_PARAMS["min_split_gain"],
        "verbose": -1,
        "seed": SEED,
        "n_jobs": 4,
    }

    # ── Run ablation ─────────────────────────────────────
    sep = "=" * 90
    print(f"{sep}")
    print("ABLATION STUDY — Fixed best hyperparams, sequential feature addition")
    print(f"{sep}\n")

    results = []

    for name, block_names in configs:
        t0 = time.time()

        X_train = np.hstack([BLOCKS[b][0] for b in block_names])
        X_val = np.hstack([BLOCKS[b][1] for b in block_names])

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        evals_result = {}
        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )

        n_rounds = model.best_iteration if model.best_iteration > 0 else \
            model.num_trees() // num_heroes
        val_probs = model.predict(X_val, num_iteration=n_rounds)

        acc5 = accuracy_at_k(val_probs, val_heroes, idx2target, k=5,
                             ban_mask_np=val_ban_mask)
        acc3 = accuracy_at_k(val_probs, val_heroes, idx2target, k=3,
                             ban_mask_np=val_ban_mask)
        acc1 = accuracy_at_k(val_probs, val_heroes, idx2target, k=1,
                             ban_mask_np=val_ban_mask)

        val_loss_final = evals_result["val"]["multi_logloss"][-1] if \
            evals_result.get("val", {}).get("multi_logloss") else None
        val_loss_best = min(evals_result["val"]["multi_logloss"]) if \
            evals_result.get("val", {}).get("multi_logloss") else None

        elapsed = time.time() - t0

        result = {
            "config": name,
            "n_features": X_train.shape[1],
            "acc1": acc1, "acc3": acc3, "acc5": acc5,
            "val_loss": val_loss_best,
            "n_rounds": n_rounds,
            "time_s": round(elapsed, 1),
        }
        results.append(result)

        delta = ""
        if len(results) > 1:
            d = acc5 - results[0]["acc5"]
            delta = f"  Δbase={d:+.3f}"

        print(f"  {name:25s}  feats={X_train.shape[1]:4d}  "
              f"@1={acc1:.3f}  @3={acc3:.3f}  @5={acc5:.3f}  "
              f"loss={val_loss_best:.4f}  rounds={n_rounds:3d}  "
              f"({elapsed:.1f}s){delta}", flush=True)

        with mlflow.start_run(run_name=f"ablation_{name}"):
            mlflow.log_params({
                "config": name,
                "n_features": X_train.shape[1],
                "blocks": ",".join(block_names),
            })
            mlflow.log_metric("accuracy_at_1", acc1)
            mlflow.log_metric("accuracy_at_3", acc3)
            mlflow.log_metric("accuracy_at_5", acc5)
            if val_loss_best is not None:
                mlflow.log_metric("val_logloss", val_loss_best)
            mlflow.log_metric("n_boost_rounds", n_rounds)
            mlflow.set_tag("model_type", "lgbm_ablation")
            mlflow.set_tag("ablation_config", name)

        del model, dtrain, dval, evals_result
        gc.collect()

    # ── Summary ──────────────────────────────────────────
    print(f"\n{sep}")
    print("ABLATION RESULTS SUMMARY")
    print(sep)

    results_df = pd.DataFrame(results)
    baseline_acc5 = results_df.iloc[0]["acc5"]
    results_df["delta_vs_baseline"] = results_df["acc5"] - baseline_acc5
    results_df["delta_vs_prev"] = results_df["acc5"].diff().fillna(0)

    print(results_df.to_string(index=False, float_format="%.4f"))

    results_df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'ablation_results.csv'}")

    best_idx = results_df["acc5"].idxmax()
    best_row = results_df.iloc[best_idx]
    print(f"\nBest config: {best_row['config']}  "
          f"Acc@5={best_row['acc5']:.3f}  "
          f"(Δ from baseline={best_row['delta_vs_baseline']:+.3f})")

    print(f"\nStep-by-step impact:")
    for _, row in results_df.iterrows():
        d = row["delta_vs_baseline"]
        icon = "+" if d > 0 else ("-" if d < 0 else "=")
        print(f"  {icon} {row['config']:25s}  "
              f"@5={row['acc5']:.3f}  Δbase={d:+.4f}  "
              f"feats={int(row['n_features'])}")

    # ── Plot ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        names_short = [r["config"].split("_", 1)[1] for r in results]
        acc5s = [r["acc5"] for r in results]
        colors = ["#55A868" if r["acc5"] > results[0]["acc5"]
                  else "#C44E52" if r["acc5"] < results[0]["acc5"]
                  else "#4C72B0" for r in results]

        axes[0].barh(range(len(names_short)), acc5s, color=colors)
        axes[0].set_yticks(range(len(names_short)))
        axes[0].set_yticklabels(names_short, fontsize=9)
        axes[0].axvline(x=results[0]["acc5"], color="gray", linestyle="--",
                        alpha=0.7, label=f"Baseline ({results[0]['acc5']:.3f})")
        axes[0].set_xlabel("Acc@5")
        axes[0].set_title("Ablation: Acc@5 by Configuration")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        deltas = results_df["delta_vs_baseline"].values
        dcolors = ["#55A868" if d > 0 else "#C44E52" if d < 0 else "#999999"
                   for d in deltas]
        axes[1].bar(range(len(names_short)), deltas, color=dcolors)
        axes[1].set_xticks(range(len(names_short)))
        axes[1].set_xticklabels(names_short, fontsize=7, rotation=45, ha="right")
        axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        axes[1].set_ylabel("Δ Acc@5 vs Baseline")
        axes[1].set_title("Delta vs Baseline")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "ablation_results.png", dpi=150,
                    bbox_inches="tight")
        print(f"\nPlot saved to {RESULTS_DIR / 'ablation_results.png'}")
        plt.close()
    except ImportError:
        print("matplotlib not installed, skipping plots")

    print(f"\n{sep}")
    print("ABLATION STUDY COMPLETE")
    print(sep, flush=True)


if __name__ == "__main__":
    main()
