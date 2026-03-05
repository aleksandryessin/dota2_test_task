#!/usr/bin/env python3
"""Ablation V2: diagnostic experiments.

Tests:
  A. Each new feature group INDIVIDUALLY added to baseline (not cumulative)
  B. Wider model (more leaves, deeper) with new features
  C. Ban compression: replace 127-dim multi-hot with top-K + role counts
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
                feats[i, 0] += 1
                if role in (1, 2):
                    feats[i, 6] += 1
                elif role in (3, 4, 5):
                    feats[i, 7] += 1
            else:
                feats[i, 1] += 1
                if role == 1:
                    feats[i, 2] += 1
                elif role == 2:
                    feats[i, 3] += 1
                elif role == 3:
                    feats[i, 4] += 1
                elif role in (4, 5):
                    feats[i, 5] += 1
    return feats


def precompute_hero_role_bans(df):
    N = len(df)
    feats = np.zeros((N, 5), dtype=np.float32)
    for i in range(N):
        for h in df["pre_fp_bans"].iloc[i]:
            role = HERO_ROLES.get(h, 0)
            if 1 <= role <= 5:
                feats[i, role - 1] += 1
    return feats


def precompute_pair_top10(df, train_df, target2idx, alpha=50.0):
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


def compress_bans(df, hero2idx, importance_arr, top_k=30):
    """Replace 127-dim multi-hot with top-K most important bans + role counts."""
    if importance_arr is None:
        sorted_heroes = list(range(len(hero2idx)))
    else:
        ctx_dim = 9
        ban_start = ctx_dim
        ban_imp = importance_arr[ban_start:ban_start + len(hero2idx)]
        sorted_heroes = np.argsort(ban_imp)[::-1][:top_k].tolist()

    N = len(df)
    ban_topk = np.zeros((N, top_k), dtype=np.float32)
    ban_roles = np.zeros((N, 5), dtype=np.float32)
    ban_count = np.zeros((N, 1), dtype=np.float32)

    hero_to_col = {list(hero2idx.keys())[list(hero2idx.values()).index(h)]: j
                   for j, h in enumerate(sorted_heroes)}

    for i in range(N):
        bans = df["pre_fp_bans"].iloc[i]
        ban_count[i, 0] = len(bans)
        for h in bans:
            if h in hero2idx:
                hidx = hero2idx[h]
                if hidx in [sorted_heroes[j] for j in range(top_k)]:
                    col = sorted_heroes.index(hidx)
                    ban_topk[i, col] = 1.0
            role = HERO_ROLES.get(h, 0)
            if 1 <= role <= 5:
                ban_roles[i, role - 1] += 1

    return np.hstack([ban_topk, ban_roles, ban_count])


def train_and_eval(X_train, X_val, y_train, y_val, val_heroes, idx2target,
                   val_ban_mask, num_heroes, lgb_params_override=None):
    """Train LightGBM, return metrics."""
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
    if lgb_params_override:
        lgb_params.update(lgb_params_override)

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
    val_loss = min(evals_result["val"]["multi_logloss"]) if \
        evals_result.get("val", {}).get("multi_logloss") else None
    importance = model.feature_importance(importance_type="gain")
    del model, dtrain, dval, evals_result
    gc.collect()
    return acc1, acc3, acc5, val_loss, n_rounds, importance


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

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Classes: {num_heroes}")

    captain_enc = build_target_encoding(
        train_df, "fp_captain", target2idx, alpha=BEST_PARAMS["captain_alpha"])
    team_enc = build_target_encoding(
        train_df, "fp_team_id", target2idx, alpha=BEST_PARAMS["team_alpha"])
    train_meta = build_rolling_meta(train_df, target2idx, window=BEST_PARAMS["meta_window"])
    val_meta = build_rolling_meta(
        pd.concat([train_df, val_df], ignore_index=True),
        target2idx, window=BEST_PARAMS["meta_window"])[-VAL_SIZE:]

    print("Pre-computing features...", flush=True)
    t0 = time.time()
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

    tr_cap_top, tr_team_top, tr_meta_top, tr_cap_H, tr_team_H = \
        precompute_priors(train_df, captain_enc, team_enc, train_meta)
    va_cap_top, va_team_top, va_meta_top, va_cap_H, va_team_H = \
        precompute_priors(val_df, captain_enc, team_enc, val_meta)

    tr_opp_cap, tr_opp_team = precompute_opponent_priors(train_df, captain_enc, team_enc)
    va_opp_cap, va_opp_team = precompute_opponent_priors(val_df, captain_enc, team_enc)

    tr_ban_ovlp = precompute_ban_overlap(train_df, captain_enc, team_enc, train_meta, target2idx)
    va_ban_ovlp = precompute_ban_overlap(val_df, captain_enc, team_enc, val_meta, target2idx)

    tr_ban_side = precompute_ban_side(train_df)
    va_ban_side = precompute_ban_side(val_df)

    tr_roles = precompute_hero_role_bans(train_df)
    va_roles = precompute_hero_role_bans(val_df)

    tr_pair = precompute_pair_top10(train_df, train_df, target2idx)
    va_pair = precompute_pair_top10(val_df, train_df, target2idx)

    tr_exp = precompute_experience(train_df, train_df)
    va_exp = precompute_experience(val_df, train_df)

    print(f"  Done in {time.time()-t0:.0f}s\n", flush=True)

    BLOCKS = {
        "ctx":      (train_ctx, val_ctx),
        "ban_mh":   (train_ban_mh, val_ban_mh),
        "cap_top":  (tr_cap_top, va_cap_top),
        "team_top": (tr_team_top, va_team_top),
        "meta_top": (tr_meta_top, va_meta_top),
        "cap_H":    (tr_cap_H, va_cap_H),
        "team_H":   (tr_team_H, va_team_H),
        "opp_cap":  (tr_opp_cap, va_opp_cap),
        "opp_team": (tr_opp_team, va_opp_team),
        "ban_ovlp": (tr_ban_ovlp, va_ban_ovlp),
        "ban_side": (tr_ban_side, va_ban_side),
        "roles":    (tr_roles, va_roles),
        "pair":     (tr_pair, va_pair),
        "exp":      (tr_exp, va_exp),
    }

    BASE = ["ctx", "ban_mh", "cap_top", "team_top", "meta_top", "cap_H", "team_H"]

    def assemble(block_names):
        tr = np.hstack([BLOCKS[b][0] for b in block_names])
        va = np.hstack([BLOCKS[b][1] for b in block_names])
        return tr, va

    sep = "=" * 95
    all_results = []

    def run_config(name, block_names, lgb_override=None):
        t0 = time.time()
        X_tr, X_va = assemble(block_names)
        acc1, acc3, acc5, vloss, nrnd, imp = train_and_eval(
            X_tr, X_va, y_train, y_val, val_heroes, idx2target,
            val_ban_mask, num_heroes, lgb_override)
        elapsed = time.time() - t0
        r = {"config": name, "n_features": X_tr.shape[1],
             "acc1": acc1, "acc3": acc3, "acc5": acc5,
             "val_loss": vloss, "n_rounds": nrnd, "time_s": round(elapsed, 1)}
        all_results.append(r)
        print(f"  {name:35s}  feats={X_tr.shape[1]:4d}  "
              f"@1={acc1:.3f}  @3={acc3:.3f}  @5={acc5:.3f}  "
              f"loss={vloss:.4f}  rnd={nrnd:3d}  ({elapsed:.0f}s)", flush=True)
        return imp

    # ══════════════════════════════════════════════════════
    # PART A: Individual feature additions (original params)
    # ══════════════════════════════════════════════════════
    print(f"{sep}")
    print("PART A: Individual feature groups added to baseline (original params)")
    print(f"{sep}\n")

    baseline_imp = run_config("A0_baseline", BASE)
    run_config("A1_+opponent_only",    BASE + ["opp_cap", "opp_team"])
    run_config("A2_+ban_overlap_only", BASE + ["ban_ovlp"])
    run_config("A3_+ban_side_only",    BASE + ["ban_side"])
    run_config("A4_+hero_roles_only",  BASE + ["roles"])
    run_config("A5_+pair_only",        BASE + ["pair"])
    run_config("A6_+experience_only",  BASE + ["exp"])

    # ══════════════════════════════════════════════════════
    # PART B: Same configs but with wider model
    # ══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("PART B: Wider model (num_leaves=63, max_depth=8, colsample=0.7)")
    print(f"{sep}\n")

    wider = {"num_leaves": 63, "max_depth": 8, "colsample_bytree": 0.7}
    run_config("B0_baseline_wider", BASE, wider)
    run_config("B1_+opponent_wider",    BASE + ["opp_cap", "opp_team"], wider)
    run_config("B2_+ban_overlap_wider", BASE + ["ban_ovlp"], wider)
    run_config("B3_+ban_side_wider",    BASE + ["ban_side"], wider)
    run_config("B4_+hero_roles_wider",  BASE + ["roles"], wider)
    run_config("B5_best_combo_wider",
               BASE + ["opp_cap", "opp_team", "ban_ovlp", "roles"], wider)

    # ══════════════════════════════════════════════════════
    # PART C: Ban compression
    # ══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("PART C: Ban compression (top-K bans + role counts instead of 127-dim)")
    print(f"{sep}\n")

    for k in [20, 30, 50]:
        tr_ban_comp = compress_bans(train_df, hero2idx, baseline_imp, top_k=k)
        va_ban_comp = compress_bans(val_df, hero2idx, baseline_imp, top_k=k)
        BLOCKS[f"ban_comp_{k}"] = (tr_ban_comp, va_ban_comp)
        compressed = ["ctx", f"ban_comp_{k}", "cap_top", "team_top",
                      "meta_top", "cap_H", "team_H"]
        run_config(f"C_ban_top{k}+roles", compressed)

    # ══════════════════════════════════════════════════════
    # PART D: Cross-validation on baseline (noise check)
    # ══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("PART D: Temporal 5-fold CV on baseline (noise estimation)")
    print(f"{sep}\n")

    full_y = df["first_pick_hero"].values
    full_game_pos, _ = build_series_features(df, target2idx)
    full_ctx = build_context(df, game_positions=full_game_pos)
    full_ban_mh = np.zeros((len(df), len(hero2idx)), dtype=np.float32)
    for i, bans in enumerate(df["pre_fp_bans"]):
        for h in bans:
            if h in hero2idx:
                full_ban_mh[i, hero2idx[h]] = 1.0
    full_meta = build_rolling_meta(df, target2idx, window=BEST_PARAMS["meta_window"])

    n_total = len(df)
    fold_size = 200
    n_folds = 5
    cv_accs = []

    for fold in range(n_folds):
        val_end = n_total - fold * fold_size
        val_start = val_end - fold_size
        if val_start < 1000:
            break

        cv_train = df.iloc[:val_start].copy()
        cv_val = df.iloc[val_start:val_end].copy()

        cv_target_heroes = sorted(cv_train["first_pick_hero"].unique())
        cv_t2i = {h: i for i, h in enumerate(cv_target_heroes)}
        cv_i2t = {i: h for h, i in cv_t2i.items()}
        cv_nheroes = len(cv_t2i)

        cv_cap_enc = build_target_encoding(
            cv_train, "fp_captain", cv_t2i, alpha=BEST_PARAMS["captain_alpha"])
        cv_team_enc = build_target_encoding(
            cv_train, "fp_team_id", cv_t2i, alpha=BEST_PARAMS["team_alpha"])

        cv_train_meta = build_rolling_meta(cv_train, cv_t2i, window=BEST_PARAMS["meta_window"])
        cv_val_meta = build_rolling_meta(
            pd.concat([cv_train, cv_val], ignore_index=True),
            cv_t2i, window=BEST_PARAMS["meta_window"])[-fold_size:]

        cv_train_gp, _ = build_series_features(cv_train, cv_t2i)
        cv_val_gp, _ = build_series_features(cv_val, cv_t2i)

        cv_tr_ctx = build_context(cv_train, game_positions=cv_train_gp)
        cv_va_ctx = build_context(cv_val, game_positions=cv_val_gp)

        cv_tr_ban = np.zeros((len(cv_train), len(hero2idx)), dtype=np.float32)
        for i, bans in enumerate(cv_train["pre_fp_bans"]):
            for h in bans:
                if h in hero2idx:
                    cv_tr_ban[i, hero2idx[h]] = 1.0
        cv_va_ban = np.zeros((len(cv_val), len(hero2idx)), dtype=np.float32)
        for i, bans in enumerate(cv_val["pre_fp_bans"]):
            for h in bans:
                if h in hero2idx:
                    cv_va_ban[i, hero2idx[h]] = 1.0

        cv_default = cv_cap_enc["__default__"]
        cv_tr_cap = np.zeros((len(cv_train), 10), dtype=np.float32)
        cv_tr_team = np.zeros((len(cv_train), 10), dtype=np.float32)
        cv_tr_meta = np.zeros((len(cv_train), 10), dtype=np.float32)
        cv_tr_cH = np.zeros((len(cv_train), 1), dtype=np.float32)
        cv_tr_tH = np.zeros((len(cv_train), 1), dtype=np.float32)
        for i in range(len(cv_train)):
            cdist = cv_cap_enc.get(cv_train["fp_captain"].iloc[i], cv_default)
            tid = cv_train["fp_team_id"].iloc[i]
            tdist = cv_team_enc.get(tid, cv_default) if pd.notna(tid) else cv_default
            cv_tr_cap[i] = np.sort(cdist)[::-1][:10]
            cv_tr_team[i] = np.sort(tdist)[::-1][:10]
            cv_tr_cH[i, 0] = -np.sum(cdist * np.log(cdist + 1e-8))
            cv_tr_tH[i, 0] = -np.sum(tdist * np.log(tdist + 1e-8))
            cv_tr_meta[i] = np.sort(cv_train_meta[i])[::-1][:10]

        cv_va_cap = np.zeros((fold_size, 10), dtype=np.float32)
        cv_va_team = np.zeros((fold_size, 10), dtype=np.float32)
        cv_va_meta_top = np.zeros((fold_size, 10), dtype=np.float32)
        cv_va_cH = np.zeros((fold_size, 1), dtype=np.float32)
        cv_va_tH = np.zeros((fold_size, 1), dtype=np.float32)
        for i in range(fold_size):
            cdist = cv_cap_enc.get(cv_val["fp_captain"].iloc[i], cv_default)
            tid = cv_val["fp_team_id"].iloc[i]
            tdist = cv_team_enc.get(tid, cv_default) if pd.notna(tid) else cv_default
            cv_va_cap[i] = np.sort(cdist)[::-1][:10]
            cv_va_team[i] = np.sort(tdist)[::-1][:10]
            cv_va_cH[i, 0] = -np.sum(cdist * np.log(cdist + 1e-8))
            cv_va_tH[i, 0] = -np.sum(tdist * np.log(tdist + 1e-8))
            cv_va_meta_top[i] = np.sort(cv_val_meta[i])[::-1][:10]

        X_tr = np.hstack([cv_tr_ctx, cv_tr_ban, cv_tr_cap, cv_tr_team,
                          cv_tr_meta, cv_tr_cH, cv_tr_tH])
        X_va = np.hstack([cv_va_ctx, cv_va_ban, cv_va_cap, cv_va_team,
                          cv_va_meta_top, cv_va_cH, cv_va_tH])

        y_tr = cv_train["first_pick_hero"].map(cv_t2i).values
        y_va = cv_val["first_pick_hero"].map(cv_t2i).values

        # Drop unmapped val heroes
        valid_mask = ~pd.isna(pd.Series(y_va))
        if valid_mask.sum() < fold_size:
            y_va_clean = y_va[valid_mask]
            X_va = X_va[valid_mask]
        else:
            y_va_clean = y_va

        cv_val_heroes = cv_val["first_pick_hero"].tolist()
        cv_ban_mask = build_ban_mask(cv_val, cv_t2i)

        lgb_params = {
            "objective": "multiclass", "num_class": cv_nheroes,
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
            "verbose": -1, "seed": SEED, "n_jobs": 4,
        }

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval_ds = lgb.Dataset(X_va, label=y_va_clean, reference=dtrain)
        evals_result = {}
        model = lgb.train(
            lgb_params, dtrain, num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[dtrain, dval_ds], valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )
        n_rounds = model.best_iteration if model.best_iteration > 0 else \
            model.num_trees() // cv_nheroes
        val_probs = model.predict(X_va, num_iteration=n_rounds)
        a5 = accuracy_at_k(val_probs, cv_val_heroes, cv_i2t, k=5,
                           ban_mask_np=cv_ban_mask)
        cv_accs.append(a5)
        print(f"  Fold {fold}: val[{val_start}:{val_end}]  "
              f"Acc@5={a5:.3f}  rounds={n_rounds}", flush=True)
        del model, dtrain, dval_ds
        gc.collect()

    if cv_accs:
        mean_acc = np.mean(cv_accs)
        std_acc = np.std(cv_accs)
        print(f"\n  CV Acc@5: {mean_acc:.3f} ± {std_acc:.3f}  "
              f"(range: {min(cv_accs):.3f} - {max(cv_accs):.3f})")

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("FULL SUMMARY")
    print(sep)

    results_df = pd.DataFrame(all_results)
    baseline_acc = all_results[0]["acc5"]
    results_df["delta"] = results_df["acc5"] - baseline_acc
    print(results_df.to_string(index=False, float_format="%.4f"))
    results_df.to_csv(RESULTS_DIR / "ablation_v2_results.csv", index=False)

    best_idx = results_df["acc5"].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\nBest overall: {best['config']}  Acc@5={best['acc5']:.3f}")

    if cv_accs:
        print(f"CV noise estimate: ±{std_acc:.3f} (1 std)")
        print(f"Meaningful improvement threshold: >{2*std_acc:.3f}")

    # ── Plot ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 8))
        names_short = [r["config"] for r in all_results]
        acc5s = [r["acc5"] for r in all_results]
        colors = ["#55A868" if a > baseline_acc else "#C44E52" if a < baseline_acc
                  else "#4C72B0" for a in acc5s]
        bars = ax.barh(range(len(names_short)), acc5s, color=colors)
        ax.set_yticks(range(len(names_short)))
        ax.set_yticklabels(names_short, fontsize=8)
        ax.axvline(x=baseline_acc, color="gray", linestyle="--", alpha=0.7,
                   label=f"Baseline ({baseline_acc:.3f})")
        if cv_accs:
            ax.axvspan(baseline_acc - std_acc, baseline_acc + std_acc,
                       alpha=0.15, color="gray", label=f"±1σ noise ({std_acc:.3f})")
        ax.set_xlabel("Acc@5")
        ax.set_title("Ablation V2: All Configurations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "ablation_v2_results.png", dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {RESULTS_DIR / 'ablation_v2_results.png'}")
        plt.close()
    except ImportError:
        pass

    print(f"\n{sep}")
    print("ABLATION V2 COMPLETE")
    print(sep, flush=True)


if __name__ == "__main__":
    main()
