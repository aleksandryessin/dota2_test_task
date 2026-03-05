#!/usr/bin/env python3
"""Quick temporal 5-fold CV to estimate noise on baseline Acc@5."""

import os, sys, gc, time
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
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
SEED = 42
FOLD_SIZE = 200
N_FOLDS = 5


def accuracy_at_k(probs, heroes, idx2target, k=5, ban_mask_np=None):
    if ban_mask_np is not None:
        probs = probs.copy()
        probs[ban_mask_np > 0] = 0.0
    hits = 0
    for i, hero in enumerate(heroes):
        if hero not in idx2target.values():
            continue
        top_k = [idx2target[j] for j in np.argsort(probs[i])[-k:][::-1]]
        if hero in top_k:
            hits += 1
    return hits / len(heroes)


def build_fold_features(train_df, val_df, hero2idx, target2idx):
    """Build baseline features for one fold."""
    cap_enc = build_target_encoding(
        train_df, "fp_captain", target2idx, alpha=BEST_PARAMS["captain_alpha"])
    team_enc = build_target_encoding(
        train_df, "fp_team_id", target2idx, alpha=BEST_PARAMS["team_alpha"])
    train_meta = build_rolling_meta(train_df, target2idx, window=BEST_PARAMS["meta_window"])
    val_meta = build_rolling_meta(
        pd.concat([train_df, val_df], ignore_index=True),
        target2idx, window=BEST_PARAMS["meta_window"])[-len(val_df):]

    tr_gp, _ = build_series_features(train_df, target2idx)
    va_gp, _ = build_series_features(val_df, target2idx)

    default = cap_enc["__default__"]

    def make_features(df, meta_arr, game_pos):
        ctx = build_context(df, game_positions=game_pos)
        ban_mh = np.zeros((len(df), len(hero2idx)), dtype=np.float32)
        for i, bans in enumerate(df["pre_fp_bans"]):
            for h in bans:
                if h in hero2idx:
                    ban_mh[i, hero2idx[h]] = 1.0
        N = len(df)
        cap_top = np.zeros((N, 10), dtype=np.float32)
        team_top = np.zeros((N, 10), dtype=np.float32)
        meta_top = np.zeros((N, 10), dtype=np.float32)
        cap_H = np.zeros((N, 1), dtype=np.float32)
        team_H = np.zeros((N, 1), dtype=np.float32)
        caps = df["fp_captain"].values
        teams = df["fp_team_id"].values
        for i in range(N):
            cd = cap_enc.get(caps[i], default)
            tid = teams[i]
            td = team_enc.get(tid, default) if pd.notna(tid) else default
            cap_top[i] = np.sort(cd)[::-1][:10]
            team_top[i] = np.sort(td)[::-1][:10]
            cap_H[i, 0] = -np.sum(cd * np.log(cd + 1e-8))
            team_H[i, 0] = -np.sum(td * np.log(td + 1e-8))
            meta_top[i] = np.sort(meta_arr[i])[::-1][:10]
        return np.hstack([ctx, ban_mh, cap_top, team_top, meta_top, cap_H, team_H])

    X_tr = make_features(train_df, train_meta, tr_gp)
    X_va = make_features(val_df, val_meta, va_gp)
    return X_tr, X_va


def main():
    np.random.seed(SEED)
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...", flush=True)
    df, hero2idx, _ = parse_drafts(pd.read_csv(cfg["data"]["path"]))
    n_total = len(df)

    print(f"Total samples: {n_total:,}")
    print(f"Running {N_FOLDS}-fold temporal CV (fold_size={FOLD_SIZE})\n")

    cv_results = []
    for fold in range(N_FOLDS):
        val_end = n_total - fold * FOLD_SIZE
        val_start = val_end - FOLD_SIZE
        if val_start < 5000:
            break

        t0 = time.time()
        train_df = df.iloc[:val_start].copy()
        val_df = df.iloc[val_start:val_end].copy()

        target_heroes = sorted(train_df["first_pick_hero"].unique())
        t2i = {h: i for i, h in enumerate(target_heroes)}
        i2t = {i: h for h, i in t2i.items()}
        n_classes = len(t2i)

        X_tr, X_va = build_fold_features(train_df, val_df, hero2idx, t2i)

        y_tr = train_df["first_pick_hero"].map(t2i).values
        y_va_raw = val_df["first_pick_hero"].map(t2i)
        valid = y_va_raw.notna()
        y_va = y_va_raw[valid].astype(int).values
        X_va = X_va[valid.values]

        val_heroes = val_df.loc[valid.values, "first_pick_hero"].tolist()
        ban_mask = build_ban_mask(val_df[valid.values], t2i)

        lgb_params = {
            "objective": "multiclass", "num_class": n_classes,
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
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        evals_result = {}
        model = lgb.train(
            lgb_params, dtrain, num_boost_round=600,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )
        n_rounds = model.best_iteration if model.best_iteration > 0 else \
            model.num_trees() // n_classes
        probs = model.predict(X_va, num_iteration=n_rounds)

        a5 = accuracy_at_k(probs, val_heroes, i2t, k=5, ban_mask_np=ban_mask)
        a3 = accuracy_at_k(probs, val_heroes, i2t, k=3, ban_mask_np=ban_mask)
        a1 = accuracy_at_k(probs, val_heroes, i2t, k=1, ban_mask_np=ban_mask)

        elapsed = time.time() - t0
        cv_results.append({"fold": fold, "val_range": f"{val_start}:{val_end}",
                           "n_valid": len(y_va), "acc1": a1, "acc3": a3,
                           "acc5": a5, "rounds": n_rounds, "time_s": round(elapsed, 1)})

        print(f"  Fold {fold}: [{val_start}:{val_end}]  valid={len(y_va)}  "
              f"@1={a1:.3f}  @3={a3:.3f}  @5={a5:.3f}  "
              f"rnd={n_rounds}  ({elapsed:.0f}s)", flush=True)

        del model, dtrain, dval
        gc.collect()

    accs5 = [r["acc5"] for r in cv_results]
    accs3 = [r["acc3"] for r in cv_results]
    accs1 = [r["acc1"] for r in cv_results]

    print(f"\n{'='*70}")
    print(f"CV Results ({len(cv_results)} folds):")
    print(f"  Acc@1: {np.mean(accs1):.3f} ± {np.std(accs1):.3f}  "
          f"(range: {min(accs1):.3f} - {max(accs1):.3f})")
    print(f"  Acc@3: {np.mean(accs3):.3f} ± {np.std(accs3):.3f}  "
          f"(range: {min(accs3):.3f} - {max(accs3):.3f})")
    print(f"  Acc@5: {np.mean(accs5):.3f} ± {np.std(accs5):.3f}  "
          f"(range: {min(accs5):.3f} - {max(accs5):.3f})")
    print(f"\n  Significance threshold (2σ): ±{2*np.std(accs5):.3f}")
    print(f"  Any feature change < {2*np.std(accs5):.3f} is likely noise")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
