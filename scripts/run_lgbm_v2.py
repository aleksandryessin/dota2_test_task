#!/usr/bin/env python3
"""LightGBM with V2 features for ensemble baseline."""

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features_v2 import (
    parse_drafts, build_target_encoding, build_rolling_meta,
    build_ban_sequences, build_context, build_ban_mask,
    build_series_features,
    CONTEXT_SCALAR_DIM, MAX_BAN_SEQ_LEN,
)

VAL_SIZE = 200
SEED = 42


def build_lgbm_features(df, hero2idx, target2idx, captain_enc, team_enc,
                        meta_arr, game_pos, num_heroes):
    """Build flat feature matrix for LightGBM from V2 features."""
    ctx = build_context(df, game_positions=game_pos)

    ban_multihot = np.zeros((len(df), len(hero2idx)), dtype=np.float32)
    for i, bans in enumerate(df["pre_fp_bans"]):
        for h in bans:
            if h in hero2idx:
                ban_multihot[i, hero2idx[h]] = 1.0

    default = captain_enc["__default__"]
    cap_top = np.zeros((len(df), 10), dtype=np.float32)
    team_top = np.zeros((len(df), 10), dtype=np.float32)
    meta_top = np.zeros((len(df), 10), dtype=np.float32)
    cap_entropy = np.zeros(len(df), dtype=np.float32)
    team_entropy = np.zeros(len(df), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        cap_dist = captain_enc.get(row["fp_captain"], default)
        team_dist = team_enc.get(row.get("fp_team_id", np.nan), default)
        if pd.isna(row.get("fp_team_id", np.nan)):
            team_dist = default

        cap_sorted = np.sort(cap_dist)[::-1]
        cap_top[i] = cap_sorted[:10]
        team_sorted = np.sort(team_dist)[::-1]
        team_top[i] = team_sorted[:10]

        cap_entropy[i] = -np.sum(cap_dist * np.log(cap_dist + 1e-8))
        team_entropy[i] = -np.sum(team_dist * np.log(team_dist + 1e-8))

        if meta_arr.ndim == 2:
            meta_sorted = np.sort(meta_arr[i])[::-1]
        else:
            meta_sorted = np.sort(meta_arr)[::-1]
        meta_top[i] = meta_sorted[:10]

    features = np.hstack([
        ctx,
        ban_multihot,
        cap_top, team_top, meta_top,
        cap_entropy.reshape(-1, 1),
        team_entropy.reshape(-1, 1),
    ])
    return features


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


def main():
    np.random.seed(SEED)

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dota2-first-pick")

    print("Loading data...", flush=True)
    df, hero2idx, idx2hero = parse_drafts(pd.read_csv(config["data"]["path"]))

    train_df = df.iloc[:-VAL_SIZE].copy()
    val_df = df.iloc[-VAL_SIZE:].copy()

    target_heroes = sorted(train_df["first_pick_hero"].unique())
    target2idx = {h: i for i, h in enumerate(target_heroes)}
    idx2target = {i: h for h, i in target2idx.items()}
    num_heroes = len(target2idx)

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Classes: {num_heroes}", flush=True)

    captain_enc = build_target_encoding(train_df, "fp_captain", target2idx, alpha=20.0)
    team_enc = build_target_encoding(train_df, "fp_team_id", target2idx, alpha=30.0)
    train_meta_arr = build_rolling_meta(train_df, target2idx, window=5000)
    val_meta_arr = build_rolling_meta(
        pd.concat([train_df, val_df], ignore_index=True), target2idx, window=5000
    )[-VAL_SIZE:]

    train_game_pos, _ = build_series_features(train_df, target2idx)
    val_game_pos, _ = build_series_features(val_df, target2idx)

    print("Building features...", flush=True)
    X_train = build_lgbm_features(
        train_df, hero2idx, target2idx, captain_enc, team_enc,
        train_meta_arr, train_game_pos, num_heroes)
    y_train = train_df["first_pick_hero"].map(target2idx).values

    X_val = build_lgbm_features(
        val_df, hero2idx, target2idx, captain_enc, team_enc,
        val_meta_arr, val_game_pos, num_heroes)
    val_heroes = val_df["first_pick_hero"].tolist()
    val_ban_mask = build_ban_mask(val_df, target2idx)

    print(f"Feature shape: {X_train.shape}", flush=True)

    params = {
        "objective": "multiclass",
        "num_class": num_heroes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": SEED,
        "n_jobs": -1,
    }

    print("Training LightGBM...", flush=True)
    t0 = time.time()

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=None, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=300,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(50)],
    )

    train_sec = time.time() - t0
    print(f"Training time: {train_sec:.1f}s", flush=True)

    val_probs = model.predict(X_val)

    acc5 = accuracy_at_k(val_probs, val_heroes, idx2target, k=5, ban_mask_np=val_ban_mask)
    acc3 = accuracy_at_k(val_probs, val_heroes, idx2target, k=3, ban_mask_np=val_ban_mask)
    acc1 = accuracy_at_k(val_probs, val_heroes, idx2target, k=1, ban_mask_np=val_ban_mask)

    print(f"\n{'=' * 60}")
    print(f"LightGBM V2 Results:")
    print(f"  acc@1={acc1:.3f}  acc@3={acc3:.3f}  acc@5={acc5:.3f}")
    print(f"  Train time: {train_sec:.1f}s")
    print(f"{'=' * 60}", flush=True)

    np.save("lgbm_val_probs.npy", val_probs)
    print("Saved val probabilities to lgbm_val_probs.npy")

    np.save("lgbm_train_probs.npy", model.predict(X_train))

    with mlflow.start_run(run_name="lgbm_v2_features"):
        mlflow.log_params({k: str(v) for k, v in params.items()
                          if k not in ("verbose", "n_jobs")})
        mlflow.log_param("val_size", str(VAL_SIZE))
        mlflow.log_param("n_features", str(X_train.shape[1]))
        mlflow.log_metric("accuracy_at_1", acc1)
        mlflow.log_metric("accuracy_at_3", acc3)
        mlflow.log_metric("accuracy_at_5", acc5)
        mlflow.log_metric("train_time_sec", round(train_sec, 1))
        mlflow.set_tag("model_type", "lgbm_v2")
    print("Logged to MLflow.", flush=True)

    model.save_model("lgbm_v2_model.txt")
    print("Model saved to lgbm_v2_model.txt")


if __name__ == "__main__":
    main()
