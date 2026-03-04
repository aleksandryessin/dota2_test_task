#!/usr/bin/env python3
"""Optuna HPO for LightGBM V2 — optimizes Accuracy@5.

Features:
  - LightGBM early stopping per trial (stops if val logloss stops improving)
  - Optuna pruning via LightGBMPruningCallback (kills bad trials mid-training)
  - MLflow logging of every trial (params, metrics, feature importance)
  - Best model saved with full diagnostics
  - Visualization: optimization history, param importance, feature importance, loss curves
"""

import os
import sys
import time
import json
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features_v2 import (
    parse_drafts, build_target_encoding, build_rolling_meta,
    build_context, build_ban_mask, build_series_features,
)

# ── Config ──────────────────────────────────────────────
VAL_SIZE = 200
SEED = 42
N_TRIALS = 40
MAX_BOOST_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 50

RESULTS_DIR = Path("optuna_lgbm_results")


# ── Feature builder ─────────────────────────────────────
def build_lgbm_features(df, hero2idx, captain_enc, team_enc,
                        meta_arr, game_pos):
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

        cap_top[i] = np.sort(cap_dist)[::-1][:10]
        team_top[i] = np.sort(team_dist)[::-1][:10]
        cap_entropy[i] = -np.sum(cap_dist * np.log(cap_dist + 1e-8))
        team_entropy[i] = -np.sum(team_dist * np.log(team_dist + 1e-8))

        if meta_arr.ndim == 2:
            meta_top[i] = np.sort(meta_arr[i])[::-1][:10]
        else:
            meta_top[i] = np.sort(meta_arr)[::-1][:10]

    return np.hstack([
        ctx, ban_multihot,
        cap_top, team_top, meta_top,
        cap_entropy.reshape(-1, 1),
        team_entropy.reshape(-1, 1),
    ])


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
    RESULTS_DIR.mkdir(exist_ok=True)

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dota2-first-pick")

    # ── Load data once ──────────────────────────────────
    print("Loading & parsing data...", flush=True)
    df, hero2idx, idx2hero = parse_drafts(pd.read_csv(config["data"]["path"]))

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
    print(f"Trials: {N_TRIALS}  Max rounds: {MAX_BOOST_ROUNDS}  "
          f"Early stop: {EARLY_STOPPING_ROUNDS}", flush=True)

    # ── Storage for trial histories ─────────────────────
    all_trials = []
    best_model = [None]
    best_importance = [None]

    def objective(trial):
        # Feature engineering hyperparameters
        captain_alpha = trial.suggest_float("captain_alpha", 5.0, 80.0, log=True)
        team_alpha = trial.suggest_float("team_alpha", 10.0, 100.0, log=True)
        meta_window = trial.suggest_int("meta_window", 2000, 15000, step=1000)

        # Build features with trial-specific encoding params
        captain_enc = build_target_encoding(
            train_df, "fp_captain", target2idx, alpha=captain_alpha)
        team_enc = build_target_encoding(
            train_df, "fp_team_id", target2idx, alpha=team_alpha)
        train_meta_arr = build_rolling_meta(train_df, target2idx, window=meta_window)
        val_meta_arr = build_rolling_meta(
            pd.concat([train_df, val_df], ignore_index=True),
            target2idx, window=meta_window)[-VAL_SIZE:]

        X_train = build_lgbm_features(
            train_df, hero2idx, captain_enc, team_enc,
            train_meta_arr, train_game_pos)
        X_val = build_lgbm_features(
            val_df, hero2idx, captain_enc, team_enc,
            val_meta_arr, val_game_pos)

        # LightGBM hyperparameters
        params = {
            "objective": "multiclass",
            "num_class": num_heroes,
            "metric": "multi_logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "verbose": -1,
            "seed": SEED,
            "n_jobs": -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train with early stopping + Optuna pruning callback
        pruning_cb = LightGBMPruningCallback(trial, "val")
        evals_result = {}

        t0 = time.time()
        try:
            model = lgb.train(
                params, dtrain,
                num_boost_round=MAX_BOOST_ROUNDS,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=0),
                    lgb.record_evaluation(evals_result),
                    pruning_cb,
                ],
            )
        except optuna.exceptions.TrialPruned:
            raise

        train_sec = time.time() - t0
        n_rounds = model.best_iteration if model.best_iteration > 0 else model.num_trees() // num_heroes

        val_probs = model.predict(X_val, num_iteration=n_rounds)

        acc5 = accuracy_at_k(val_probs, val_heroes, idx2target, k=5, ban_mask_np=val_ban_mask)
        acc3 = accuracy_at_k(val_probs, val_heroes, idx2target, k=3, ban_mask_np=val_ban_mask)
        acc1 = accuracy_at_k(val_probs, val_heroes, idx2target, k=1, ban_mask_np=val_ban_mask)

        trial_info = {
            "number": trial.number,
            "acc5": acc5, "acc3": acc3, "acc1": acc1,
            "n_rounds": n_rounds, "train_sec": round(train_sec, 1),
            "train_loss": evals_result.get("train", {}).get("multi_logloss", []),
            "val_loss": evals_result.get("val", {}).get("multi_logloss", []),
        }
        all_trials.append(trial_info)

        # Track best model for later analysis
        if best_model[0] is None or acc5 > best_model[0]["acc5"]:
            best_model[0] = {"acc5": acc5, "model": model, "trial": trial.number,
                             "n_rounds": n_rounds, "params": dict(trial.params)}
            best_importance[0] = model.feature_importance(importance_type="gain")

        # Log to MLflow
        with mlflow.start_run(run_name=f"lgbm_optuna_{trial.number:03d}"):
            mlflow.log_params({k: str(v) for k, v in params.items()
                              if k not in ("verbose", "n_jobs")})
            mlflow.log_params({
                "captain_alpha": str(captain_alpha),
                "team_alpha": str(team_alpha),
                "meta_window": str(meta_window),
            })
            mlflow.log_metric("accuracy_at_1", acc1)
            mlflow.log_metric("accuracy_at_3", acc3)
            mlflow.log_metric("accuracy_at_5", acc5)
            mlflow.log_metric("n_boost_rounds", n_rounds)
            mlflow.log_metric("train_time_sec", round(train_sec, 1))

            if evals_result.get("val", {}).get("multi_logloss"):
                for step, loss_val in enumerate(evals_result["val"]["multi_logloss"]):
                    mlflow.log_metric("val_logloss", loss_val, step=step)
            if evals_result.get("train", {}).get("multi_logloss"):
                for step, loss_val in enumerate(evals_result["train"]["multi_logloss"]):
                    mlflow.log_metric("train_logloss", loss_val, step=step)

            mlflow.set_tag("model_type", "lgbm_optuna")

        star = " ★ NEW BEST" if acc5 >= (best_model[0] or {}).get("acc5", 0) else ""
        print(f"  Trial {trial.number:3d}  acc@1={acc1:.3f}  @3={acc3:.3f}  "
              f"@5={acc5:.3f}  rounds={n_rounds:3d}  ({train_sec:.0f}s){star}",
              flush=True)

        return acc5

    # ── Run Optuna ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Starting Optuna HPO: {N_TRIALS} trials")
    print(f"{'=' * 70}\n", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=30,
        ),
        study_name="lgbm_v2_hpo",
    )

    t_start = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    total_time = time.time() - t_start

    # ── Results ─────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"OPTUNA COMPLETE — {N_TRIALS} trials in {total_time:.0f}s "
          f"({total_time/60:.1f} min)")
    print(sep)

    best = study.best_trial
    print(f"\nBest trial #{best.number}:  Acc@5 = {best.value:.3f}")
    print(f"Parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # ── Save best model ─────────────────────────────────
    if best_model[0] is not None:
        best_model[0]["model"].save_model(str(RESULTS_DIR / "best_lgbm_model.txt"))
        print(f"\nBest model saved to {RESULTS_DIR / 'best_lgbm_model.txt'}")

    # ── Save trial results table ────────────────────────
    trial_rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "acc5": t.value}
            row.update(t.params)
            trial_rows.append(row)
    results_df = pd.DataFrame(trial_rows).sort_values("acc5", ascending=False)
    results_df.to_csv(RESULTS_DIR / "trial_results.csv", index=False)
    print(f"\nTop-10 trials:")
    print(results_df.head(10).to_string(index=False))

    # ── Save study object ───────────────────────────────
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=2)

    # ── Visualizations ──────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Optimization history
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trial_nums = [t.number for t in completed]
        trial_vals = [t.value for t in completed]
        best_so_far = np.maximum.accumulate(trial_vals)

        axes[0, 0].scatter(trial_nums, trial_vals, alpha=0.6, s=30, label="Trial")
        axes[0, 0].plot(trial_nums, best_so_far, "r-", linewidth=2, label="Best so far")
        axes[0, 0].axhline(y=0.475, color="gray", linestyle="--", alpha=0.5, label="Baseline (0.475)")
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("Acc@5")
        axes[0, 0].set_title("Optimization History")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Loss curves of best trial
        best_trial_info = None
        for t in all_trials:
            if t["number"] == best.number:
                best_trial_info = t
                break

        if best_trial_info and best_trial_info["train_loss"]:
            rounds = range(1, len(best_trial_info["train_loss"]) + 1)
            axes[0, 1].plot(rounds, best_trial_info["train_loss"], "b-",
                           linewidth=1, alpha=0.7, label="Train")
            if best_trial_info["val_loss"]:
                axes[0, 1].plot(rounds, best_trial_info["val_loss"], "r-",
                               linewidth=1.5, label="Validation")
            axes[0, 1].set_xlabel("Boosting Round")
            axes[0, 1].set_ylabel("Multi-Logloss")
            axes[0, 1].set_title(f"Loss Curve — Best Trial #{best.number}")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature importance (best model)
        if best_importance[0] is not None:
            ctx_names = ["fp_team", "year", "month", "dow", "hour",
                         "cluster", "s_type", "league", "game_pos"]
            ban_names = [f"ban_{i}" for i in range(len(hero2idx))]
            cap_names = [f"cap_t{i}" for i in range(1, 11)]
            team_names = [f"team_t{i}" for i in range(1, 11)]
            meta_names = [f"meta_t{i}" for i in range(1, 11)]
            entropy_names = ["cap_H", "team_H"]
            all_names = ctx_names + ban_names + cap_names + team_names + meta_names + entropy_names

            imp = best_importance[0]
            if len(all_names) == len(imp):
                sorted_idx = np.argsort(imp)[-20:]
                axes[1, 0].barh(range(20), imp[sorted_idx], color="steelblue")
                axes[1, 0].set_yticks(range(20))
                axes[1, 0].set_yticklabels([all_names[i] for i in sorted_idx], fontsize=8)
                axes[1, 0].set_xlabel("Gain")
                axes[1, 0].set_title("Top-20 Features — Best Model")

                # Group importance
                group_imp = {"Context": 0, "Bans": 0, "Captain": 0,
                             "Team": 0, "Meta": 0, "Entropy": 0}
                for j, name in enumerate(all_names):
                    if name.startswith("ban_"):
                        group_imp["Bans"] += imp[j]
                    elif name.startswith("cap_"):
                        group_imp["Captain"] += imp[j]
                    elif name.startswith("team_"):
                        group_imp["Team"] += imp[j]
                    elif name.startswith("meta_"):
                        group_imp["Meta"] += imp[j]
                    elif name in ("cap_H", "team_H"):
                        group_imp["Entropy"] += imp[j]
                    else:
                        group_imp["Context"] += imp[j]

                labels = list(group_imp.keys())
                sizes = [group_imp[l] for l in labels]
                colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
                axes[1, 1].pie(sizes, labels=labels, autopct="%1.1f%%",
                              colors=colors[:len(labels)])
                axes[1, 1].set_title("Importance by Group — Best Model")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "optuna_results.png", dpi=150, bbox_inches="tight")
        print(f"\nPlots saved to {RESULTS_DIR / 'optuna_results.png'}")
        plt.close()

        # 4. Param importance (Optuna)
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            importances = optuna.importance.get_param_importances(study)
            names = list(importances.keys())
            vals = list(importances.values())
            ax2.barh(range(len(names)), vals, color="steelblue")
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names)
            ax2.invert_yaxis()
            ax2.set_xlabel("Importance")
            ax2.set_title("Hyperparameter Importance (Optuna fANOVA)")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "param_importance.png", dpi=150, bbox_inches="tight")
            print(f"Param importance saved to {RESULTS_DIR / 'param_importance.png'}")
            plt.close()
        except Exception as e:
            print(f"Could not compute param importance: {e}")

    except ImportError:
        print("matplotlib not installed, skipping plots")

    # ── Log best trial summary to MLflow ────────────────
    with mlflow.start_run(run_name=f"lgbm_optuna_BEST_{best.number:03d}"):
        mlflow.log_params({k: str(v) for k, v in best.params.items()})
        mlflow.log_metric("accuracy_at_5", best.value)
        best_trial_data = next((t for t in all_trials if t["number"] == best.number), None)
        if best_trial_data:
            mlflow.log_metric("accuracy_at_3", best_trial_data["acc3"])
            mlflow.log_metric("accuracy_at_1", best_trial_data["acc1"])
            mlflow.log_metric("n_boost_rounds", best_trial_data["n_rounds"])
        mlflow.log_metric("total_hpo_time_sec", round(total_time, 1))
        mlflow.log_metric("n_trials", N_TRIALS)
        mlflow.set_tag("model_type", "lgbm_optuna_best")

        for artifact in ["optuna_results.png", "param_importance.png",
                         "trial_results.csv", "best_params.json"]:
            path = RESULTS_DIR / artifact
            if path.exists():
                mlflow.log_artifact(str(path))

    print(f"\n{sep}")
    print(f"DONE. Best: Trial #{best.number}, Acc@5 = {best.value:.3f}")
    print(f"Results dir: {RESULTS_DIR}/")
    print(f"MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(sep, flush=True)


if __name__ == "__main__":
    main()
