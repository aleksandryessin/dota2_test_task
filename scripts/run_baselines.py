#!/usr/bin/env python3
"""Train all baseline models and log results to MLflow."""

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_dataset, preprocess_dataset, split_train_val, build_target_vocab
from src.evaluation.metrics import evaluate_model
from src.tracking.mlflow_utils import setup_experiment, log_run
from src.models.popularity import PopularityModel
from src.models.lgbm_model import LGBMModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    setup_experiment(
        config["mlflow"]["experiment_name"],
        config["mlflow"]["tracking_uri"],
    )

    print("=" * 60, flush=True)
    print("Loading data...", flush=True)
    df = load_dataset(config["data"]["path"])
    df, hero2idx, idx2hero = preprocess_dataset(df)
    train_df, val_df = split_train_val(df, config["data"]["val_size"])
    target2idx, idx2target = build_target_vocab(train_df)

    print(f"Heroes: {len(hero2idx)} total, {len(target2idx)} first-pick targets", flush=True)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}", flush=True)
    print("=" * 60, flush=True)

    # MPS (Apple GPU) segfaults if torch runs after LightGBM's OpenMP init,
    # so neural models must train before LightGBM.
    models = [
        PopularityModel(),
        LSTMModel(**config["lstm"]),
        TransformerModel(**config["transformer"]),
        LGBMModel(**config["lgbm"]),
    ]

    results = []
    for model in models:
        print(f"\n{'—' * 40}", flush=True)
        print(f"Training: {model.name}", flush=True)
        print(f"{'—' * 40}", flush=True)

        t0 = time.time()
        model.fit(train_df, hero2idx, idx2hero, target2idx, idx2target)
        train_time = time.time() - t0
        print(f"  Training complete in {train_time:.1f}s, evaluating...", flush=True)

        metrics = evaluate_model(model, val_df)
        metrics["train_time_sec"] = round(train_time, 2)

        params = model.get_params()
        params["num_heroes"] = len(hero2idx)
        params["num_target_classes"] = len(target2idx)

        log_run(model.name, params, metrics)
        results.append((model.name, metrics))

        for k, v in metrics.items():
            print(f"  {k}: {v}", flush=True)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'Acc@1':>8} {'Acc@3':>8} {'Acc@5':>8} {'Time(s)':>10}")
    print("-" * 55)
    for name, m in results:
        print(
            f"{name:<15} {m['accuracy_at_1']:>8.3f} {m['accuracy_at_3']:>8.3f} "
            f"{m['accuracy_at_5']:>8.3f} {m['train_time_sec']:>10.1f}"
        )
    print("=" * 60)
    tracking_uri = config["mlflow"]["tracking_uri"]
    print("\nAll runs logged to MLflow. Start UI with:")
    print(f"  mlflow ui --backend-store-uri {tracking_uri}")


if __name__ == "__main__":
    main()
