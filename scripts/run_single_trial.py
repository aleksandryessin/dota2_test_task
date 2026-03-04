#!/usr/bin/env python3
"""Run a single trial with fixed hyperparameters (trial #011 config)."""

import os, sys, time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_dataset, preprocess_dataset, split_train_val, build_target_vocab
from src.data.features import build_sequence_data, CONTEXT_DIM, MAX_BAN_SEQ_LEN
from scripts.run_transformers import FlexTransformerNet, accuracy_at_5, get_device

MAX_EPOCHS = 100
PATIENCE = 10

CFG = dict(
    embed_dim=128, nhead=2, num_layers=3, ff_mult=4,
    head_dim=512, dropout=0.43877373925540614,
    lr=0.000973622059845191, weight_decay=2.502917330901082e-05,
    batch_size=512, label_smoothing=0.07348915065796066,
)


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dota2-first-pick")

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {CFG}")
    print(f"MAX_EPOCHS={MAX_EPOCHS}, PATIENCE={PATIENCE}")
    print("=" * 80, flush=True)

    print("Loading data...", flush=True)
    df = load_dataset(config["data"]["path"])
    df, hero2idx, idx2hero = preprocess_dataset(df)
    train_df, val_df = split_train_val(df, config["data"]["val_size"])
    target2idx, idx2target = build_target_vocab(train_df)

    train_ban_np, train_ctx_np = build_sequence_data(train_df, hero2idx, MAX_BAN_SEQ_LEN)
    train_tgt_np = train_df["first_pick_hero"].map(target2idx).values
    val_ban_np, val_ctx_np = build_sequence_data(val_df, hero2idx, MAX_BAN_SEQ_LEN)

    train_ban = torch.tensor(train_ban_np, dtype=torch.long, device=device)
    train_ctx = torch.tensor(train_ctx_np, dtype=torch.float32, device=device)
    train_tgt = torch.tensor(train_tgt_np, dtype=torch.long, device=device)
    val_ban = torch.tensor(val_ban_np, dtype=torch.long, device=device)
    val_ctx = torch.tensor(val_ctx_np, dtype=torch.float32, device=device)
    val_heroes = val_df["first_pick_hero"].tolist()

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}", flush=True)
    print("=" * 80, flush=True)

    net = FlexTransformerNet(
        len(hero2idx), CFG["embed_dim"], CFG["nhead"], CFG["num_layers"], CFG["ff_mult"],
        CONTEXT_DIM, len(target2idx), CFG["head_dim"], CFG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    n = len(train_tgt)
    bs = CFG["batch_size"]
    best_loss = float("inf")
    patience_ctr = 0
    best_state = None

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        nb = 0
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            loss = criterion(net(train_ban[idx], train_ctx[idx]), train_tgt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        avg = total_loss / nb

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc5 = accuracy_at_5(net, val_ban, val_ctx, val_heroes, idx2target)
            print(
                f"  epoch {epoch+1:3d}/{MAX_EPOCHS}  loss={avg:.4f}  "
                f"acc@5={acc5:.3f}  ({time.time()-t0:.0f}s)", flush=True,
            )
        else:
            print(
                f"  epoch {epoch+1:3d}/{MAX_EPOCHS}  loss={avg:.4f}  "
                f"({time.time()-t0:.0f}s)", flush=True,
            )

        if avg < best_loss - 1e-4:
            best_loss = avg
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stop at epoch {epoch+1}", flush=True)
            break

    train_sec = time.time() - t0

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)

    acc5 = accuracy_at_5(net, val_ban, val_ctx, val_heroes, idx2target)
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"FINAL: acc@5 = {acc5:.3f}  best_loss = {best_loss:.4f}  "
          f"epochs = {epoch+1}  time = {train_sec:.1f}s")
    print(sep, flush=True)

    with mlflow.start_run(run_name="tf_trial037_100ep"):
        mlflow.log_params({k: str(v) for k, v in CFG.items()})
        mlflow.log_params({"max_epochs": str(MAX_EPOCHS), "patience": str(PATIENCE)})
        mlflow.log_metric("accuracy_at_5", acc5)
        mlflow.log_metric("best_train_loss", best_loss)
        mlflow.log_metric("epochs_trained", epoch + 1)
        mlflow.log_metric("train_time_sec", round(train_sec, 1))
        mlflow.set_tag("model_type", "transformer_trial037_100ep")
    print("Logged to MLflow.", flush=True)


if __name__ == "__main__":
    main()
