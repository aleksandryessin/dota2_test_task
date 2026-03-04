#!/usr/bin/env python3
"""Optuna hyperparameter search for Transformer — optimizes Accuracy@5."""

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
import optuna
import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_dataset, preprocess_dataset, split_train_val, build_target_vocab
from src.data.features import build_sequence_data, CONTEXT_DIM, MAX_BAN_SEQ_LEN

# ── Search space ─────────────────────────────────────────────
#   embed_dim    ∈ {32, 64, 128}
#   nhead        ∈ {2, 4, 8}           (all divide 32/64/128)
#   num_layers   ∈ {1, 2, 3, 4}
#   ff_mult      ∈ {2, 4, 8}           feedforward = embed_dim × ff_mult
#   head_dim     ∈ {128, 256, 512}
#   dropout      ∈ [0.05, 0.5]
#   lr           ∈ [5e-5, 5e-3]        log-uniform
#   weight_decay ∈ [1e-6, 1e-3]        log-uniform
#   batch_size   ∈ {256, 512, 1024}
#   label_smooth ∈ [0.0, 0.15]
# ─────────────────────────────────────────────────────────────

N_TRIALS = 50
MAX_EPOCHS = 100
PATIENCE = 10


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FlexTransformerNet(nn.Module):
    """Transformer with configurable architecture for HPO."""

    def __init__(self, vocab_size, embed_dim, nhead, num_layers, ff_mult,
                 context_dim, num_classes, head_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(MAX_BAN_SEQ_LEN, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim + context_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim // 2, num_classes),
        )

    def forward(self, ban_seq, context):
        B, L = ban_seq.shape
        pos = torch.arange(L, device=ban_seq.device).unsqueeze(0).expand(B, -1)
        x = self.embed(ban_seq) + self.pos_embed(pos)

        pad_mask = ban_seq == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(torch.cat([pooled, context], dim=1))


def accuracy_at_5(net, val_ban, val_ctx, val_heroes, idx2target):
    net.eval()
    with torch.no_grad():
        logits = net(val_ban, val_ctx)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    hits = 0
    for i, hero in enumerate(val_heroes):
        top5 = [idx2target[j] for j in np.argsort(probs[i])[-5:][::-1]]
        if hero in top5:
            hits += 1
    return hits / len(val_heroes)


def train_one(net, train_ban, train_ctx, train_tgt, device, cfg):
    """Train, return (best_state_dict, best_loss, epochs_done)."""
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["label_smoothing"],
    )

    n = len(train_tgt)
    bs = cfg["batch_size"]
    best_loss = float("inf")
    patience_ctr = 0
    best_state = None

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
        if avg < best_loss - 1e-4:
            best_loss = avg
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)
    return best_loss, epoch + 1


def objective(trial, ctx):
    device = ctx["device"]

    embed_dim = trial.suggest_categorical("embed_dim", [64, 128])
    nhead = trial.suggest_categorical("nhead", [4, 16])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    ff_mult = trial.suggest_categorical("ff_mult", [2, 4, 8])
    head_dim = trial.suggest_categorical("head_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.05, 0.5)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)

    cfg = dict(
        embed_dim=embed_dim, nhead=nhead, num_layers=num_layers,
        ff_mult=ff_mult, head_dim=head_dim, dropout=dropout,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        label_smoothing=label_smoothing,
    )

    net = FlexTransformerNet(
        ctx["vocab_size"], embed_dim, nhead, num_layers, ff_mult,
        CONTEXT_DIM, ctx["num_classes"], head_dim, dropout,
    ).to(device)

    t0 = time.time()
    best_loss, epochs = train_one(
        net, ctx["train_ban"], ctx["train_ctx"], ctx["train_tgt"], device, cfg,
    )
    train_sec = time.time() - t0

    acc5 = accuracy_at_5(
        net, ctx["val_ban"], ctx["val_ctx"], ctx["val_heroes"], ctx["idx2target"],
    )

    with mlflow.start_run(run_name=f"tf_trial_{trial.number:03d}"):
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
        mlflow.log_metric("accuracy_at_5", acc5)
        mlflow.log_metric("best_train_loss", best_loss)
        mlflow.log_metric("epochs_trained", epochs)
        mlflow.log_metric("train_time_sec", round(train_sec, 1))
        mlflow.set_tag("model_type", "transformer_optuna")

    print(
        f"  #{trial.number:3d}  acc@5={acc5:.3f}  loss={best_loss:.3f}  "
        f"ep={epochs:2d}  {train_sec:5.1f}s  "
        f"emb={embed_dim} nh={nhead} nl={num_layers} ff={ff_mult} "
        f"hd={head_dim} lr={lr:.4f} do={dropout:.2f}",
        flush=True,
    )
    return acc5


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dota2-first-pick")

    device = get_device()
    print(f"Device: {device}")
    print(f"Trials: {N_TRIALS}, max epochs: {MAX_EPOCHS}, patience: {PATIENCE}")
    print("=" * 80, flush=True)

    print("Loading data...", flush=True)
    df = load_dataset(config["data"]["path"])
    df, hero2idx, idx2hero = preprocess_dataset(df)
    train_df, val_df = split_train_val(df, config["data"]["val_size"])
    target2idx, idx2target = build_target_vocab(train_df)

    train_ban_np, train_ctx_np = build_sequence_data(train_df, hero2idx, MAX_BAN_SEQ_LEN)
    train_tgt_np = train_df["first_pick_hero"].map(target2idx).values
    val_ban_np, val_ctx_np = build_sequence_data(val_df, hero2idx, MAX_BAN_SEQ_LEN)

    ctx = {
        "device": device,
        "vocab_size": len(hero2idx),
        "num_classes": len(target2idx),
        "idx2target": idx2target,
        "train_ban": torch.tensor(train_ban_np, dtype=torch.long, device=device),
        "train_ctx": torch.tensor(train_ctx_np, dtype=torch.float32, device=device),
        "train_tgt": torch.tensor(train_tgt_np, dtype=torch.long, device=device),
        "val_ban": torch.tensor(val_ban_np, dtype=torch.long, device=device),
        "val_ctx": torch.tensor(val_ctx_np, dtype=torch.float32, device=device),
        "val_heroes": val_df["first_pick_hero"].tolist(),
    }

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  "
          f"Heroes: {ctx['vocab_size']} vocab / {ctx['num_classes']} targets", flush=True)
    print("=" * 80, flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda trial: objective(trial, ctx), n_trials=N_TRIALS)

    print(f"\n{'=' * 80}")
    print("SEARCH COMPLETE")
    print(f"{'=' * 80}")
    print(f"Best trial #{study.best_trial.number}  →  accuracy@5 = {study.best_value:.3f}")
    print(f"\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with mlflow.start_run(run_name="transformer_BEST"):
        mlflow.log_params({k: str(v) for k, v in study.best_params.items()})
        mlflow.log_metric("accuracy_at_5", study.best_value)
        mlflow.set_tag("model_type", "transformer_optuna_best")

    print(f"\nAll runs in MLflow:")
    print(f"  mlflow ui --backend-store-uri {config['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
