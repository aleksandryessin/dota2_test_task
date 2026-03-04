#!/usr/bin/env python3
"""Ensemble: CandidateScorerNet + LightGBM weighted average."""

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features_v2 import (
    parse_drafts, build_target_encoding, build_rolling_meta,
    build_ban_sequences, build_context, build_prior_vectors,
    build_ban_mask, build_series_features,
    augment_side_swap, augment_ban_permutation,
    CONTEXT_SCALAR_DIM, MAX_BAN_SEQ_LEN,
)
from src.models.candidate_scorer import CandidateScorerNet

VAL_SIZE = 200
MAX_EPOCHS = 80
PATIENCE = 12
SEED = 42

CFG = dict(
    embed_dim=128, hidden_dim=256, nhead=2, num_layers=2,
    dropout=0.35, lr=8e-4, weight_decay=3e-5, batch_size=512,
    label_smoothing=0.05, captain_alpha=20.0, team_alpha=30.0,
    meta_window=5000, augment_side_swap=True, augment_ban_perm=True,
    focal_gamma=2.0,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (((1 - pt) ** self.gamma) * nll).mean()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy_at_k(probs_np, val_heroes, idx2target, k=5, ban_mask_np=None):
    if ban_mask_np is not None:
        probs_np = probs_np.copy()
        probs_np[ban_mask_np > 0] = -np.inf
    hits = 0
    for i, hero in enumerate(val_heroes):
        top_k = [idx2target[j] for j in np.argsort(probs_np[i])[-k:][::-1]]
        if hero in top_k:
            hits += 1
    return hits / len(val_heroes)


def softmax_np(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("dota2-first-pick")

    device = get_device()
    print(f"Device: {device}", flush=True)

    # ── Data ──────────────────────────────────────────
    print("Loading data...", flush=True)
    df, hero2idx, idx2hero = parse_drafts(pd.read_csv(config["data"]["path"]))

    train_df = df.iloc[:-VAL_SIZE].copy()
    val_df = df.iloc[-VAL_SIZE:].copy()

    target_heroes = sorted(train_df["first_pick_hero"].unique())
    target2idx = {h: i for i, h in enumerate(target_heroes)}
    idx2target = {i: h for h, i in target2idx.items()}
    num_heroes = len(target2idx)

    captain_enc = build_target_encoding(
        train_df, "fp_captain", target2idx, alpha=CFG["captain_alpha"])
    team_enc = build_target_encoding(
        train_df, "fp_team_id", target2idx, alpha=CFG["team_alpha"])
    train_meta_arr = build_rolling_meta(train_df, target2idx, window=CFG["meta_window"])
    val_meta_arr = build_rolling_meta(
        pd.concat([train_df, val_df], ignore_index=True), target2idx,
        window=CFG["meta_window"])[-VAL_SIZE:]

    train_game_pos, train_series_prior = build_series_features(train_df, target2idx)
    val_game_pos, val_series_prior = build_series_features(val_df, target2idx)

    if CFG["augment_side_swap"]:
        train_df = augment_side_swap(train_df)
        train_game_pos = np.concatenate([train_game_pos, train_game_pos])
        train_series_prior = np.concatenate([train_series_prior, train_series_prior])
        train_meta_arr = np.concatenate([train_meta_arr, train_meta_arr])

    train_bans = build_ban_sequences(train_df, hero2idx)
    train_ctx = build_context(train_df, game_positions=train_game_pos)
    train_tgt = train_df["first_pick_hero"].map(target2idx).values
    train_cap, train_team, train_meta = build_prior_vectors(
        train_df, captain_enc, team_enc, train_meta_arr, num_heroes)

    val_bans = build_ban_sequences(val_df, hero2idx)
    val_ctx = build_context(val_df, game_positions=val_game_pos)
    val_heroes = val_df["first_pick_hero"].tolist()
    val_cap, val_team, val_meta = build_prior_vectors(
        val_df, captain_enc, team_enc, val_meta_arr, num_heroes)
    val_ban_mask = build_ban_mask(val_df, target2idx)

    if CFG["augment_ban_perm"]:
        train_bans_perm = augment_ban_permutation(train_bans)
        train_bans = np.concatenate([train_bans, train_bans_perm])
        train_ctx = np.concatenate([train_ctx, train_ctx])
        train_tgt = np.concatenate([train_tgt, train_tgt])
        train_cap = np.concatenate([train_cap, train_cap])
        train_team = np.concatenate([train_team, train_team])
        train_meta = np.concatenate([train_meta, train_meta])
        train_series_prior = np.concatenate([train_series_prior, train_series_prior])

    t_bans = torch.tensor(train_bans, dtype=torch.long, device=device)
    t_ctx = torch.tensor(train_ctx, dtype=torch.float32, device=device)
    t_tgt = torch.tensor(train_tgt, dtype=torch.long, device=device)
    t_cap = torch.tensor(train_cap, dtype=torch.float32, device=device)
    t_team = torch.tensor(train_team, dtype=torch.float32, device=device)
    t_meta = torch.tensor(train_meta, dtype=torch.float32, device=device)
    t_series = torch.tensor(train_series_prior, dtype=torch.float32, device=device)

    v_bans = torch.tensor(val_bans, dtype=torch.long, device=device)
    v_ctx = torch.tensor(val_ctx, dtype=torch.float32, device=device)
    v_cap = torch.tensor(val_cap, dtype=torch.float32, device=device)
    v_team = torch.tensor(val_team, dtype=torch.float32, device=device)
    v_meta = torch.tensor(val_meta, dtype=torch.float32, device=device)
    v_series = torch.tensor(val_series_prior, dtype=torch.float32, device=device)

    print(f"Train: {len(t_tgt):,}  Val: {len(val_heroes)}", flush=True)

    # ── Train Neural Net ──────────────────────────────
    print("\n=== Training CandidateScorerNet ===", flush=True)
    net = CandidateScorerNet(
        vocab_size=len(hero2idx), num_heroes=num_heroes,
        embed_dim=CFG["embed_dim"], hidden_dim=CFG["hidden_dim"],
        context_dim=CONTEXT_SCALAR_DIM, num_layers=CFG["num_layers"],
        nhead=CFG["nhead"], dropout=CFG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=CFG["lr"] * 0.01)
    criterion = FocalLoss(gamma=CFG["focal_gamma"])

    n = len(t_tgt)
    bs = CFG["batch_size"]
    best_acc5 = 0.0
    best_loss = float("inf")
    patience_ctr = 0
    best_state = None

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, device=device)
        total_loss, nb = 0.0, 0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            logits = net(t_bans[idx], t_ctx[idx],
                         t_cap[idx], t_team[idx], t_meta[idx], t_series[idx])
            loss = criterion(logits, t_tgt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        scheduler.step()
        avg_loss = total_loss / nb

        net.eval()
        with torch.no_grad():
            val_logits_np = net(v_bans, v_ctx, v_cap, v_team, v_meta, v_series).cpu().numpy()

        acc5 = accuracy_at_k(val_logits_np, val_heroes, idx2target, k=5, ban_mask_np=val_ban_mask)

        if acc5 > best_acc5:
            best_acc5 = acc5
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep {epoch+1:3d}/{MAX_EPOCHS}  loss={avg_loss:.4f}  "
                  f"acc@5={acc5:.3f}  ({time.time()-t0:.0f}s)", flush=True)

        if patience_ctr >= PATIENCE:
            print(f"Early stop at epoch {epoch+1}", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)

    net.eval()
    with torch.no_grad():
        neural_logits = net(v_bans, v_ctx, v_cap, v_team, v_meta, v_series).cpu().numpy()

    neural_logits_masked = neural_logits.copy()
    neural_logits_masked[val_ban_mask > 0] = -np.inf
    neural_probs = softmax_np(neural_logits_masked)

    neural_acc5 = accuracy_at_k(neural_logits, val_heroes, idx2target, k=5, ban_mask_np=val_ban_mask)
    neural_acc1 = accuracy_at_k(neural_logits, val_heroes, idx2target, k=1, ban_mask_np=val_ban_mask)
    print(f"\nNeural net: acc@1={neural_acc1:.3f}  acc@5={neural_acc5:.3f}", flush=True)

    # ── Load LightGBM probs ──────────────────────────
    print("\n=== Loading LightGBM predictions ===", flush=True)
    lgbm_probs = np.load("lgbm_val_probs.npy")
    lgbm_probs_masked = lgbm_probs.copy()
    lgbm_probs_masked[val_ban_mask > 0] = 0.0
    lgbm_probs_masked /= lgbm_probs_masked.sum(axis=1, keepdims=True) + 1e-8

    lgbm_acc5 = accuracy_at_k(lgbm_probs, val_heroes, idx2target, k=5, ban_mask_np=val_ban_mask)
    lgbm_acc1 = accuracy_at_k(lgbm_probs, val_heroes, idx2target, k=1, ban_mask_np=val_ban_mask)
    print(f"LightGBM:   acc@1={lgbm_acc1:.3f}  acc@5={lgbm_acc5:.3f}", flush=True)

    # ── Ensemble: grid search best weight ────────────
    print("\n=== Ensemble Search ===", flush=True)
    best_ens_acc5 = 0.0
    best_w = 0.5

    for w_lgbm in np.arange(0.0, 1.05, 0.05):
        w_neural = 1.0 - w_lgbm
        ens_probs = w_neural * neural_probs + w_lgbm * lgbm_probs_masked
        acc5 = accuracy_at_k(ens_probs, val_heroes, idx2target, k=5)
        acc1 = accuracy_at_k(ens_probs, val_heroes, idx2target, k=1)
        if acc5 > best_ens_acc5 or (acc5 == best_ens_acc5 and acc1 > best_ens_acc1):
            best_ens_acc5 = acc5
            best_ens_acc1 = acc1
            best_w = w_lgbm
        if w_lgbm in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
            print(f"  w_lgbm={w_lgbm:.2f}  acc@1={acc1:.3f}  acc@5={acc5:.3f}", flush=True)

    w_neural_best = 1.0 - best_w
    ens_probs = w_neural_best * neural_probs + best_w * lgbm_probs_masked
    ens_acc5 = accuracy_at_k(ens_probs, val_heroes, idx2target, k=5)
    ens_acc3 = accuracy_at_k(ens_probs, val_heroes, idx2target, k=3)
    ens_acc1 = accuracy_at_k(ens_probs, val_heroes, idx2target, k=1)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"BEST ENSEMBLE: w_neural={w_neural_best:.2f}  w_lgbm={best_w:.2f}")
    print(f"  acc@1={ens_acc1:.3f}  acc@3={ens_acc3:.3f}  acc@5={ens_acc5:.3f}")
    print(f"\nIndividual models:")
    print(f"  Neural:   acc@5={neural_acc5:.3f}")
    print(f"  LightGBM: acc@5={lgbm_acc5:.3f}")
    print(sep, flush=True)

    with mlflow.start_run(run_name="v2_ensemble"):
        mlflow.log_param("w_neural", f"{w_neural_best:.2f}")
        mlflow.log_param("w_lgbm", f"{best_w:.2f}")
        mlflow.log_metric("accuracy_at_1", ens_acc1)
        mlflow.log_metric("accuracy_at_3", ens_acc3)
        mlflow.log_metric("accuracy_at_5", ens_acc5)
        mlflow.log_metric("neural_acc5", neural_acc5)
        mlflow.log_metric("lgbm_acc5", lgbm_acc5)
        mlflow.set_tag("model_type", "v2_ensemble")
    print("Logged to MLflow.", flush=True)


if __name__ == "__main__":
    main()
