import numpy as np
import pandas as pd

MAX_BAN_SEQ_LEN = 6
CONTEXT_DIM = 6


def build_ban_multihot(df, hero2idx):
    """Multi-hot encoding of pre-first-pick bans."""
    n = len(df)
    num_heroes = len(hero2idx)
    features = np.zeros((n, num_heroes), dtype=np.float32)
    for i, bans in enumerate(df["pre_fp_bans"]):
        for h in bans:
            if h in hero2idx:
                features[i, hero2idx[h]] = 1.0
    return features


def build_time_features(df):
    dt = pd.to_datetime(df["start_time"], unit="s")
    return np.column_stack([
        dt.dt.year.values,
        dt.dt.month.values,
        dt.dt.dayofweek.values,
        dt.dt.hour.values,
    ]).astype(np.float32)


def build_tabular_features(df, hero2idx):
    """Full feature matrix for tree-based models: bans + time + context."""
    ban_feats = build_ban_multihot(df, hero2idx)
    time_feats = build_time_features(df)
    fp_team = df["first_pick_team"].values.reshape(-1, 1).astype(np.float32)
    cluster = df["cluster"].values.reshape(-1, 1).astype(np.float32)
    return np.hstack([ban_feats, time_feats, fp_team, cluster])


def build_sequence_data(df, hero2idx, max_len=MAX_BAN_SEQ_LEN):
    """Build padded ban sequences and normalized context for neural models.

    Returns (ban_seqs: int64 array, context: float32 array).
    Hero indices are shifted +1 so that 0 = padding.
    """
    ban_seqs = []
    for bans in df["pre_fp_bans"]:
        indices = [hero2idx[h] + 1 for h in bans[:max_len] if h in hero2idx]
        indices += [0] * (max_len - len(indices))
        ban_seqs.append(indices)

    ban_seqs = np.array(ban_seqs, dtype=np.int64)
    time_feats = build_time_features(df)

    context = np.column_stack([
        df["first_pick_team"].values.astype(np.float32),
        (time_feats[:, 0] - 2023) / 3,
        time_feats[:, 1] / 12,
        time_feats[:, 2] / 6,
        time_feats[:, 3] / 24,
        df["cluster"].values.astype(np.float32) / 1000,
    ]).astype(np.float32)

    return ban_seqs, context
