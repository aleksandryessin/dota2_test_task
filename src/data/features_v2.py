"""V2 features: captain/team target encoding, rolling meta, augmentation."""

import json
import numpy as np
import pandas as pd

MAX_BAN_SEQ_LEN = 6


def parse_drafts(df):
    """Extract pre-first-pick bans, first-pick team, and first-pick captain/team ID."""
    all_heroes = set(df["first_pick_hero"].unique())
    bans_list, fp_teams = [], []

    for pb_str in df["picks_bans"]:
        actions = json.loads(pb_str)
        for a in actions:
            all_heroes.add(a["hero_id"])
        actions_sorted = sorted(actions, key=lambda x: x["order"])
        bans = []
        fp_team = None
        for a in actions_sorted:
            if a["is_pick"]:
                fp_team = a["team"]
                break
            bans.append(a["hero_id"])
        bans_list.append(bans)
        fp_teams.append(fp_team)

    df = df.copy()
    df["pre_fp_bans"] = bans_list
    df["first_pick_team"] = fp_teams
    df["fp_captain"] = np.where(df["first_pick_team"] == 0,
                                df["radiant_captain"], df["dire_captain"])
    df["fp_team_id"] = np.where(df["first_pick_team"] == 0,
                                df["radiant_team_id"], df["dire_team_id"])

    hero_ids_sorted = sorted(all_heroes)
    hero2idx = {h: i for i, h in enumerate(hero_ids_sorted)}
    idx2hero = {i: h for h, i in hero2idx.items()}
    return df, hero2idx, idx2hero


def build_target_encoding(train_df, column, target2idx, alpha=20.0):
    """Smoothed target encoding: P(hero | entity) for each captain/team.

    Returns dict mapping entity_id → np.array of shape [num_target_classes].
    Smoothing: (count * empirical + alpha * global) / (count + alpha)
    """
    num_classes = len(target2idx)

    global_counts = train_df["first_pick_hero"].value_counts()
    global_dist = np.zeros(num_classes, dtype=np.float32)
    for h, cnt in global_counts.items():
        if h in target2idx:
            global_dist[target2idx[h]] = cnt
    global_dist /= global_dist.sum() + 1e-8

    encoding = {}
    for entity_id, group in train_df.groupby(column):
        counts = group["first_pick_hero"].value_counts()
        n = len(group)
        empirical = np.zeros(num_classes, dtype=np.float32)
        for h, cnt in counts.items():
            if h in target2idx:
                empirical[target2idx[h]] = cnt
        empirical /= n + 1e-8
        smoothed = (n * empirical + alpha * global_dist) / (n + alpha)
        encoding[entity_id] = smoothed

    encoding["__default__"] = global_dist
    return encoding


def build_rolling_meta(train_df, target2idx, window=5000):
    """Per-sample rolling hero pick frequency over the preceding `window` matches.

    Returns np.array of shape [N, num_target_classes].
    Each row is the meta distribution computed from the `window` matches before that sample.
    """
    num_classes = len(target2idx)
    heroes = train_df["first_pick_hero"].values
    N = len(heroes)

    hero_indices = np.full(N, -1, dtype=np.int32)
    for i, h in enumerate(heroes):
        if h in target2idx:
            hero_indices[i] = target2idx[h]

    counts = np.zeros(num_classes, dtype=np.float64)
    meta_arr = np.zeros((N, num_classes), dtype=np.float32)

    for i in range(N):
        total = min(i, window)
        if total > 0:
            row = counts.copy()
            row /= total
            meta_arr[i] = row.astype(np.float32)

        idx = hero_indices[i]
        if idx >= 0:
            counts[idx] += 1

        if i >= window:
            old_idx = hero_indices[i - window]
            if old_idx >= 0:
                counts[old_idx] -= 1

    return meta_arr


def build_series_features(df, target2idx):
    """Game position within series + prior vector of heroes first-picked earlier in the series.

    Returns:
        game_positions: np.array [N] float32 — game position / max_games (normalized)
        series_prior: np.array [N, num_classes] float32 — binary flags for previous fps in series
    """
    num_classes = len(target2idx)
    game_pos = np.zeros(len(df), dtype=np.float32)
    series_prior = np.zeros((len(df), num_classes), dtype=np.float32)

    series_fps = {}  # series_id → list of (row_index, first_pick_hero)
    for i, (_, row) in enumerate(df.iterrows()):
        sid = row.get("series_id", np.nan)
        if pd.isna(sid):
            game_pos[i] = 0.0
            continue

        if sid not in series_fps:
            series_fps[sid] = []

        prev_picks = series_fps[sid]
        game_pos[i] = len(prev_picks) / 3.0

        for prev_hero in prev_picks:
            if prev_hero in target2idx:
                series_prior[i, target2idx[prev_hero]] = 1.0

        series_fps[sid].append(row["first_pick_hero"])

    return game_pos, series_prior


def build_ban_sequences(df, hero2idx, max_len=MAX_BAN_SEQ_LEN):
    """Padded ban token IDs (hero_idx + 1, 0 = padding)."""
    ban_seqs = []
    for bans in df["pre_fp_bans"]:
        indices = [hero2idx[h] + 1 for h in bans[:max_len] if h in hero2idx]
        indices += [0] * (max_len - len(indices))
        ban_seqs.append(indices)
    return np.array(ban_seqs, dtype=np.int64)


def build_context(df, game_positions=None):
    """Scalar context features: team side, time, cluster, series_type, leagueid, game_in_series."""
    dt = pd.to_datetime(df["start_time"], unit="s")
    series_type = df["series_type"].fillna(0).values.astype(np.float32)
    leagueid = df["leagueid"].fillna(0).values.astype(np.float32)
    league_max = max(leagueid.max(), 1.0)
    cols = [
        df["first_pick_team"].values.astype(np.float32),
        (dt.dt.year.values - 2023) / 3,
        dt.dt.month.values / 12,
        dt.dt.dayofweek.values / 6,
        dt.dt.hour.values / 24,
        df["cluster"].values.astype(np.float32) / 1000,
        series_type / 3,
        leagueid / league_max,
    ]
    if game_positions is not None:
        cols.append(game_positions)
    return np.column_stack(cols).astype(np.float32)


CONTEXT_SCALAR_DIM = 9


def build_prior_vectors(df, captain_enc, team_enc, meta_arr, num_heroes):
    """Per-sample prior distributions: captain + team + meta.

    meta_arr: [N, num_heroes] per-sample rolling meta OR [num_heroes] single vector.
    Returns (cap_arr, team_arr, meta_broadcast) each [N, num_heroes].
    """
    default = captain_enc["__default__"]
    cap_priors, team_priors = [], []

    for _, row in df.iterrows():
        cap_id = row["fp_captain"]
        team_id = row.get("fp_team_id", np.nan)

        cap_priors.append(captain_enc.get(cap_id, default))
        team_priors.append(team_enc.get(team_id, default) if pd.notna(team_id) else default)

    cap_out = np.stack(cap_priors)
    team_out = np.stack(team_priors)

    if meta_arr.ndim == 1:
        meta_out = np.tile(meta_arr, (len(df), 1))
    else:
        meta_out = meta_arr

    return cap_out, team_out, meta_out


def build_ban_mask(df, target2idx):
    """Binary mask [N, num_classes]: 1 where hero is banned (cannot be first-picked)."""
    num_classes = len(target2idx)
    masks = np.zeros((len(df), num_classes), dtype=np.float32)
    for i, bans in enumerate(df["pre_fp_bans"]):
        for h in bans:
            if h in target2idx:
                masks[i, target2idx[h]] = 1.0
    return masks


def augment_side_swap(df):
    """Double data by swapping radiant/dire sides.

    For each match: flip first_pick_team, swap captain/team IDs.
    Bans and first_pick_hero stay the same.
    """
    swapped = df.copy()
    swapped["first_pick_team"] = 1 - swapped["first_pick_team"]
    swapped["fp_captain"] = np.where(
        swapped["first_pick_team"] == 0,
        swapped["radiant_captain"], swapped["dire_captain"],
    )
    swapped["fp_team_id"] = np.where(
        swapped["first_pick_team"] == 0,
        swapped["radiant_team_id"], swapped["dire_team_id"],
    )
    return pd.concat([df, swapped], ignore_index=True)


def augment_ban_permutation(ban_seqs):
    """Randomly permute non-padding ban tokens in each sequence."""
    result = ban_seqs.copy()
    for i in range(len(result)):
        nonzero = result[i] != 0
        n_nonzero = nonzero.sum()
        if n_nonzero > 1:
            perm = np.random.permutation(n_nonzero)
            result[i, :n_nonzero] = result[i, :n_nonzero][perm]
    return result
