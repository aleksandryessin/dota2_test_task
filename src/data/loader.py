import json

import pandas as pd
from tqdm import tqdm

VAL_SIZE = 200


def load_dataset(path="data/test_task_dataset.csv"):
    return pd.read_csv(path)


def preprocess_dataset(df):
    """Parse all drafts, extract pre-first-pick info, build hero vocabulary.

    Returns (df_with_new_cols, hero2idx, idx2hero).
    """
    all_heroes = set(df["first_pick_hero"].unique())
    bans_list = []
    fp_teams = []

    for pb_str in tqdm(df["picks_bans"], desc="Parsing drafts"):
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

    hero_ids_sorted = sorted(all_heroes)
    hero2idx = {h: i for i, h in enumerate(hero_ids_sorted)}
    idx2hero = {i: h for h, i in hero2idx.items()}

    return df, hero2idx, idx2hero


def build_target_vocab(train_df):
    """Build target class mapping from training first-pick heroes."""
    target_heroes = sorted(train_df["first_pick_hero"].unique())
    target2idx = {h: i for i, h in enumerate(target_heroes)}
    idx2target = {i: h for h, i in target2idx.items()}
    return target2idx, idx2target


def split_train_val(df, val_size=VAL_SIZE):
    return df.iloc[:-val_size].copy(), df.iloc[-val_size:].copy()
