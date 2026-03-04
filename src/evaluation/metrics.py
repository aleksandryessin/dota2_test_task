def accuracy_at_k(true_heroes, rankings, k=5):
    hits = sum(
        1 for true_h, ranking in zip(true_heroes, rankings)
        if true_h in ranking[:k]
    )
    return hits / len(true_heroes)


def evaluate_model(model, val_df, ks=(1, 3, 5)):
    rankings = model.predict_ranking(val_df)
    true_heroes = val_df["first_pick_hero"].tolist()
    return {f"accuracy_at_{k}": accuracy_at_k(true_heroes, rankings, k) for k in ks}
