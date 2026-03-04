import numpy as np

from src.models.base import BaseModel
from src.data.features import build_tabular_features


class LGBMModel(BaseModel):
    name = "lightgbm"

    def __init__(self, n_estimators=500, num_leaves=63, learning_rate=0.05, min_child_samples=20):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.model = None
        self.hero2idx = None
        self.target2idx = None
        self.idx2target = None

    def fit(self, train_df, hero2idx, idx2hero, target2idx, idx2target):
        self.hero2idx = hero2idx
        self.target2idx = target2idx
        self.idx2target = idx2target

        import pandas as pd
        import lightgbm as lgb

        X = build_tabular_features(train_df, hero2idx)
        y = train_df["first_pick_hero"].map(target2idx).values
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        self.feature_names = feature_names

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            verbose=-1,
            n_jobs=-1,
        )
        self.model.fit(X_df, y)

    def predict_ranking(self, df):
        import pandas as pd
        X = build_tabular_features(df, self.hero2idx)
        X_df = pd.DataFrame(X, columns=self.feature_names)
        probs = self.model.predict_proba(X_df)

        rankings = []
        for i in range(len(df)):
            sorted_indices = np.argsort(probs[i])[::-1]
            rankings.append([self.idx2target[idx] for idx in sorted_indices])
        return rankings

    def get_params(self):
        return {
            "model": "lightgbm",
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
        }
