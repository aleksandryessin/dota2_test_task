from src.models.base import BaseModel


class PopularityModel(BaseModel):
    name = "popularity"

    def __init__(self):
        self.ranking = []

    def fit(self, train_df, hero2idx, idx2hero, target2idx, idx2target):
        self.ranking = train_df["first_pick_hero"].value_counts().index.tolist()

    def predict_ranking(self, df):
        return [self.ranking] * len(df)

    def get_params(self):
        return {"model": "popularity", "n_heroes_ranked": len(self.ranking)}
