from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train_df, hero2idx, idx2hero, target2idx, idx2target):
        pass

    @abstractmethod
    def predict_ranking(self, df):
        """Return list of hero_id rankings (most probable first) for each row."""
        pass

    @abstractmethod
    def get_params(self):
        pass
