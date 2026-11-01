from dataclasses import dataclass

import joblib
import pandas as pd

from app.ml.features import FEATURE_COLUMNS


@dataclass
class PitRiskModel:
    model: object
    feature_columns: list[str]

    def predict_proba_from_dict(self, features: dict[str, float | int]) -> float:
        row = pd.DataFrame([[features[c] for c in self.feature_columns]], columns=self.feature_columns)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(row)[0, 1])
        score = float(self.model.decision_function(row)[0])
        return 1.0 / (1.0 + pow(2.718281828459045, -score))

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns}, path)

    @classmethod
    def load(cls, path: str) -> "PitRiskModel":
        payload = joblib.load(path)
        return cls(model=payload["model"], feature_columns=payload.get("feature_columns", FEATURE_COLUMNS))
