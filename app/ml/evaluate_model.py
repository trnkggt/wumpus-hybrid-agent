from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from app.ml.features import FEATURE_COLUMNS
from app.ml.models import PitRiskModel
from app.ml.train import world_level_split


def evaluate_saved_model(
    dataset_csv: str,
    model_path: str,
    report_path: str | None = None,
) -> None:
    df = pd.read_csv(dataset_csv)
    _train_df, test_df = world_level_split(df)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label_pit"]

    model = PitRiskModel.load(model_path)
    y_pred = model.model.predict(X_test)

    text = "\n".join([
        f"Dataset: {dataset_csv}",
        f"Model:   {model_path}",
        "",
        "Confusion matrix:",
        str(confusion_matrix(y_test, y_pred)),
        "",
        "Classification report:",
        classification_report(y_test, y_pred, zero_division=0),
    ])
    print(text)

    if report_path is not None:
        Path(report_path).write_text(text + "\n")


if __name__ == "__main__":
    default_dataset = Path("artifacts/frontier_dataset.csv")
    default_model = Path("artifacts/pit_risk_logreg.joblib")
    evaluate_saved_model(
        str(default_dataset),
        str(default_model),
        report_path="artifacts/model_report.txt",
    )
