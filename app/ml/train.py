from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.dataset import generate_frontier_dataset
from app.ml.features import FEATURE_COLUMNS
from app.ml.models import PitRiskModel

DATASET_SEED = 42
TEST_SIZE = 0.2
VAL_FRACTION = 0.25


def _subset(df: pd.DataFrame, world_ids: list[int]) -> pd.DataFrame:
    return df[df["world_id"].isin(world_ids)].copy()


def world_level_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = DATASET_SEED,
):
    """World-level train/test split.

    The returned test rows are identical to ``world_level_splits`` so
    ``evaluate_model.py`` can reconstruct the same held-out worlds.
    """
    world_ids = df["world_id"].drop_duplicates().tolist()
    train_ids, test_ids = train_test_split(
        world_ids,
        test_size=test_size,
        random_state=random_state,
    )
    return _subset(df, train_ids), _subset(df, test_ids)


def world_level_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_fraction: float = VAL_FRACTION,
    random_state: int = DATASET_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """World-level train/validation/test split.

    The first split is identical to ``world_level_split``, so the test set is
    unchanged for back-compatibility with the saved-model evaluator.
    """
    world_ids = df["world_id"].drop_duplicates().tolist()
    train_val_ids, test_ids = train_test_split(
        world_ids,
        test_size=test_size,
        random_state=random_state,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_fraction,
        random_state=random_state,
    )
    return _subset(df, train_ids), _subset(df, val_ids), _subset(df, test_ids)


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_pit_models(df: pd.DataFrame, save_dir: str | Path = "artifacts") -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = world_level_splits(df)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label_pit"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["label_pit"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label_pit"]

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results: dict = {}
    metric_rows: list[dict] = []
    best_name: str | None = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_metrics = evaluate_classifier(model, X_val, y_val)
        test_metrics = evaluate_classifier(model, X_test, y_test)
        results[name] = {"validation": val_metrics, "test": test_metrics}

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_name = name

        wrapper = PitRiskModel(model=model, feature_columns=FEATURE_COLUMNS)
        wrapper.save(str(save_dir / f"pit_risk_{name}.joblib"))

        metric_rows.append({
            "model": name,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        })

    if best_name is not None:
        results["best_model"] = {"name": best_name, "validation_f1": best_f1}

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.insert(0, "dataset_worlds", df["world_id"].nunique())
    metrics_df.insert(1, "dataset_rows", len(df))
    metrics_df.insert(2, "train_worlds", train_df["world_id"].nunique())
    metrics_df.insert(3, "val_worlds", val_df["world_id"].nunique())
    metrics_df.insert(4, "test_worlds", test_df["world_id"].nunique())
    metrics_df.to_csv(save_dir / "model_metrics.csv", index=False)

    return results


def main() -> None:
    df = generate_frontier_dataset(
        num_worlds=500,
        width=4,
        height=4,
        num_pits=3,
        max_steps=100,
        seed=DATASET_SEED,
    )
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    dataset_path = output_dir / "frontier_dataset.csv"
    df.to_csv(dataset_path, index=False)

    results = train_pit_models(df, save_dir=output_dir)
    print("Saved dataset to", dataset_path)
    print("Training results:")
    for name, metrics in results.items():
        print(name, metrics)


if __name__ == "__main__":
    main()
