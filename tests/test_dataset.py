from app.ml.dataset import generate_frontier_dataset
from app.ml.features import FEATURE_COLUMNS


def test_generate_frontier_dataset_smoke():
    df = generate_frontier_dataset(
        num_worlds=2,
        width=4,
        height=4,
        num_pits=3,
        max_steps=20,
        seed=7,
    )

    assert not df.empty
    for column in FEATURE_COLUMNS + [
        "world_id",
        "step",
        "label_pit",
        "label_wumpus",
        "label_hazard",
    ]:
        assert column in df.columns

    assert set(df["label_pit"].unique()).issubset({0, 1})
    assert set(df["label_hazard"].unique()).issubset({0, 1})
    assert df["world_id"].nunique() <= 2
