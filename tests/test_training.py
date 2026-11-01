import pandas as pd

from app.ml.train import world_level_split, world_level_splits


def make_world_frame() -> pd.DataFrame:
    return pd.DataFrame({"world_id": list(range(20)), "value": list(range(20))})


def test_splits_are_disjoint_and_keep_the_same_test_set():
    df = make_world_frame()

    train_df, test_df = world_level_split(df)
    train_part, val_df, test_part = world_level_splits(df)

    assert set(test_df["world_id"]) == set(test_part["world_id"])
    assert set(train_df["world_id"]) == set(train_part["world_id"]) | set(val_df["world_id"])

    assert not (set(train_part["world_id"]) & set(val_df["world_id"]))
    assert not (set(train_part["world_id"]) & set(test_part["world_id"]))
    assert not (set(val_df["world_id"]) & set(test_part["world_id"]))

    assert len(test_part) == 4  # 20% of 20 worlds
    assert len(val_df) == 4     # 25% of the 16-world train/validation portion
    assert len(train_part) == 12
