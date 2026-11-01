from dataclasses import asdict
import random

import numpy as np
import pandas as pd

from app.agent.KB import KnowledgeBase
from app.agent.hybrid_wumpus_agent import HybridWumpusAgent
from app.agent.wumpus_agent import WumpusAgent
from app.domain.cave import CaveGeometry
from app.domain.world import World
from app.evaluation.metrics import summarize_episode
from app.main import build_random_grid, run_game
from app.ml.models import PitRiskModel

# Training uses DATASET_SEED (42) in app.ml.train.  Evaluation must use a
# different seed so the hybrid is not scored on world layouts it trained on.
EVAL_SEED = 2025


def compare_agents(
    pit_model_path: str,
    num_worlds: int,
    width: int = 4,
    height: int = 4,
    num_pits: int = 3,
    seed: int = EVAL_SEED,
    max_steps: int = 100,
    risk_threshold: float = 0.6,
    utility_floor: float = -8.0,
) -> pd.DataFrame:
    rng = random.Random(seed)
    pit_model = PitRiskModel.load(pit_model_path)
    rows: list[dict] = []

    for world_id in range(num_worlds):
        world_seed = rng.randint(0, 10**9)
        random.seed(world_seed)
        cave_geo = CaveGeometry(width=width, height=height)
        grid, layout = build_random_grid(width, height, num_pits=num_pits)

        for agent_name in ["baseline", "hybrid"]:
            world = World(cave_geo=cave_geo, grid=[row[:] for row in grid])
            world.grid = world.clone_grid()
            kb = KnowledgeBase(cave_geo=cave_geo)
            if agent_name == "baseline":
                agent = WumpusAgent(kb=kb)
            else:
                agent = HybridWumpusAgent(
                    kb=kb,
                    pit_model=pit_model,
                    risk_threshold=risk_threshold,
                    utility_floor=utility_floor,
                )

            result = run_game(
                world=world,
                agent=agent,
                max_steps=max_steps,
                trace=False,
                show_kb=False,
            )
            metrics = summarize_episode(
                result=result,
                used_arrow=not world.agent.has_arrow,
                risky_entries=world.risky_entries,
            )
            row = {
                "world_id": world_id,
                "agent": agent_name,
                **asdict(metrics),
                "layout": str(layout),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("agent").agg(
        average_score=("score", "mean"),
        median_score=("score", "median"),
        average_steps=("steps", "mean"),
        survival_rate=("survived", "mean"),
        escape_rate=("escaped", "mean"),
        gold_rate=("escaped_with_gold", "mean"),
        death_rate=("outcome", lambda s: (s == "dead").mean()),
        max_steps_rate=("outcome", lambda s: (s == "max_steps").mean()),
        arrow_usage_rate=("used_arrow", "mean"),
        average_risky_entries=("risky_entries", "mean"),
    )
    return grouped.reset_index()


def paired_differences(
    results_df: pd.DataFrame,
    metric: str,
    n_boot: int = 2000,
    seed: int = EVAL_SEED,
) -> dict[str, float | int | str]:
    """Bootstrap the paired hybrid - baseline difference for one metric."""
    # Cast to float so boolean metrics (survived, escaped, ...) can be subtracted.
    pivot = results_df.pivot(index="world_id", columns="agent", values=metric).astype(float)
    diff = (pivot["hybrid"] - pivot["baseline"]).dropna().to_numpy()
    if diff.size == 0:
        return {
            "metric": metric,
            "baseline_mean": float("nan"),
            "hybrid_mean": float("nan"),
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_worlds": 0,
        }

    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(diff, size=diff.size, replace=True).mean() for _ in range(n_boot)]
    )
    return {
        "metric": metric,
        "baseline_mean": float(pivot["baseline"].mean()),
        "hybrid_mean": float(pivot["hybrid"].mean()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_worlds": int(diff.size),
    }


if __name__ == "__main__":
    df = compare_agents(
        pit_model_path="artifacts/pit_risk_logreg.joblib",
        num_worlds=100,
        seed=EVAL_SEED,
    )
    print(df.head())
    print(aggregate_results(df))
