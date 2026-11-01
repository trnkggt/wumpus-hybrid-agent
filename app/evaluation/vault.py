"""Gold-vault diagnostic.

A 3x3 world where the gold sits in a safe cell surrounded by breeze cells, so
the baseline cannot prove it safe and never enters it.  The hybrid's predicted
pit probability for that cell shows whether the classifier can represent the
global logical constraint that other pits explain the breezes.

Usage:
    uv run python -m app.evaluation.vault --threshold 0.6
"""

import argparse
from pathlib import Path

import pandas as pd

from app.agent.KB import KnowledgeBase
from app.agent.hybrid_wumpus_agent import HybridWumpusAgent
from app.agent.wumpus_agent import WumpusAgent
from app.domain.cave import CaveGeometry
from app.domain.cell import Cell, CellCoord
from app.domain.world import World
from app.main import run_game
from app.ml.features import extract_cell_features
from app.ml.models import PitRiskModel

GEO = CaveGeometry(width=3, height=3)
GOLD = CellCoord(1, 1)
MODEL_NAMES = ["logreg", "tree", "random_forest", "gradient_boosting"]


def build_vault_grid() -> list[list[Cell]]:
    grid = [[Cell() for _ in range(3)] for _ in range(3)]
    for x, y in [(0, 2), (2, 0), (2, 2)]:
        grid[y][x].has_pit = True
    grid[1][2].has_wumpus = True   # (2, 1)
    grid[1][1].has_gold = True     # the vault
    return grid


class LoggingHybrid(HybridWumpusAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vault_probabilities: list[float] = []

    def _choose_frontier_by_risk(self, state):
        if GOLD in self.kb.frontier_cells():
            features = extract_cell_features(
                kb=self.kb,
                candidate=GOLD,
                agent_pos=state.pos,
                has_arrow=state.has_arrow,
                start=self.start,
            )
            self.vault_probabilities.append(
                self.pit_model.predict_proba_from_dict(features)
            )
        return super()._choose_frontier_by_risk(state)


def baseline_result() -> dict:
    world = World(GEO, build_vault_grid())
    return run_game(
        world,
        WumpusAgent(KnowledgeBase(GEO)),
        max_steps=100,
        trace=False,
        show_kb=False,
    )


def hybrid_result(model_path: str, threshold: float, utility_floor: float) -> tuple[dict, float]:
    world = World(GEO, build_vault_grid())
    agent = LoggingHybrid(
        KnowledgeBase(GEO),
        PitRiskModel.load(model_path),
        risk_threshold=threshold,
        utility_floor=utility_floor,
    )
    result = run_game(world, agent, max_steps=100, trace=False, show_kb=False)
    probability = agent.vault_probabilities[0] if agent.vault_probabilities else float("nan")
    return result, probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gold-vault diagnostic.")
    parser.add_argument("--model-dir", default="artifacts")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--utility-floor", type=float, default=-8.0)
    parser.add_argument("--output", default="artifacts/vault_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline = baseline_result()
    rows = [{
        "model": "baseline",
        "threshold": None,
        "vault_pit_prob": None,
        "outcome": baseline["outcome"],
        "gold": baseline["has_gold"],
        "score": baseline["score"],
    }]

    for name in MODEL_NAMES:
        path = Path(args.model_dir) / f"pit_risk_{name}.joblib"
        if not path.exists():
            print(f"skip {name}: {path} missing (run app.ml.train first)")
            continue
        result, probability = hybrid_result(
            str(path),
            threshold=args.threshold,
            utility_floor=args.utility_floor,
        )
        rows.append({
            "model": name,
            "threshold": args.threshold,
            "vault_pit_prob": probability,
            "outcome": result["outcome"],
            "gold": result["has_gold"],
            "score": result["score"],
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)


if __name__ == "__main__":
    main()
