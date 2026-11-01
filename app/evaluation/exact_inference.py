"""Exact pit posterior for small Wumpus worlds.

Enumerates every pit assignment consistent with the knowledge base's breeze
observations and the known pit count.  This is used as a calibration check on
the learned classifier: it answers whether the model's refusal to enter a
frontier is due to overconfidence or to irreducible uncertainty.

Usage:
    uv run python -m app.evaluation.exact_inference
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from app.agent.KB import KnowledgeBase
from app.agent.hybrid_wumpus_agent import HybridWumpusAgent
from app.domain.cave import CaveGeometry
from app.domain.cell import CellCoord
from app.domain.world import World
from app.evaluation.vault import GEO, GOLD, build_vault_grid
from app.main import run_game
from app.ml.features import extract_cell_features
from app.ml.models import PitRiskModel


def all_cells(geo: CaveGeometry) -> list[CellCoord]:
    return [CellCoord(x, y) for x in range(geo.width) for y in range(geo.height)]


def exact_pit_probabilities(
    kb: KnowledgeBase,
    num_pits: int,
) -> tuple[dict[CellCoord, float], int]:
    """Return each unresolved cell's pit posterior and the number of valid worlds.

    This is a pit-only posterior: the Wumpus is handled by the symbolic layer.
    Known pits are fixed; all other cells not ruled out by a no-breeze
    observation are enumerated, subject to every breeze having at least one
    adjacent pit and to the known total pit count.
    """
    known_pits = set(kb.pit)
    remaining = num_pits - len(known_pits)
    if remaining < 0:
        return {}, 0

    unknown = [
        cell
        for cell in all_cells(kb.cave_geo)
        if cell not in kb.no_pit and cell not in kb.pit
    ]
    breeze_neighbors = [
        set(kb.cave_geo.neighbor_coords(cell))
        for cell in kb.visited
        if cell in kb.breeze_cells
    ]

    counts = {cell: 0 for cell in unknown}
    total = 0
    for combo in combinations(unknown, remaining):
        selected = known_pits | set(combo)
        if any(not (neighbors & selected) for neighbors in breeze_neighbors):
            continue
        total += 1
        for cell in combo:
            counts[cell] += 1

    probabilities = {
        cell: (counts[cell] / total if total else 0.0)
        for cell in unknown
    }
    return probabilities, total


class ExactLoggingHybrid(HybridWumpusAgent):
    def __init__(self, *args, num_pits: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_pits = num_pits
        self.exact_vault_prob: float | None = None
        self.model_vault_prob: float | None = None
        self.valid_assignments: int | None = None

    def _choose_frontier_by_risk(self, state):
        if self.model_vault_prob is None and GOLD in self.kb.frontier_cells():
            probabilities, total = exact_pit_probabilities(self.kb, self.num_pits)
            self.exact_vault_prob = probabilities.get(GOLD)
            self.valid_assignments = total
            features = extract_cell_features(
                kb=self.kb,
                candidate=GOLD,
                agent_pos=state.pos,
                has_arrow=state.has_arrow,
                start=self.start,
            )
            self.model_vault_prob = self.pit_model.predict_proba_from_dict(features)
        return super()._choose_frontier_by_risk(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact pit posterior diagnostic.")
    parser.add_argument("--model", default="artifacts/pit_risk_logreg.joblib")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--utility-floor", type=float, default=-8.0)
    parser.add_argument("--num-pits", type=int, default=3)
    parser.add_argument("--output", default="artifacts/exact_inference.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    world = World(GEO, build_vault_grid())
    agent = ExactLoggingHybrid(
        KnowledgeBase(GEO),
        PitRiskModel.load(args.model),
        risk_threshold=args.threshold,
        utility_floor=args.utility_floor,
        num_pits=args.num_pits,
    )
    result = run_game(world, agent, max_steps=100, trace=False, show_kb=False)

    row = {
        "model": args.model,
        "threshold": args.threshold,
        "num_pits": args.num_pits,
        "valid_assignments": agent.valid_assignments,
        "exact_vault_pit_prob": agent.exact_vault_prob,
        "model_vault_pit_prob": agent.model_vault_prob,
        "outcome": result["outcome"],
        "gold": result["has_gold"],
        "score": result["score"],
    }
    table = pd.DataFrame([row])
    print(table.to_string(index=False))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)


if __name__ == "__main__":
    main()
