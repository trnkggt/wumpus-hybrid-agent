from dataclasses import dataclass

from app.domain.cell import CellCoord


@dataclass(frozen=True)
class FrontierDecision:
    cell: CellCoord
    pit_prob: float
    wumpus_prob: float
    combined_risk: float
    path_cost: int
    utility: float


def combined_risk(pit_prob: float, wumpus_prob: float) -> float:
    return 1.0 - (1.0 - pit_prob) * (1.0 - wumpus_prob)


def utility_score(
    pit_prob: float,
    wumpus_prob: float,
    path_cost: int,
    risk_weight: float = 10.0,
    cost_weight: float = 1.0,
) -> float:
    risk = combined_risk(pit_prob, wumpus_prob)
    return -(risk_weight * risk + cost_weight * path_cost)
