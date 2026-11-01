import pytest

from app.agent.KB import KnowledgeBase
from app.agent.agent_state import AgentState
from app.agent.hybrid_wumpus_agent import HybridWumpusAgent
from app.agent.wumpus_agent import WumpusAgent
from app.domain.cave import CaveGeometry
from app.domain.cell import Cell, CellCoord
from app.domain.direction import Direction
from app.domain.world import World
from app.main import run_game


class ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba_from_dict(self, features):
        return self.probability


def seed34_world():
    """The layout that reproduced the CLIMB stall before the planner fix."""
    geo = CaveGeometry(width=4, height=4)
    grid = [[Cell() for _ in range(4)] for _ in range(4)]
    for x, y in [(0, 2), (2, 0), (3, 3)]:
        grid[y][x].has_pit = True
    grid[1][1].has_wumpus = True
    grid[3][0].has_gold = True
    return geo, grid


def test_baseline_does_not_stall_on_the_seed34_layout():
    geo, grid = seed34_world()
    world = World(geo, grid)
    agent = WumpusAgent(KnowledgeBase(geo))

    result = run_game(world, agent, max_steps=100, trace=False, show_kb=False)

    assert result["outcome"] != "max_steps"
    ineffective_climbs = [
        step
        for step in result["history"]
        if step["action"].name == "CLIMB" and step["before_pos"] != CellCoord(0, 0)
    ]
    assert ineffective_climbs == []


def test_hybrid_smoke_with_dummy_model():
    geo, grid = seed34_world()
    world = World(geo, grid)
    agent = HybridWumpusAgent(
        kb=KnowledgeBase(geo),
        pit_model=ConstantModel(0.05),
        utility_floor=-100.0,
    )

    result = run_game(world, agent, max_steps=100, trace=False, show_kb=False)

    assert result["outcome"] in {
        "dead",
        "escaped_with_gold",
        "escaped_without_gold",
        "max_steps",
    }


def frontier_kb():
    kb = KnowledgeBase(CaveGeometry(width=4, height=4))
    kb.visited = {CellCoord(0, 0), CellCoord(1, 0), CellCoord(0, 1)}
    kb.no_pit = set(kb.visited)
    kb.no_wumpus = set(kb.visited)
    kb.safe = set(kb.visited)
    return kb


def test_risk_threshold_governs_frontier_acceptance():
    state = AgentState(pos=CellCoord(1, 0), dir=Direction.EAST, has_arrow=True)

    rejected = HybridWumpusAgent(
        kb=frontier_kb(),
        pit_model=ConstantModel(0.55),
        risk_threshold=0.5,
        utility_floor=-100.0,
    )
    assert rejected._choose_frontier_by_risk(state) is None

    accepted = HybridWumpusAgent(
        kb=frontier_kb(),
        pit_model=ConstantModel(0.62),
        risk_threshold=0.65,
        utility_floor=-100.0,
    )
    assert accepted._choose_frontier_by_risk(state) is not None


def test_risk_threshold_is_validated():
    with pytest.raises(ValueError):
        HybridWumpusAgent(
            kb=frontier_kb(),
            pit_model=ConstantModel(0.5),
            risk_threshold=1.5,
        )
