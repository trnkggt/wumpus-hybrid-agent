from collections.abc import Callable
import random

import pandas as pd

from app.agent.KB import KnowledgeBase
from app.agent.wumpus_agent import WumpusAgent
from app.domain.cave import CaveGeometry
from app.domain.cell import CellCoord
from app.domain.world import World
from app.main import build_random_grid, run_game
from app.ml.features import extract_cell_features


class LoggingWumpusAgent(WumpusAgent):
    def __init__(self, kb: KnowledgeBase, logger: Callable[[KnowledgeBase, World], None], start: CellCoord = CellCoord(0, 0)):
        super().__init__(kb=kb, start=start)
        self.logger = logger
        self.current_world: World | None = None

    def next_action(self, percept, state):
        self.kb.observe(state.pos, percept)
        if self.current_world is not None:
            self.logger(self.kb, self.current_world)
        return self._decide(percept, state)


def generate_frontier_dataset(
    num_worlds: int,
    width: int,
    height: int,
    num_pits: int,
    max_steps: int,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    for world_id in range(num_worlds):
        world_seed = rng.randint(0, 10**9)
        random.seed(world_seed)
        cave_geo = CaveGeometry(width=width, height=height)
        grid, _layout_info = build_random_grid(width, height, num_pits=num_pits)
        world = World(cave_geo=cave_geo, grid=grid)

        step_counter = {"value": 0}

        def logger(kb: KnowledgeBase, live_world: World) -> None:
            frontier = kb.frontier_cells()
            for cell in frontier:
                feats = extract_cell_features(
                    kb=kb,
                    candidate=cell,
                    agent_pos=live_world.agent.pos,
                    has_arrow=live_world.agent.has_arrow,
                    start=CellCoord(0, 0),
                )
                truth_cell = live_world.get_cell(cell)
                row = {
                    "world_id": world_id,
                    "step": step_counter["value"],
                    **feats,
                    "label_pit": int(truth_cell.has_pit),
                    "label_wumpus": int(truth_cell.has_wumpus and live_world.wumpus_alive),
                    "label_hazard": int(truth_cell.has_pit or (truth_cell.has_wumpus and live_world.wumpus_alive)),
                }
                rows.append(row)
            step_counter["value"] += 1

        kb = KnowledgeBase(cave_geo=cave_geo)
        agent = LoggingWumpusAgent(kb=kb, logger=logger)
        agent.current_world = world
        run_game(world=world, agent=agent, max_steps=max_steps, trace=False, show_kb=False)

    return pd.DataFrame(rows)
