import random

from app.agent.KB import KnowledgeBase
from app.agent.hybrid_wumpus_agent import HybridWumpusAgent
from app.agent.wumpus_agent import WumpusAgent
from app.domain.cave import CaveGeometry
from app.domain.cell import Cell, CellCoord
from app.domain.world import World
from app.ml.models import PitRiskModel


def build_random_grid(
    width: int,
    height: int,
    num_pits: int = 3,
) -> tuple[list[list[Cell]], dict[str, object]]:
    grid = [[Cell() for _ in range(width)] for _ in range(height)]

    available_positions = [
        (x, y)
        for x in range(width)
        for y in range(height)
        if (x, y) != (0, 0)
    ]
    random.shuffle(available_positions)

    if num_pits + 2 > len(available_positions):
        raise ValueError("Too many pits for the world size.")

    pit_positions = available_positions[:num_pits]
    wumpus_pos = available_positions[num_pits]
    gold_pos = available_positions[num_pits + 1]

    for x, y in pit_positions:
        grid[y][x].has_pit = True

    wx, wy = wumpus_pos
    grid[wy][wx].has_wumpus = True

    gx, gy = gold_pos
    grid[gy][gx].has_gold = True

    layout_info = {
        "pits": pit_positions,
        "wumpus": [wumpus_pos],
        "gold": gold_pos,
    }
    return grid, layout_info


def print_world_layout(width: int, height: int, layout_info: dict, start: tuple[int, int] = (0, 0)) -> None:
    print("\nWORLD LAYOUT")
    print(f"  Pits   : {layout_info['pits']}")
    print(f"  Wumpus : {layout_info['wumpus']}")
    print(f"  Gold   : {layout_info['gold']}")
    print(f"  Grid (y={height - 1} top, y=0 bottom, x=0 left):")

    for y in range(height - 1, -1, -1):
        row = f"  y={y} | "
        for x in range(width):
            if (x, y) == start:
                cell_str = "A"
            elif (x, y) == layout_info["gold"]:
                cell_str = "G"
            elif (x, y) in layout_info["wumpus"]:
                cell_str = "W"
            elif (x, y) in layout_info["pits"]:
                cell_str = "P"
            else:
                cell_str = "."
            row += f" {cell_str} "
        print(row)

    print("=" * 60)


def print_kb_debug(kb: KnowledgeBase) -> None:
    def sort_cells(cells):
        return sorted(cells, key=lambda c: (c.y, c.x))

    print("KB")
    print("  visited         :", sort_cells(kb.visited))
    print("  safe            :", sort_cells(kb.safe))
    print("  no_pit          :", sort_cells(kb.no_pit))
    print("  no_wumpus       :", sort_cells(kb.no_wumpus))
    print("  possible_pit    :", {k: v for k, v in sorted(kb.possible_pit.items(), key=lambda kv: (kv[0].y, kv[0].x))})
    print("  possible_wumpus :", {k: v for k, v in sorted(kb.possible_wumpus.items(), key=lambda kv: (kv[0].y, kv[0].x))})
    print("  wumpus          :", kb.wumpus)
    print("  wumpus_dead     :", kb.wumpus_dead)


def run_game(
    world: World,
    agent,
    max_steps: int = 100,
    trace: bool = True,
    show_kb: bool = True,
):
    history = []

    for step in range(max_steps):
        before_pos = world.agent.pos
        before_dir = world.agent.dir
        percept = world.compute_percept()

        action = agent.next_action(percept, world.agent)
        if action.name == "MOVE_FORWARD":
            next_pos = world.cave_geo.forward_cell(world.agent.pos, world.agent.dir)
            if next_pos is not None and next_pos not in agent.kb.safe:
                world.risky_entries += 1
        result = world.apply_action(action)

        step_info = {
            "step": step,
            "before_pos": before_pos,
            "before_dir": before_dir,
            "after_pos": world.agent.pos,
            "after_dir": world.agent.dir,
            "percept": percept,
            "action": action,
            "result": result,
            "has_gold": world.agent.has_gold,
            "has_arrow": world.agent.has_arrow,
            "score": world.score,
        }
        history.append(step_info)

        if trace:
            print(
                f"step={step:02d} "
                f"before_pos={before_pos} "
                f"before_dir={before_dir.name:<5} "
                f"percept={percept} "
                f"action={action.name:<12} "
                f"after_pos={world.agent.pos} "
                f"after_dir={world.agent.dir.name:<5} "
                f"score={world.score}"
            )
            if show_kb:
                print_kb_debug(agent.kb)
                print("-" * 60)

        if result.agent_dead:
            if trace:
                print(f"FINAL RESULT: agent died, score={world.score}")
            return {
                "outcome": "dead",
                "steps": step + 1,
                "has_gold": world.agent.has_gold,
                "score": world.score,
                "history": history,
            }

        if result.climbed_out:
            if trace:
                print(f"FINAL RESULT: climbed out, has_gold={world.agent.has_gold}, score={world.score}")
            return {
                "outcome": "escaped_with_gold" if world.agent.has_gold else "escaped_without_gold",
                "steps": step + 1,
                "has_gold": world.agent.has_gold,
                "score": world.score,
                "history": history,
            }

    if trace:
        print(f"FINAL RESULT: max steps reached, score={world.score}")

    return {
        "outcome": "max_steps",
        "steps": max_steps,
        "has_gold": world.agent.has_gold,
        "score": world.score,
        "history": history,
    }


def main(seed: int = 42) -> None:
    width, height = 4, 4
    cave_geo = CaveGeometry(width=width, height=height)

    random.seed(seed)
    grid, layout_info = build_random_grid(width, height, num_pits=3)
    print(f"\nSEED: {seed}")
    print_world_layout(width, height, layout_info, start=(0, 0))

    world = World(cave_geo=cave_geo, grid=grid)
    kb = KnowledgeBase(cave_geo=cave_geo)
    agent = WumpusAgent(kb=kb)
    result = run_game(world, agent=agent, max_steps=100, trace=True, show_kb=False)

    print("\nFINAL RESULT")
    print({key: result[key] for key in ("outcome", "steps", "has_gold", "score")})


def hybrid_demo(
    model_path: str = "artifacts/pit_risk_logreg.joblib",
    seed: int = 42,
) -> None:
    width, height = 4, 4
    cave_geo = CaveGeometry(width=width, height=height)
    random.seed(seed)
    grid, layout_info = build_random_grid(width, height, num_pits=3)
    print(f"\nSEED: {seed}")
    print_world_layout(width, height, layout_info, start=(0, 0))

    world = World(cave_geo=cave_geo, grid=grid)
    kb = KnowledgeBase(cave_geo=cave_geo)
    pit_model = PitRiskModel.load(model_path)
    agent = HybridWumpusAgent(kb=kb, pit_model=pit_model)
    result = run_game(world, agent=agent, max_steps=100, trace=True, show_kb=False)

    print("\nHYBRID FINAL RESULT")
    print({key: result[key] for key in ("outcome", "steps", "has_gold", "score")})


if __name__ == "__main__":
    main()
