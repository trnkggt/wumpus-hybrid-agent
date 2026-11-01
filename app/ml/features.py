from app.agent.KB import KnowledgeBase
from app.domain.cell import CellCoord


FEATURE_COLUMNS = [
    "cell_x",
    "cell_y",
    "visited_neighbor_count",
    "breeze_neighbor_count",
    "no_breeze_neighbor_count",
    "stench_neighbor_count",
    "no_stench_neighbor_count",
    "safe_neighbor_count",
    "possible_pit_flag",
    "possible_wumpus_flag",
    "possible_pit_support",
    "possible_wumpus_support",
    "known_safe_flag",
    "known_pit_flag",
    "known_wumpus_flag",
    "wumpus_dead_flag",
    "frontier_flag",
    "visited_total",
    "manhattan_from_agent",
    "manhattan_from_start",
    "has_arrow_flag",
]


def manhattan(a: CellCoord, b: CellCoord) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def extract_cell_features(
    kb: KnowledgeBase,
    candidate: CellCoord,
    agent_pos: CellCoord,
    has_arrow: bool,
    start: CellCoord = CellCoord(0, 0),
) -> dict[str, float | int]:
    summary = kb.candidate_summary(candidate)
    features = {
        "cell_x": candidate.x,
        "cell_y": candidate.y,
        "visited_neighbor_count": int(summary["visited_neighbor_count"]),
        "breeze_neighbor_count": int(summary["breeze_neighbor_count"]),
        "no_breeze_neighbor_count": int(summary["no_breeze_neighbor_count"]),
        "stench_neighbor_count": int(summary["stench_neighbor_count"]),
        "no_stench_neighbor_count": int(summary["no_stench_neighbor_count"]),
        "safe_neighbor_count": int(summary["safe_neighbor_count"]),
        "possible_pit_flag": int(bool(summary["possible_pit_flag"])),
        "possible_wumpus_flag": int(bool(summary["possible_wumpus_flag"])),
        "possible_pit_support": int(summary["possible_pit_support"]),
        "possible_wumpus_support": int(summary["possible_wumpus_support"]),
        "known_safe_flag": int(bool(summary["known_safe_flag"])),
        "known_pit_flag": int(bool(summary["known_pit_flag"])),
        "known_wumpus_flag": int(bool(summary["known_wumpus_flag"])),
        "wumpus_dead_flag": int(bool(summary["wumpus_dead_flag"])),
        "frontier_flag": int(bool(summary["frontier_flag"])),
        "visited_total": int(summary["visited_total"]),
        "manhattan_from_agent": manhattan(candidate, agent_pos),
        "manhattan_from_start": manhattan(candidate, start),
        "has_arrow_flag": int(bool(has_arrow)),
    }
    return features
