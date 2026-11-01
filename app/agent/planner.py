from collections import deque
from dataclasses import dataclass
from typing import Callable

from app.domain.action import Action
from app.domain.cave import CaveGeometry
from app.domain.cell import CellCoord
from app.domain.direction import Direction


@dataclass(frozen=True)
class NavState:
    pos: CellCoord
    dir: Direction


def successors(
    state: NavState,
    allowed: set[CellCoord],
    cave_geo: CaveGeometry,
) -> list[tuple[Action, NavState]]:
    out: list[tuple[Action, NavState]] = []

    out.append((Action.TURN_LEFT, NavState(state.pos, state.dir.left())))
    out.append((Action.TURN_RIGHT, NavState(state.pos, state.dir.right())))

    forward = cave_geo.forward_cell(state.pos, state.dir)
    if forward is not None and forward in allowed:
        out.append((Action.MOVE_FORWARD, NavState(forward, state.dir)))

    return out


def search_nav(
    start: NavState,
    is_goal: Callable[[NavState], bool],
    allowed: set[CellCoord],
    cave_geo: CaveGeometry,
) -> deque[Action]:
    if is_goal(start):
        return deque()

    queue = deque([(start, deque())])
    seen = {start}

    while queue:
        state, path = queue.popleft()

        for action, nxt in successors(state, allowed, cave_geo):
            if nxt in seen:
                continue

            new_path = deque(path)
            new_path.append(action)

            if is_goal(nxt):
                return new_path

            seen.add(nxt)
            queue.append((nxt, new_path))

    return deque()


def plan_route(
    start_pos: CellCoord,
    start_dir: Direction,
    goals: set[CellCoord],
    allowed: set[CellCoord],
    cave_geo: CaveGeometry,
) -> deque[Action]:
    start = NavState(start_pos, start_dir)
    return search_nav(
        start=start,
        is_goal=lambda s: s.pos in goals,
        allowed=allowed,
        cave_geo=cave_geo,
    )


def firing_states_for_wumpus(
    possible_wumpus: set[CellCoord],
    allowed: set[CellCoord],
) -> set[NavState]:
    states: set[NavState] = set()

    for w in possible_wumpus:
        for pos in allowed:
            if pos == w:
                continue

            if pos.x == w.x:
                if pos.y < w.y:
                    states.add(NavState(pos, Direction.NORTH))
                elif pos.y > w.y:
                    states.add(NavState(pos, Direction.SOUTH))
            elif pos.y == w.y:
                if pos.x < w.x:
                    states.add(NavState(pos, Direction.EAST))
                elif pos.x > w.x:
                    states.add(NavState(pos, Direction.WEST))

    return states


def plan_shot(
    start_pos: CellCoord,
    start_dir: Direction,
    possible_wumpus: set[CellCoord],
    allowed: set[CellCoord],
    cave_geo: CaveGeometry,
) -> deque[Action]:
    if not possible_wumpus:
        return deque()

    start = NavState(start_pos, start_dir)
    firing_states = firing_states_for_wumpus(
        possible_wumpus=possible_wumpus,
        allowed=allowed,
    )

    if not firing_states:
        return deque()

    # search_nav returns an empty deque both when the start already satisfies
    # the goal and when no path exists.  Handle the already-aligned case here,
    # otherwise the agent would treat a valid shot as unreachable.
    if start in firing_states:
        return deque([Action.SHOOT])

    route = search_nav(
        start=start,
        is_goal=lambda s: s in firing_states,
        allowed=allowed,
        cave_geo=cave_geo,
    )

    if not route:
        return deque()

    route.append(Action.SHOOT)
    return route


def simulate_nav_actions(
    start_pos: CellCoord,
    start_dir: Direction,
    actions: deque[Action],
    cave_geo: CaveGeometry,
) -> NavState:
    pos = start_pos
    direction = start_dir

    for action in actions:
        if action == Action.TURN_LEFT:
            direction = direction.left()
        elif action == Action.TURN_RIGHT:
            direction = direction.right()
        elif action == Action.MOVE_FORWARD:
            nxt = cave_geo.forward_cell(pos, direction)
            if nxt is not None:
                pos = nxt

    return NavState(pos=pos, dir=direction)


def turns_to_face(from_dir: Direction, target_dir: Direction) -> deque[Action]:
    if from_dir == target_dir:
        return deque()

    left_once = from_dir.left()
    right_once = from_dir.right()
    if left_once == target_dir:
        return deque([Action.TURN_LEFT])
    if right_once == target_dir:
        return deque([Action.TURN_RIGHT])
    return deque([Action.TURN_LEFT, Action.TURN_LEFT])


def direction_toward(from_pos: CellCoord, to_pos: CellCoord) -> Direction:
    dx = to_pos.x - from_pos.x
    dy = to_pos.y - from_pos.y
    if dx == 1 and dy == 0:
        return Direction.EAST
    if dx == -1 and dy == 0:
        return Direction.WEST
    if dx == 0 and dy == 1:
        return Direction.NORTH
    if dx == 0 and dy == -1:
        return Direction.SOUTH
    raise ValueError("Cells are not adjacent.")


def plan_route_to_adjacent_staging_cell(
    start_pos: CellCoord,
    start_dir: Direction,
    target_cell: CellCoord,
    allowed: set[CellCoord],
    cave_geo: CaveGeometry,
) -> tuple[deque[Action], CellCoord] | tuple[deque[Action], None]:
    staging_cells = {c for c in cave_geo.neighbor_coords(target_cell) if c in allowed}
    if not staging_cells:
        return deque(), None

    best_route: deque[Action] | None = None
    best_staging: CellCoord | None = None
    for staging in staging_cells:
        route = plan_route(
            start_pos=start_pos,
            start_dir=start_dir,
            goals={staging},
            allowed=allowed,
            cave_geo=cave_geo,
        )
        if not route and staging != start_pos:
            continue
        if best_route is None or len(route) < len(best_route):
            best_route = route
            best_staging = staging

    if best_route is None or best_staging is None:
        return deque(), None
    return best_route, best_staging
