from collections import deque

from app.agent.planner import (
    direction_toward,
    plan_route,
    plan_shot,
    simulate_nav_actions,
    turns_to_face,
)
from app.domain.action import Action
from app.domain.cave import CaveGeometry
from app.domain.cell import CellCoord
from app.domain.direction import Direction


def test_plan_route_reaches_goal():
    geo = CaveGeometry(width=4, height=4)
    allowed = {CellCoord(x, y) for x in range(4) for y in range(4)}

    route = plan_route(
        start_pos=CellCoord(0, 0),
        start_dir=Direction.EAST,
        goals={CellCoord(3, 0)},
        allowed=allowed,
        cave_geo=geo,
    )

    assert route
    final = simulate_nav_actions(CellCoord(0, 0), Direction.EAST, route, geo)
    assert final.pos == CellCoord(3, 0)


def test_plan_shot_returns_shoot_when_already_aligned():
    """Regression: an already-aligned agent must shoot, not report 'no path'."""
    geo = CaveGeometry(width=4, height=1)
    allowed = {CellCoord(x, 0) for x in range(4)}

    route = plan_shot(
        start_pos=CellCoord(1, 0),
        start_dir=Direction.EAST,
        possible_wumpus={CellCoord(3, 0)},
        allowed=allowed,
        cave_geo=geo,
    )

    assert route == deque([Action.SHOOT])


def test_plan_shot_navigates_then_shoots():
    geo = CaveGeometry(width=4, height=4)
    allowed = {
        CellCoord(0, 0),
        CellCoord(1, 0),
        CellCoord(2, 0),
        CellCoord(3, 0),
        CellCoord(0, 1),
    }

    route = plan_shot(
        start_pos=CellCoord(0, 1),
        start_dir=Direction.EAST,
        possible_wumpus={CellCoord(3, 0)},
        allowed=allowed,
        cave_geo=geo,
    )

    assert route
    assert route[-1] == Action.SHOOT


def test_plan_shot_returns_empty_when_no_firing_state_is_reachable():
    geo = CaveGeometry(width=4, height=4)

    route = plan_shot(
        start_pos=CellCoord(0, 0),
        start_dir=Direction.EAST,
        possible_wumpus={CellCoord(3, 3)},
        allowed={CellCoord(0, 0)},
        cave_geo=geo,
    )

    assert not route


def test_turns_to_face():
    assert list(turns_to_face(Direction.EAST, Direction.NORTH)) == [Action.TURN_LEFT]
    assert list(turns_to_face(Direction.EAST, Direction.SOUTH)) == [Action.TURN_RIGHT]
    assert list(turns_to_face(Direction.EAST, Direction.WEST)) == [
        Action.TURN_LEFT,
        Action.TURN_LEFT,
    ]
    assert list(turns_to_face(Direction.EAST, Direction.EAST)) == []


def test_direction_toward():
    assert direction_toward(CellCoord(0, 0), CellCoord(1, 0)) == Direction.EAST
    assert direction_toward(CellCoord(0, 0), CellCoord(-1, 0)) == Direction.WEST
    assert direction_toward(CellCoord(0, 0), CellCoord(0, 1)) == Direction.NORTH
    assert direction_toward(CellCoord(0, 0), CellCoord(0, -1)) == Direction.SOUTH
