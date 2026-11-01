from app.domain.action import Action
from app.domain.cave import CaveGeometry
from app.domain.cell import Cell, CellCoord
from app.domain.direction import Direction
from app.domain.world import World


def make_world(width: int = 4, height: int = 4):
    geo = CaveGeometry(width=width, height=height)
    grid = [[Cell() for _ in range(width)] for _ in range(height)]
    return World(geo, grid), grid


def test_turns_change_direction_and_cost_one_point():
    world, _ = make_world()
    assert world.agent.dir == Direction.EAST

    world.apply_action(Action.TURN_LEFT)
    assert world.agent.dir == Direction.NORTH
    assert world.score == -1

    world.apply_action(Action.TURN_RIGHT)
    assert world.agent.dir == Direction.EAST
    assert world.score == -2


def test_move_forward_and_bump_at_wall():
    world, _ = make_world(width=2, height=2)

    result = world.apply_action(Action.MOVE_FORWARD)
    assert result.bump is False
    assert world.agent.pos == CellCoord(1, 0)
    assert world.score == -1

    result = world.apply_action(Action.MOVE_FORWARD)
    assert result.bump is True
    assert world.agent.pos == CellCoord(1, 0)
    assert world.score == -2


def test_moving_into_pit_kills_agent():
    geo = CaveGeometry(width=2, height=2)
    grid = [[Cell() for _ in range(2)] for _ in range(2)]
    grid[0][1].has_pit = True
    world = World(geo, grid)

    result = world.apply_action(Action.MOVE_FORWARD)

    assert result.agent_dead is True
    assert world.agent_alive is False
    assert world.score == -1001


def test_moving_into_wumpus_kills_agent():
    geo = CaveGeometry(width=2, height=2)
    grid = [[Cell() for _ in range(2)] for _ in range(2)]
    grid[0][1].has_wumpus = True
    world = World(geo, grid)

    result = world.apply_action(Action.MOVE_FORWARD)

    assert result.agent_dead is True
    assert world.agent_alive is False
    assert world.score == -1001


def test_grab_gold_then_climb_out_scores_bonus():
    geo = CaveGeometry(width=2, height=2)
    grid = [[Cell() for _ in range(2)] for _ in range(2)]
    grid[0][1].has_gold = True
    world = World(geo, grid)

    world.apply_action(Action.MOVE_FORWARD)  # (1, 0)
    result = world.apply_action(Action.GRAB)
    assert result.gold_collected_now is True
    assert world.agent.has_gold is True
    assert grid[0][1].has_gold is False

    world.apply_action(Action.TURN_LEFT)   # NORTH
    world.apply_action(Action.TURN_LEFT)   # WEST
    world.apply_action(Action.MOVE_FORWARD)  # back to (0, 0)
    result = world.apply_action(Action.CLIMB)

    assert result.climbed_out is True
    assert world.score == 994  # six actions at -1, plus the +1000 gold bonus


def test_climb_without_gold_does_not_score_bonus():
    world, _ = make_world(width=2, height=2)

    result = world.apply_action(Action.CLIMB)

    assert result.climbed_out is True
    assert world.score == -1


def test_shoot_kills_wumpus_and_consumes_arrow():
    geo = CaveGeometry(width=3, height=1)
    grid = [[Cell() for _ in range(3)]]
    grid[0][2].has_wumpus = True
    world = World(geo, grid)

    result = world.apply_action(Action.SHOOT)

    assert result.scream is True
    assert world.wumpus_alive is False
    assert world.agent.has_arrow is False
    assert world.score == -11  # action cost plus the 10-point arrow cost

    assert world.compute_percept().scream is True
    assert world.compute_percept().scream is False


def test_percepts_reflect_adjacent_hazards_and_gold():
    geo = CaveGeometry(width=2, height=2)
    grid = [[Cell() for _ in range(2)] for _ in range(2)]
    grid[0][1].has_pit = True      # (1, 0), adjacent to start
    grid[1][0].has_wumpus = True   # (0, 1), adjacent to start
    grid[0][0].has_gold = True     # under the agent
    world = World(geo, grid)

    percept = world.compute_percept()

    assert percept.breeze is True
    assert percept.stench is True
    assert percept.glitter is True
