from dataclasses import dataclass

from app.agent.agent_state import AgentState
from app.domain.action import Action
from app.domain.cave import CaveGeometry
from app.domain.cell import Cell, CellCoord
from app.domain.direction import Direction
from app.domain.percept import Percept


@dataclass
class WorldStepResult:
    bump: bool = False
    scream: bool = False
    agent_dead: bool = False
    gold_collected_now: bool = False
    climbed_out: bool = False


class World:
    def __init__(self, cave_geo: CaveGeometry, grid: list[list[Cell]]):
        self.cave_geo: CaveGeometry = cave_geo
        self.grid: list[list[Cell]] = grid
        self.agent: AgentState = AgentState(
            pos=CellCoord(0, 0),
            dir=Direction.EAST,
            has_arrow=True,
            has_gold=False,
        )

        self.wumpus_alive = any(
            self.grid[y][x].has_wumpus
            for y in range(cave_geo.height)
            for x in range(cave_geo.width)
        )
        self.gold_collected: bool = False
        self.agent_alive: bool = True

        self._bump_last: bool = False
        self._scream_last: bool = False

        self.score: int = 0
        self.risky_entries: int = 0

    def clone_grid(self) -> list[list[Cell]]:
        return [
            [Cell(has_pit=cell.has_pit, has_wumpus=cell.has_wumpus, has_gold=cell.has_gold) for cell in row]
            for row in self.grid
        ]

    def get_cell(self, pos: CellCoord) -> Cell:
        return self.grid[pos.y][pos.x]

    def place_pit(self, pos: CellCoord) -> None:
        self.get_cell(pos).has_pit = True

    def place_wumpus(self, pos: CellCoord) -> None:
        self.get_cell(pos).has_wumpus = True
        self.wumpus_alive = True

    def place_gold(self, pos: CellCoord) -> None:
        self.get_cell(pos).has_gold = True

    def compute_percept(self) -> Percept:
        if not self.agent_alive:
            return Percept()

        current = self.agent.pos
        neighbors = self.cave_geo.neighbor_coords(current)

        breeze = any(self.get_cell(c).has_pit for c in neighbors)
        stench = self.wumpus_alive and any(self.get_cell(c).has_wumpus for c in neighbors)
        glitter = (not self.gold_collected) and self.get_cell(current).has_gold

        percept = Percept(
            breeze=breeze,
            stench=stench,
            glitter=glitter,
            bump=self._bump_last,
            scream=self._scream_last,
        )

        self._bump_last = False
        self._scream_last = False
        return percept

    def _shoot_arrow(self, pos: CellCoord, dir: Direction) -> bool:
        if not self.wumpus_alive:
            return False

        dx, dy = dir.delta()
        x, y = pos.x, pos.y

        while True:
            x += dx
            y += dy
            current = CellCoord(x, y)

            if not self.cave_geo.in_bounds(current):
                return False

            cell = self.get_cell(current)
            if cell.has_wumpus:
                cell.has_wumpus = False
                return True

    def apply_action(self, action: Action) -> WorldStepResult:
        res = WorldStepResult()

        if not self.agent_alive:
            res.agent_dead = True
            return res

        self.score -= 1

        match action:
            case Action.TURN_LEFT:
                self.agent.dir = self.agent.dir.left()
                return res

            case Action.TURN_RIGHT:
                self.agent.dir = self.agent.dir.right()
                return res

            case Action.MOVE_FORWARD:
                next_pos = self.cave_geo.forward_cell(self.agent.pos, self.agent.dir)
                if next_pos is None:
                    res.bump = True
                    self._bump_last = True
                    return res

                self.agent.pos = next_pos
                cell = self.get_cell(next_pos)

                if cell.has_pit:
                    self.agent_alive = False
                    res.agent_dead = True
                    self.score -= 1000
                    return res

                if self.wumpus_alive and cell.has_wumpus:
                    self.agent_alive = False
                    res.agent_dead = True
                    self.score -= 1000
                    return res

                return res

            case Action.GRAB:
                cell = self.get_cell(self.agent.pos)
                if cell.has_gold and not self.gold_collected:
                    self.gold_collected = True
                    self.agent.has_gold = True
                    cell.has_gold = False
                    res.gold_collected_now = True
                return res

            case Action.SHOOT:
                if not self.agent.has_arrow:
                    return res

                self.score -= 10
                self.agent.has_arrow = False
                killed = self._shoot_arrow(self.agent.pos, self.agent.dir)

                if killed:
                    self.wumpus_alive = False
                    self._scream_last = True
                    res.scream = True

                return res

            case Action.CLIMB:
                if self.agent.pos == CellCoord(0, 0):
                    res.climbed_out = True
                    if self.agent.has_gold:
                        self.score += 1000
                return res

            case Action.NO_OP:
                return res
