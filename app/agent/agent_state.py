from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from app.domain.action import Action
from app.domain.cell import CellCoord
from app.domain.direction import Direction


@dataclass
class AgentState:
    pos: CellCoord = CellCoord(0, 0)
    dir: Direction = Direction.EAST
    has_arrow: bool = True
    has_gold: bool = False
    risk_tolerance: float = 0.25

    visited: set[CellCoord] = field(default_factory=set)
    backtrack_stack: list[CellCoord] = field(default_factory=list)
    plan: Deque[Action] = field(default_factory=deque)
