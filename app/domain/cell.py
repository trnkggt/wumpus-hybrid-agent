from dataclasses import dataclass


@dataclass(frozen=True)
class CellCoord:
    x: int
    y: int


@dataclass
class Cell:
    has_pit: bool = False
    has_wumpus: bool = False
    has_gold: bool = False
