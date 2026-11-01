from enum import Enum, auto


class TurnSide(Enum):
    LEFT = "Left"
    RIGHT = "Right"


class Direction(Enum):
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()

    def _direction_guide(self, l_or_r: TurnSide) -> dict["Direction", "Direction"]:
        if l_or_r is TurnSide.LEFT:
            return {
                Direction.NORTH: Direction.WEST,
                Direction.SOUTH: Direction.EAST,
                Direction.EAST: Direction.NORTH,
                Direction.WEST: Direction.SOUTH,
            }
        return {
            Direction.NORTH: Direction.EAST,
            Direction.EAST: Direction.SOUTH,
            Direction.SOUTH: Direction.WEST,
            Direction.WEST: Direction.NORTH,
        }

    def delta(self) -> tuple[int, int]:
        return {
            Direction.NORTH: (0, 1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, -1),
            Direction.WEST: (-1, 0),
        }[self]

    def left(self) -> "Direction":
        return self._direction_guide(TurnSide.LEFT)[self]

    def right(self) -> "Direction":
        return self._direction_guide(TurnSide.RIGHT)[self]
