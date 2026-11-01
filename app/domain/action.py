from enum import Enum, auto


class Action(Enum):
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    MOVE_FORWARD = auto()
    GRAB = auto()
    SHOOT = auto()
    CLIMB = auto()
    NO_OP = auto()
