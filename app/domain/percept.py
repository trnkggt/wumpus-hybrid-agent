from dataclasses import dataclass


@dataclass(frozen=True)
class Percept:
    breeze: bool = False
    stench: bool = False
    glitter: bool = False
    bump: bool = False
    scream: bool = False
