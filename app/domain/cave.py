from dataclasses import dataclass

from app.domain.cell import CellCoord
from app.domain.direction import Direction


@dataclass(frozen=True)
class CaveGeometry:
    width: int
    height: int

    def in_bounds(self, c: CellCoord) -> bool:
        return 0 <= c.x < self.width and 0 <= c.y < self.height

    def neighbor_coords(self, pos: CellCoord) -> list[CellCoord]:
        candidates = [
            CellCoord(pos.x + 1, pos.y),
            CellCoord(pos.x - 1, pos.y),
            CellCoord(pos.x, pos.y + 1),
            CellCoord(pos.x, pos.y - 1),
        ]
        return [c for c in candidates if self.in_bounds(c)]

    def forward_cell(self, state_pos: CellCoord, state_dir: Direction) -> CellCoord | None:
        dx, dy = state_dir.delta()
        coords = CellCoord(state_pos.x + dx, state_pos.y + dy)
        if self.in_bounds(coords):
            return coords
        return None
