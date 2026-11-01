from dataclasses import dataclass, field

from app.domain.cave import CaveGeometry
from app.domain.cell import CellCoord
from app.domain.percept import Percept


@dataclass
class KnowledgeBase:
    cave_geo: CaveGeometry

    visited: set[CellCoord] = field(default_factory=set)
    no_pit: set[CellCoord] = field(default_factory=set)
    no_wumpus: set[CellCoord] = field(default_factory=set)
    pit: set[CellCoord] = field(default_factory=set)
    wumpus: CellCoord | None = None
    wumpus_dead: bool = False

    possible_pit: dict[CellCoord, set[CellCoord]] = field(default_factory=dict)
    possible_wumpus: dict[CellCoord, set[CellCoord]] = field(default_factory=dict)

    breeze_cells: set[CellCoord] = field(default_factory=set)
    stench_cells: set[CellCoord] = field(default_factory=set)
    no_breeze_cells: set[CellCoord] = field(default_factory=set)
    no_stench_cells: set[CellCoord] = field(default_factory=set)

    safe: set[CellCoord] = field(default_factory=set)

    def observe(self, coords: CellCoord, percept: Percept) -> None:
        self.visited.add(coords)
        self.no_pit.add(coords)
        self.no_wumpus.add(coords)

        neighbors = self.cave_geo.neighbor_coords(coords)

        if percept.scream:
            self.wumpus_dead = True
            self.wumpus = None
            self.possible_wumpus.clear()

        if percept.breeze:
            self.breeze_cells.add(coords)
            for nbh in neighbors:
                self._add_possible_pit_evidence(nbh, coords)
        else:
            self.no_breeze_cells.add(coords)
            for nbh in neighbors:
                self.no_pit.add(nbh)

        if not self.wumpus_dead:
            if percept.stench:
                self.stench_cells.add(coords)
                for nbh in neighbors:
                    self._add_possible_wumpus_evidence(nbh, coords)
            else:
                self.no_stench_cells.add(coords)
                for nbh in neighbors:
                    self.no_wumpus.add(nbh)

        self._normalize()
        self._deduce_singletons()
        self._normalize()
        self._recompute_safe()

    def is_safe(self, cell: CellCoord) -> bool:
        return cell in self.safe

    def is_known_pit(self, cell: CellCoord) -> bool:
        return cell in self.pit

    def is_known_wumpus(self, cell: CellCoord) -> bool:
        return (not self.wumpus_dead) and (self.wumpus == cell)

    def visited_neighbors(self, cell: CellCoord) -> set[CellCoord]:
        return {n for n in self.cave_geo.neighbor_coords(cell) if n in self.visited}

    def frontier_cells(self) -> set[CellCoord]:
        frontier: set[CellCoord] = set()
        for cell in self.visited:
            for nbh in self.cave_geo.neighbor_coords(cell):
                if nbh in self.visited:
                    continue
                if nbh in self.pit:
                    continue
                if self.wumpus is not None and nbh == self.wumpus and not self.wumpus_dead:
                    continue
                frontier.add(nbh)
        return frontier

    def candidate_summary(self, cell: CellCoord) -> dict[str, int | bool]:
        visited_neighbors = self.visited_neighbors(cell)
        return {
            "visited_neighbor_count": len(visited_neighbors),
            "breeze_neighbor_count": sum(1 for n in visited_neighbors if n in self.breeze_cells),
            "no_breeze_neighbor_count": sum(1 for n in visited_neighbors if n in self.no_breeze_cells),
            "stench_neighbor_count": sum(1 for n in visited_neighbors if n in self.stench_cells),
            "no_stench_neighbor_count": sum(1 for n in visited_neighbors if n in self.no_stench_cells),
            "safe_neighbor_count": sum(1 for n in self.cave_geo.neighbor_coords(cell) if n in self.safe),
            "possible_pit_flag": cell in self.possible_pit,
            "possible_wumpus_flag": cell in self.possible_wumpus,
            "possible_pit_support": len(self.possible_pit.get(cell, set())),
            "possible_wumpus_support": len(self.possible_wumpus.get(cell, set())),
            "known_safe_flag": cell in self.safe,
            "known_pit_flag": cell in self.pit,
            "known_wumpus_flag": (self.wumpus == cell) and (not self.wumpus_dead),
            "wumpus_dead_flag": self.wumpus_dead,
            "visited_total": len(self.visited),
            "frontier_flag": cell in self.frontier_cells(),
        }

    def _normalize(self) -> None:
        for cell in list(self.possible_pit.keys()):
            if cell in self.safe or cell in self.no_pit or cell in self.pit:
                self.possible_pit.pop(cell, None)

        for cell in list(self.possible_wumpus.keys()):
            if cell in self.safe or cell in self.no_wumpus:
                self.possible_wumpus.pop(cell, None)

        if self.wumpus is not None and self.wumpus in self.no_wumpus:
            self.wumpus = None

    def _add_possible_pit_evidence(self, candidate: CellCoord, from_cell: CellCoord) -> None:
        if candidate in self.no_pit or candidate in self.safe or candidate in self.pit:
            return
        self.possible_pit.setdefault(candidate, set()).add(from_cell)

    def _add_possible_wumpus_evidence(self, candidate: CellCoord, from_cell: CellCoord) -> None:
        if self.wumpus_dead or candidate in self.no_wumpus or candidate in self.safe:
            return
        self.possible_wumpus.setdefault(candidate, set()).add(from_cell)

    def _recompute_safe(self) -> None:
        if self.wumpus_dead:
            base = set(self.no_pit)
        else:
            base = self.no_pit.intersection(self.no_wumpus)

        hazards = set(self.pit)
        if self.wumpus is not None and not self.wumpus_dead:
            hazards.add(self.wumpus)

        self.safe = base.difference(hazards)

    def _deduce_singletons(self) -> None:
        for v in list(self.breeze_cells):
            candidates = [
                n for n in self.cave_geo.neighbor_coords(v)
                if (n not in self.no_pit) and (n not in self.safe) and (n not in self.pit)
            ]
            if len(candidates) == 1:
                p = candidates.pop()
                self.pit.add(p)
                self.possible_pit.pop(p, None)

        if self.wumpus_dead:
            return

        if self.wumpus is None:
            for v in list(self.stench_cells):
                candidates = [
                    n for n in self.cave_geo.neighbor_coords(v)
                    if (n not in self.no_wumpus) and (n not in self.safe)
                ]
                if len(candidates) == 1:
                    p = candidates.pop()
                    self.wumpus = p
                    self.possible_wumpus.pop(p, None)
                    break
