from app.agent.KB import KnowledgeBase
from app.domain.cave import CaveGeometry
from app.domain.cell import CellCoord
from app.domain.percept import Percept


def test_no_breeze_no_stench_marks_neighbors_safe():
    kb = KnowledgeBase(CaveGeometry(width=2, height=2))

    kb.observe(CellCoord(0, 0), Percept())

    assert CellCoord(0, 0) in kb.safe
    assert CellCoord(1, 0) in kb.safe
    assert CellCoord(0, 1) in kb.safe
    assert CellCoord(1, 1) not in kb.safe


def test_breeze_adds_possible_pit_evidence():
    kb = KnowledgeBase(CaveGeometry(width=2, height=2))

    kb.observe(CellCoord(0, 0), Percept(breeze=True))

    assert CellCoord(1, 0) in kb.possible_pit
    assert CellCoord(0, 1) in kb.possible_pit


def test_singleton_pit_is_deduced():
    kb = KnowledgeBase(CaveGeometry(width=2, height=2))
    kb.observe(CellCoord(0, 0), Percept())  # (1, 0) and (0, 1) are no_pit
    kb.observe(CellCoord(1, 0), Percept(breeze=True))

    assert CellCoord(1, 1) in kb.pit
    assert CellCoord(1, 1) not in kb.safe


def test_no_stench_marks_neighbors_no_wumpus():
    kb = KnowledgeBase(CaveGeometry(width=2, height=2))

    kb.observe(CellCoord(0, 0), Percept())

    assert CellCoord(1, 0) in kb.no_wumpus
    assert CellCoord(0, 1) in kb.no_wumpus


def test_singleton_wumpus_is_deduced():
    kb = KnowledgeBase(CaveGeometry(width=2, height=2))
    kb.observe(CellCoord(0, 0), Percept(stench=True))
    kb.observe(CellCoord(1, 0), Percept())

    assert kb.wumpus == CellCoord(0, 1)


def test_scream_clears_wumpus_and_makes_its_cell_safe():
    kb = KnowledgeBase(CaveGeometry(width=2, height=2))
    kb.observe(CellCoord(0, 0), Percept(stench=True))
    kb.observe(CellCoord(1, 0), Percept())
    assert kb.wumpus == CellCoord(0, 1)

    kb.observe(CellCoord(0, 0), Percept(scream=True))

    assert kb.wumpus_dead is True
    assert kb.wumpus is None
    assert kb.possible_wumpus == {}
    assert CellCoord(0, 1) in kb.safe
