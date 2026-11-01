import pytest

from app.agent.KB import KnowledgeBase
from app.domain.cave import CaveGeometry
from app.domain.cell import CellCoord
from app.evaluation.exact_inference import exact_pit_probabilities


def test_exact_pit_posterior_for_vault_state():
    kb = KnowledgeBase(CaveGeometry(width=3, height=3))
    kb.visited = {CellCoord(0, 0), CellCoord(1, 0), CellCoord(0, 1)}
    kb.no_pit = set(kb.visited)
    kb.safe = set(kb.visited)
    kb.breeze_cells = {CellCoord(1, 0), CellCoord(0, 1)}

    probabilities, total = exact_pit_probabilities(kb, num_pits=3)

    assert total == 13
    assert probabilities[CellCoord(1, 1)] == pytest.approx(10 / 13)
