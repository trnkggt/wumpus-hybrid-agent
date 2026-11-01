import random

import pytest


@pytest.fixture(autouse=True)
def preserve_random_state():
    """Keep tests from leaking global random state into each other."""
    state = random.getstate()
    yield
    random.setstate(state)
