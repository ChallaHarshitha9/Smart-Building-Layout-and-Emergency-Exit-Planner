import pytest

from tests.factories import make_layout, make_navgraph


@pytest.fixture
def layout():
    return make_layout()


@pytest.fixture
def navgraph():
    return make_navgraph()
