import pytest

from paydayradar import synthetic
from paydayradar.pipeline import Workspace


@pytest.fixture(scope="session")
def data():
    return synthetic.generate(seed=42)


@pytest.fixture(scope="session")
def ws(data):
    inv, exp = data
    return Workspace(inv, exp, as_of=synthetic.DEFAULT_AS_OF,
                     opening_balance=synthetic.DEFAULT_OPENING_BALANCE, backend="sklearn")
