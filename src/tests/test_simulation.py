import numpy as np
import pytest

from random import random

from src.bandit import Bandit
from src.simulation import BanditCollection


@pytest.fixture
def bandits():
    return [Bandit(random()) for _ in range(10)]


@pytest.fixture
def bandit_collection(bandits: list[Bandit]):
    return BanditCollection(bandits)


def test_optimal_bandit(bandit_collection: BanditCollection):
    assert bandit_collection.optimal_bandit.parameter_hat == np.inf
