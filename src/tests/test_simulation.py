import numpy as np
import pytest

from random import random

from src.bandit import Bandit
from src.simulation import Simulation


@pytest.fixture
def bandits():
    return [Bandit(random()) for _ in range(10)]


@pytest.fixture
def simulation(bandits: list[Bandit]):
    return Simulation(bandits)


def test_optimal_bandit(simulation: Simulation):
    assert simulation.optimal_bandit.parameter_hat == np.inf
