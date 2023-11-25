import pytest

from math import inf

from src.bandit import Bandit, BanditCollection
from src.simulation import EpsilonSimulation
from random import random


@pytest.fixture(scope="module")
def bandits():
    return [Bandit(random()) for _ in range(10)]


@pytest.fixture(scope="module")
def bandit_collection(bandits: list[Bandit]):
    return BanditCollection(bandits)


@pytest.fixture(scope="module")
def epsilon_simulation(bandit_collection: BanditCollection) -> EpsilonSimulation:
    return EpsilonSimulation(bandit_collection=bandit_collection,
                             num_simulations=50,
                             random_bound=0.2)


def test_gen_random_value(epsilon_simulation: EpsilonSimulation):
    uniform_value = epsilon_simulation.gen_random_value()

    assert 0 <= uniform_value <= 1


def test_bandit_strategy_low(epsilon_simulation: EpsilonSimulation):
    random_value = 0.1

    bandit = epsilon_simulation._bandit_strategy(random_value)
    assert isinstance(bandit, Bandit)


def test_bandit_strategy_high(epsilon_simulation: EpsilonSimulation):
    random_value = 0.5

    bandit = epsilon_simulation._bandit_strategy(random_value)
    assert bandit.parameter_hat == inf
