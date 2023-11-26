import pytest

from math import inf

from src.bandit import BernoulliBandit, BanditCollection
from src.simulation import EpsilonSimulation
from random import random


@pytest.fixture(scope="module")
def bandits():
    return [BernoulliBandit(random()) for _ in range(10)]


@pytest.fixture(scope="module")
def bandit_collection(bandits: list[BernoulliBandit]):
    return BanditCollection(bandits)


@pytest.fixture(scope="module")
def epsilon_simulation(bandit_collection: BanditCollection) -> EpsilonSimulation:
    return EpsilonSimulation(
        bandit_collection=bandit_collection, num_simulations=50, random_bound=0.2
    )


def test_gen_random_value(epsilon_simulation: EpsilonSimulation):
    uniform_value = epsilon_simulation.gen_random_value()

    assert 0 <= uniform_value <= 1


def test_bandit_strategy_low(epsilon_simulation: EpsilonSimulation):
    random_value = 0.1

    bandit = epsilon_simulation._bandit_strategy(random_value)
    assert isinstance(bandit, BernoulliBandit)


def test_bandit_strategy_high(epsilon_simulation: EpsilonSimulation):
    random_value = 0.5

    bandit = epsilon_simulation._bandit_strategy(random_value)
    assert bandit.parameter_hat == inf


def test_simulate_one(epsilon_simulation: EpsilonSimulation):
    epsilon_simulation.simulate_one()

    assert epsilon_simulation.simulation_num == 1
    assert min(epsilon_simulation.bandit_collection).parameter_hat in {0, 1}
    assert min(epsilon_simulation.bandit_collection).num_simulations == 1


def test_simulation(epsilon_simulation: EpsilonSimulation):
    epsilon_simulation.simulation()
    expected_simulations = epsilon_simulation.num_simulations
    bandit_sims = [
        bandit.num_simulations
        for bandit in epsilon_simulation.bandit_collection.bandits
    ]

    assert epsilon_simulation.simulation_num == expected_simulations
    assert sum(bandit_sims) == expected_simulations

    # We expect that each bandit is pulled atleast once
    assert min(bandit_sims) > 0
