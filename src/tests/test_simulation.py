from math import inf
from random import random

import pytest

from src.bandit import BanditCollection, BernoulliBandit
from src.simulation import (EpsilonFirstStrategy,
                            EpsilonGreegyStrategy,
                            EpsilonDecreasingStrategy)


@pytest.fixture(scope="module")
def bandits():
    return [BernoulliBandit(random()) for _ in range(10)]


@pytest.fixture(scope="module")
def bandit_collection(bandits: list[BernoulliBandit]):
    return BanditCollection(bandits)


@pytest.fixture(scope="module")
def epsilon_strategy(bandit_collection: BanditCollection) -> EpsilonSimulation:
    return EpsilonFirstStrategy(
        bandit_collection=bandit_collection,
        num_simulations=50,
        random_bound=0.2,
    )


def test_gen_random_value(epsilon_strategy: EpsilonFirstStrategy):
    uniform_value = epsilon_strategy.gen_random_value()

    assert 0 <= uniform_value <= 1


def test_bandit_strategy_low(epsilon_strategy: EpsilonFirstStrategy):
    random_value = 0.1

    bandit = epsilon_strategy._bandit_strategy(random_value)
    assert isinstance(bandit, BernoulliBandit)


def test_bandit_strategy_high(epsilon_strategy: EpsilonFirstStrategy):
    random_value = 0.5

    bandit = epsilon_strategy._bandit_strategy(random_value)
    assert bandit.parameter_hat == inf


def test_simulate_one(epsilon_strategy: EpsilonFirstStrategy):
    epsilon_strategy.simulate_one()

    assert epsilon_strategy.simulation_num == 1
    assert min(epsilon_strategy.bandit_collection).parameter_hat in {0, 1}
    assert min(epsilon_strategy.bandit_collection).num_simulations == 1


def test_simulation(epsilon_strategy: EpsilonFirstStrategy):
    epsilon_strategy.simulation()
    expected_simulations = epsilon_strategy.num_simulations
    bandit_sims = [
        bandit.num_simulations
        for bandit in epsilon_strategy.bandit_collection.bandits
    ]

    assert epsilon_strategy.simulation_num == expected_simulations
    assert sum(bandit_sims) == expected_simulations

    # We expect that each bandit is pulled atleast once
    assert min(bandit_sims) > 0
