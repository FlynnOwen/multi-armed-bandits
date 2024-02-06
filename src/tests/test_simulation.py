from math import inf
from random import random

import pytest

from src.bandit import BanditCollection, BernoulliBandit
from src.simulation import (
    EpsilonGreedyStrategy,
)


@pytest.fixture()
def bandits():
    return [BernoulliBandit(random()) for _ in range(10)]


@pytest.fixture()
def bandit_collection(bandits: list[BernoulliBandit]):
    return BanditCollection(bandits)


@pytest.fixture()
def epsilon_greedy_strategy(bandit_collection) -> EpsilonGreedyStrategy:
    return EpsilonGreedyStrategy(
        bandit_collection=bandit_collection,
        num_simulations=50,
        epsilon=0.2,
    )


def test_simulation_num(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    assert epsilon_greedy_strategy.simulation_num == 0


def test_gen_random_uniform(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    uniform_value = epsilon_greedy_strategy.gen_random_uniform()

    assert 0 <= uniform_value <= 1


def test_bandit_strategy_low(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    random_value = 0.1

    bandit = epsilon_greedy_strategy._bandit_strategy(random_value)
    assert isinstance(bandit, BernoulliBandit)


def test_bandit_strategy_high(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    random_value = 0.5

    bandit = epsilon_greedy_strategy._bandit_strategy(random_value)
    assert bandit.parameter_hat == inf


def test_simulate_one(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    epsilon_greedy_strategy.simulate_one()

    assert epsilon_greedy_strategy.simulation_num == 1
    assert min(epsilon_greedy_strategy.bandit_collection).parameter_hat in {0, 1}
    assert len(min(epsilon_greedy_strategy.bandit_collection)) == 1


def test_full_simulation_counts(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    epsilon_greedy_strategy.full_simulation()
    expected_simulations = epsilon_greedy_strategy.num_simulations

    assert epsilon_greedy_strategy.simulation_num == expected_simulations

    # We expect that each bandit is pulled atleast once
    assert len(min(epsilon_greedy_strategy.bandit_collection)) > 0

    # Single simulation gives expected counts
    epsilon_greedy_strategy.simulate_one()
    assert epsilon_greedy_strategy.simulation_num == expected_simulations + 1


def test_full_simulation_counts_cutoff(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    expected_simulations = epsilon_greedy_strategy.num_simulations

    epsilon_greedy_strategy.simulate_one()
    epsilon_greedy_strategy.full_simulation()

    assert epsilon_greedy_strategy.simulation_num == expected_simulations


def test_simulate_counts(epsilon_greedy_strategy: EpsilonGreedyStrategy):
    nums_sims = 100
    epsilon_greedy_strategy.simulate(num_sims=nums_sims)

    assert epsilon_greedy_strategy.simulation_num == nums_sims
