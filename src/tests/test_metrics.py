from random import random

import pytest

from src.bandit import BanditCollection, BernoulliBandit
from src.metrics import OneParameterMetrics
from src.simulation import EpsilonGreedyStrategy


@pytest.fixture(scope="module")
def bandits():
    return [BernoulliBandit(random()) for _ in range(10)]


@pytest.fixture(scope="module")
def bandit_collection(bandits: list[BernoulliBandit]):
    return BanditCollection(bandits)


@pytest.fixture(scope="module")
def metrics(bandit_collection: BanditCollection):
    simulation = EpsilonGreedyStrategy(
        bandit_collection=bandit_collection,
        num_simulations=50,
        epsilon=0.2,
    )
    simulation.full_simulation()
    return OneParameterMetrics(simulation)


def test_metrics_num_simulations(metrics: OneParameterMetrics):
    assert metrics.num_simulations == 50


def test_metrics_errors(metrics: OneParameterMetrics):
    assert 0 <= metrics.mae <= 1
    assert metrics.mape >= 0


def test_metrics_rewards(metrics: OneParameterMetrics):
    assert 0 <= metrics.total_reward <= 50
