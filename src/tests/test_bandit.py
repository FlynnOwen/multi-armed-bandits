from math import inf
from random import random

import pytest

from src.bandit import BanditCollection, BernoulliBandit


@pytest.fixture()
def mock_parameter_hat():
    return 0.5


@pytest.fixture()
def true_parameter():
    return 0.5


@pytest.fixture()
def bandits():
    return [BernoulliBandit(random()) for _ in range(10)]


@pytest.fixture()
def bandit_collection(bandits: list[BernoulliBandit]):
    return BanditCollection(bandits)


def test_optimal_bandit(bandit_collection: BanditCollection):
    assert bandit_collection.optimal_bandit.parameter_hat == inf


@pytest.fixture()
def bandit(true_parameter, monkeypatch, mock_parameter_hat):
    bandit = BernoulliBandit(true_parameter)
    monkeypatch.setattr(BernoulliBandit, "parameter_hat", mock_parameter_hat)

    return bandit


def test_eq(bandit, mock_parameter_hat):
    assert bandit == mock_parameter_hat


def test_le(bandit, mock_parameter_hat):
    assert bandit <= mock_parameter_hat
    assert bandit < mock_parameter_hat + 0.1


def test_lt(bandit, mock_parameter_hat):
    assert bandit < mock_parameter_hat + 0.1


def test_ge(bandit, mock_parameter_hat):
    assert bandit >= mock_parameter_hat
    assert bandit >= mock_parameter_hat - 0.1


def test_gt(bandit, mock_parameter_hat):
    assert bandit > mock_parameter_hat - 0.1


def test_generate(bandit):
    bandit.generate()

    assert bandit._results[0] in {0, 1}


def test_reward(bandit):
    assert bandit.reward in {0, 1}
