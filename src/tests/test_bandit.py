import pytest

from math import inf
from random import random

from src.bandit import Bandit, BanditCollection


@pytest.fixture
def mock_parameter_hat():
    return 0.5


@pytest.fixture
def true_parameter():
    return 0.5

@pytest.fixture
def bandits():
    return [Bandit(random()) for _ in range(10)]


@pytest.fixture
def bandit_collection(bandits: list[Bandit]):
    return BanditCollection(bandits)


def test_optimal_bandit(bandit_collection: BanditCollection):
    assert bandit_collection.optimal_bandit.parameter_hat == inf



@pytest.fixture
def bandit(true_parameter, monkeypatch, mock_parameter_hat):
    bandit = Bandit(true_parameter)
    monkeypatch.setattr(Bandit, "parameter_hat", mock_parameter_hat)

    return bandit


def test_eq(bandit, mock_parameter_hat):
    assert bandit == mock_parameter_hat


def test_le(bandit, mock_parameter_hat):
    assert bandit <= mock_parameter_hat and bandit < mock_parameter_hat + 0.1


def test_lt(bandit, mock_parameter_hat):
    assert bandit < mock_parameter_hat + 0.1


def test_ge(bandit, mock_parameter_hat):
    assert bandit >= mock_parameter_hat and bandit >= mock_parameter_hat - 0.1


def test_gt(bandit, mock_parameter_hat):
    assert bandit > mock_parameter_hat - 0.1


def test_generate(bandit):
    result = bandit.generate()

    assert result in {0,1}
    assert bandit._results == [result]
