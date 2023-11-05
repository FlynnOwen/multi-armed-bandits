import pytest

from src.bandit import Bandit


@pytest.fixture
def true_parameter():
    return 0.5

@pytest.fixture
def bandit(true_parameter):
    return Bandit(true_parameter)


def test_eq(bandit, true_parameter):
    assert bandit == true_parameter


def test_le(bandit, true_parameter):
    assert bandit <= true_parameter and bandit < true_parameter + 0.1


def test_lt(bandit, true_parameter):
    assert bandit < true_parameter + 0.1


def test_ge(bandit, true_parameter):
    assert bandit >= true_parameter and bandit >= true_parameter - 0.1


def test_gt(bandit, true_parameter):
    assert bandit > true_parameter - 0.1
