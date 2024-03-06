from math import inf
from random import random

import pytest

from src.bandit import (
    BanditCollection, 
    BernoulliBandit,
    PoissonBandit,
    GaussianBandit,
    distribution_factory,
    OneParamDistribution,
    TwoParamDistribution
)


@pytest.fixture()
def mock_parameter_hat_bernoulli():
    return 0.5


@pytest.fixture()
def parameter_bernoulli():
    return 0.5


@pytest.fixture()
def bandits_bernoulli():
    return [BernoulliBandit(random()) for _ in range(10)]


@pytest.fixture()
def bandit_collection_bernoulli(bandits_bernoulli: list[BernoulliBandit]):
    return BanditCollection(bandits_bernoulli)


@pytest.fixture()
def mock_parameter_hat_poisson():
    return 5


@pytest.fixture()
def parameter_poisson():
    return 5


@pytest.fixture()
def bandits_poisson(parameter_poisson):
    return [PoissonBandit(parameter_poisson) for _ in range(10)]


@pytest.fixture()
def bandit_collection_poisson(bandits_poisson: list[PoissonBandit]):
    return BanditCollection(bandits_poisson)


@pytest.fixture()
def mock_parameter_hat_gaussian():
    return 100


@pytest.fixture()
def parameter_gaussian():
    return 100

@pytest.fixture()
def mock_secondary_parameter_hat_gaussian():
    return 1


@pytest.fixture()
def parameter_secondary_gaussian():
    return 1


@pytest.fixture()
def bandits_gaussian(parameter_gaussian, parameter_secondary_gaussian):
    return [GaussianBandit(parameter_gaussian,
                           parameter_secondary_gaussian)
                           for _ in range(10)]


@pytest.fixture()
def bandit_collection_gaussian(bandits_gaussian: list[GaussianBandit]):
    return BanditCollection(bandits_gaussian)

@pytest.mark.parametrize("bandit_collection, expected_parameter_hat",  # noqa
                          [
                              ("bandit_collection_bernoulli", inf),
                              ("bandit_collection_poisson", inf),
                              ("bandit_collection_gaussian", inf)
                              ])
def test_optimal_bandit(request, bandit_collection, expected_parameter_hat):
    bandit_collection = request.getfixturevalue(bandit_collection)
    assert bandit_collection.optimal_bandit.parameter_hat == expected_parameter_hat


@pytest.fixture()
def bandit_bernoulli(parameter_bernoulli, monkeypatch, mock_parameter_hat_bernoulli):
    bandit = BernoulliBandit(parameter_bernoulli)
    monkeypatch.setattr(BernoulliBandit, "parameter_hat", mock_parameter_hat_bernoulli)

    return bandit

@pytest.fixture()
def bandit_poisson(parameter_poisson, monkeypatch, mock_parameter_hat_poisson):
    bandit = PoissonBandit(parameter_poisson)
    monkeypatch.setattr(PoissonBandit, "parameter_hat", mock_parameter_hat_poisson)

    return bandit

@pytest.fixture()
def bandit_gaussian(parameter_gaussian,
                    parameter_secondary_gaussian,
                    monkeypatch,
                    mock_parameter_hat_gaussian,
                    mock_secondary_parameter_hat_gaussian):
    bandit = GaussianBandit(parameter_gaussian, parameter_secondary_gaussian)
    monkeypatch.setattr(GaussianBandit, "parameter_hat", mock_parameter_hat_gaussian)
    monkeypatch.setattr(GaussianBandit, "secondary_parameter_hat",
                        mock_secondary_parameter_hat_gaussian)

    return bandit


@pytest.mark.parametrize("bandit, expected_value",  # noqa
                          [
                              ("bandit_bernoulli", "mock_parameter_hat_bernoulli"),
                              ("bandit_poisson", "mock_parameter_hat_poisson"),
                              ("bandit_gaussian", "mock_parameter_hat_gaussian")
                              ])
def test_eq(request, bandit, expected_value):
    bandit = request.getfixturevalue(bandit)
    expected_value = request.getfixturevalue(expected_value)
    assert bandit == expected_value


@pytest.mark.parametrize("bandit, expected_value",  # noqa
                          [
                              ("bandit_bernoulli", "mock_parameter_hat_bernoulli"),
                              ("bandit_poisson", "mock_parameter_hat_poisson"),
                              ("bandit_gaussian", "mock_parameter_hat_gaussian")
                              ])
def test_le(request, bandit, expected_value):
    bandit = request.getfixturevalue(bandit)
    expected_value = request.getfixturevalue(expected_value)
    assert bandit <= expected_value
    assert bandit < expected_value + 0.1


@pytest.mark.parametrize("bandit, expected_value",  # noqa
                          [
                              ("bandit_bernoulli", "mock_parameter_hat_bernoulli"),
                              ("bandit_poisson", "mock_parameter_hat_poisson"),
                              ("bandit_gaussian", "mock_parameter_hat_gaussian")
                              ])
def test_lt(request, bandit, expected_value):
    bandit = request.getfixturevalue(bandit)
    expected_value = request.getfixturevalue(expected_value)
    assert bandit < expected_value + 0.1


@pytest.mark.parametrize("bandit, expected_value",  # noqa
                          [
                              ("bandit_bernoulli", "mock_parameter_hat_bernoulli"),
                              ("bandit_poisson", "mock_parameter_hat_poisson"),
                              ("bandit_gaussian", "mock_parameter_hat_gaussian")
                              ])
def test_ge(request, bandit, expected_value):
    bandit = request.getfixturevalue(bandit)
    expected_value = request.getfixturevalue(expected_value)
    assert bandit >= expected_value
    assert bandit >= expected_value - 0.1


@pytest.mark.parametrize("bandit, expected_value",  # noqa
                          [
                              ("bandit_bernoulli", "mock_parameter_hat_bernoulli"),
                              ("bandit_poisson", "mock_parameter_hat_poisson"),
                              ("bandit_gaussian", "mock_parameter_hat_gaussian")
                              ])
def test_gt(request, bandit, expected_value):
    bandit = request.getfixturevalue(bandit)
    expected_value = request.getfixturevalue(expected_value)
    assert bandit > expected_value - 0.1


def test_generate_bernoulli(bandit_bernoulli):
    bandit_bernoulli.generate()

    assert bandit_bernoulli._results[0] in {0, 1}


def test_generate_poisson(bandit_poisson):
    bandit_poisson.generate()
    result = bandit_poisson._results[0]

    assert isinstance(result, int)
    assert result >= 0


def test_generate_gaussian(bandit_gaussian):
    bandit_gaussian.generate()
    result = bandit_gaussian._results[0]

    assert isinstance(result, float)


@pytest.mark.parametrize("bandit",  # noqa
                          ["bandit_bernoulli",
                           "bandit_poisson",
                            "bandit_gaussian"
                            ])
def test_reward(request, bandit):
    bandit = request.getfixturevalue(bandit)
    result = bandit.generate()

    assert bandit.reward == result == bandit._results[0]


@pytest.mark.parametrize("distribution, expected_bandit",  # noqa
                          [
                              (OneParamDistribution.bernoulli, BernoulliBandit),
                              (OneParamDistribution.poisson, PoissonBandit),
                              (TwoParamDistribution.gaussian, GaussianBandit)
                              ])
def test_distribution_factory(distribution, expected_bandit):
    bandit = distribution_factory(distribution=distribution)

    assert bandit == expected_bandit
