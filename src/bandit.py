from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import total_ordering
from math import inf
from random import choice
from typing import Any

from numpy.random import binomial


class Bandit(ABC):
    """
    Base class for a Bandit.

    Note that bandits may follow different distributions (e.g) Bernoulli.
    """

    def __init__(self,
                 true_parameter: float) -> None:
        self.true_parameter = true_parameter
        self._results: list[float] = []

    @abstractmethod
    def generate(self) -> None:
        """
        'Pull' the armed bandit.
        The result is a value sampled from the distribution
        that the armed bandit follows.

        The result is then cached.
        """

    @property
    @abstractmethod
    def parameter_hat(self) -> float:
        """
        The estimated parameter(s).
        """

    @property
    @abstractmethod
    def residual(self) -> float:
        """
        Residual between the true parameter(s)
        and the estimated parameter(s).
        """

    @property
    def reward(self) -> float:
        return sum(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __eq__(self, value: float) -> bool:
        return self.parameter_hat == value

    def __lt__(self, value: float) -> bool:
        return self.parameter_hat < value


@total_ordering
class BernoulliBandit(Bandit):
    """Single bandit, simulating over a Bernoulli distribution."""

    def __init__(self,
                 true_parameter: float) -> None:
        if 0 > true_parameter < 1:
            raise ValueError(
                f"true_parameter must be 0 <= true_parameter <= 1"
                f"got value {true_parameter} instead."
            )
        super().__init__(true_parameter)

    def generate(self) -> float:
        """
        Generates a single result from the pull of the
        armed bandit.
        """
        result = binomial(1, p=self.true_parameter)
        self._results.append(result)

        return result

    @property
    def parameter_hat(self) -> float:
        """
        The estimated parameter for this bandit.
        Note that the Maximum Likelihood estimator is used.
        """
        if not self._results:
            return inf
        return sum(self._results) / len(self._results)

    @property
    def residual(self) -> float:
        """
        The difference between the true parameter of an armed
        bandit and the estimated parameter.
        """
        return self.true_parameter - self.parameter_hat


@dataclass
class BanditCollection:
    """
    Container class for a collection of armed bandits.
    """

    bandits: list[Bandit]

    def __iter__(self):  # noqa: ANN204
        return iter(self.bandits)

    def __len__(self) -> int:
        return len(self.bandits)

    @property
    def simulation_num(self) -> int:
        return sum([len(bandit) for bandit in self.bandits])

    @property
    def optimal_bandit(self) -> Bandit:
        return self.bandits[self.bandits.index(max(self.bandits))]

    @property
    def random_bandit(self) -> Bandit:
        return choice(self.bandits)

    def add_bandit(self, bandit: Bandit) -> None:
        self.bandits.append(bandit)
