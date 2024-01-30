from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import total_ordering
from math import inf
from random import choice, randint
from typing import Any


@dataclass
class Bandit(ABC):
    """
    Base class for a Bandit.

    Note that bandits may follow different distributions (e.g) Bernoulli.
    """

    true_parameter: Any
    _results: list[Any] = field(default_factory=list, init=False)

    @abstractmethod
    def generate(self):
        """
        'Pull' the armed bandit.
        The result is a value sampled from the distribution
        that the armed bandit follows.

        The result is then cached.
        """

    @property
    @abstractmethod
    def parameter_hat(self):
        """
        The estimated parameter(s).
        """

    @property
    @abstractmethod
    def residual(self):
        """
        Residual between the true parameter(s)
        and the estimated parameter(s).
        """

    @property
    @abstractmethod
    def reward(self):
        """
        The total utility gained from this bandit.
        """

    def __len__(self):
        return len(self._results)

    @abstractmethod
    def __eq__(self, value: float):
        pass

    @abstractmethod
    def __lt__(self, value: float):
        pass


@total_ordering
@dataclass
class BernoulliBandit(Bandit):
    """Single bandit, simulating over a Bernoulli distribution."""

    true_parameter: float
    _results: list[int] = field(default_factory=list, init=False)

    def generate(self):
        """
        Generates a single result from the pull of the
        armed bandit.
        """
        result = randint(0, 1)
        self._results.append(result)

    @property
    def parameter_hat(self):
        """
        The estimated parameter for this bandit.
        Note that the Maximum Likelihood estimator is used.
        """
        if not self._results:
            return inf
        else:
            return sum(self._results) / len(self._results)

    @property
    def residual(self):
        """
        The difference between the true parameter of an armed
        bandit and the estimated parameter.
        """
        return self.true_parameter - self.parameter_hat

    @property
    def reward(self):
        return sum(self._results)

    @property
    def num_simulations(self):
        """
        Number of simulations that this armed bandit has performed.
        """
        return len(self._results)

    def __eq__(self, value: float):
        return self.parameter_hat == value

    def __lt__(self, value: float):
        return self.parameter_hat < value


@dataclass
class BanditCollection:
    """
    Container class for a collection of armed bandits.
    """

    bandits: list[Bandit]

    def __iter__(self):
        return iter(self.bandits)

    def __len__(self):
        return len(self.bandits)

    @property
    def optimal_bandit(self) -> Bandit:
        return self.bandits[self.bandits.index(max(self.bandits))]

    @property
    def random_bandit(self) -> Bandit:
        return choice(self.bandits)

    def add_bandit(self, bandit: Bandit):
        self.bandits.append(bandit)
