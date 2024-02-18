from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import total_ordering
from math import inf
from random import choice

from numpy.random import binomial


@total_ordering
class Bandit(ABC):
    """
    Base class for a bandit that is from the single-parameter
    family of exponential distributions.

    Examples may include Bernoulli or Poisson.
    """
    num_parameters = 1

    def __init__(self,
                 parameter: float) -> None:
        self.parameter = parameter
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
        The estimated primary parameter.
        """

    def _residual(self,
                  parameter: float,
                  parameter_hat: float) -> float:
        """
        Calculate difference between a true
        parameter and an estimated parameter.
        """
        return parameter - parameter_hat

    @property
    def residual(self) -> float:
        """
        Residual between the true primary parameter
        and the estimated primary parameter.
        """
        return self._residual(self.parameter, self.parameter_hat)

    @property
    def reward(self) -> float:
        return sum(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __eq__(self, value: float) -> bool:
        return self.parameter_hat == value

    def __lt__(self, value: float) -> bool:
        return self.parameter_hat < value


class TwoParameterBandit(Bandit):
    """
    An extension for bandits that are from the two-parameter
    family of exponential distributions.

    Examples may include Gaussian.
    """
    num_parameters = 2

    def __init__(self,
                 parameter: float,
                 secondary_parameter: float) -> None:
        super().__init__(parameter)
        self.secondary_parameter = secondary_parameter

    @property
    @abstractmethod
    def secondary_parameter_hat(self) -> float:
        """
        The estimated secondary parameter.
        """

    @property
    def secondary_residual(self) -> float:
        """
        Residual between the true secondary parameter
        and the estimated seconday parameter.
        """
        return self._residual(self.secondary_parameter, self.secondary_parameter_hat)



class BernoulliBandit(Bandit):
    """
    Single bandit, simulating over a Bernoulli distribution.
    """

    def __init__(self,
                 parameter: float) -> None:
        if 0 > parameter < 1:
            raise ValueError(
                f"parameter must be 0 <= parameter <= 1"
                f"got value {parameter} instead."
            )
        super().__init__(parameter)

    def generate(self) -> float:
        """
        Generates a single result from the pull of the
        armed bandit.
        """
        result = binomial(1, p=self.parameter)
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
        return self.parameter - self.parameter_hat


@dataclass
class BanditCollection:
    """
    Container class for a collection of armed bandits.
    """

    bandits: list[Bandit]

    def __post_init__(self):
        self.num_parameters = self.random_bandit.num_parameters

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
