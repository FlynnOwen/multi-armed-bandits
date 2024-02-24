from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from functools import total_ordering
from math import inf, sqrt
from random import choice

from numpy.random import binomial, normal


class Distribution(StrEnum):
    bernoulli = "bernoulli"
    gaussian = "gaussian"

    @classmethod
    @property
    def one_parameter_family(cls):  # noqa: ANN206
        return {cls.bernoulli}

    @classmethod
    @property
    def two_parameter_family(cls):  # noqa: ANN206
        return {cls.gaussian}


def distribution_factory(distribution: Distribution) -> Bandit:  # noqa: ANN003
    distribution_map = {
        Distribution.bernoulli: BernoulliBandit,
        Distribution.gaussian: GaussianBandit,
    }

    return distribution_map[distribution]


@total_ordering
class Bandit(ABC):
    """
    Base class for a bandit that is from the single-parameter
    family of exponential distributions.

    Examples may include Bernoulli or Poisson.
    """

    num_parameters = 1

    def __init__(self, parameter: float, **kwargs) -> None:
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

    def _residual(self, parameter: float, parameter_hat: float) -> float:
        """
        Calculate difference between a true
        parameter and an estimated parameter.
        """
        return parameter - parameter_hat

    @property
    def residual(self) -> float:
        """
        The difference between the true parameter of an armed
        bandit and the estimated parameter.
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


class TwoParameterBandit(Bandit, ABC):
    """
    An extension for bandits that are from the two-parameter
    family of exponential distributions.

    Examples may include Gaussian.
    """

    num_parameters = 2

    def __init__(self, parameter: float, secondary_parameter: float, **kwargs) -> None:
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

    def __init__(self, parameter: float, **kwargs) -> None:
        if parameter < 0 or parameter > 1:
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
        The estimated theta for this bandit.
        """
        if not self._results:
            return inf
        return sum(self._results) / len(self._results)


class GaussianBandit(TwoParameterBandit):
    """
    Single bandit, simulating over a Gaussian distribution.
    """

    def __init__(self, parameter: float, secondary_parameter: float, **kwargs) -> None:
        if secondary_parameter < 0:
            raise ValueError(
                "secondary_parameter must be < 0"
                f"got value {secondary_parameter} instead."
            )
        super().__init__(parameter, secondary_parameter)

    def generate(self) -> float:
        """
        Generates a single result from the pull of the
        armed bandit.
        """
        result = normal(loc=self.parameter, scale=self.secondary_parameter)
        self._results.append(result)

        return result

    @property
    def parameter_hat(self) -> float:
        """
        The estimated mean for this bandit.
        """
        if not self._results:
            return inf
        return sum(self._results) / len(self._results)

    @property
    def secondary_parameter_hat(self) -> float:
        """
        The estimated variance for this bandit.
        """
        return sqrt(sum([(result - self.parameter_hat)**2 for result in self._results]) / len(
            self._results
        ))


@dataclass
class BanditCollection:
    """
    Container class for a collection of armed bandits.
    """

    bandits: list[Bandit]
    results: list[int] = field(default_factory=list)
    num_parameters: int = 1

    @classmethod
    def from_parameter_list( # noqa
        cls,
        distribution: Distribution,
        parameter_one_values: list[float]
    ):
        """
        Constructor using list(s) of parameters and a bandit type.
        """
        bandit_type = distribution_factory(distribution=distribution)
        bandits = [bandit_type(parameter) for parameter in parameter_one_values]
        return cls(bandits=bandits)

    @classmethod
    def from_parameter_distribution( #noqa
        cls,
        distribution: Distribution,
        num_bandits: int,
        parameter_one_mean: float,
        parameter_one_std: float
    ):
        """
        Constructor using and a bandit type,
        the number of bandits to generate, and the
        mean and std of each parameter.
        """
        parameters = normal(loc=parameter_one_mean,
                            scale=parameter_one_std,
                            size=num_bandits).tolist()

        return cls.from_parameter_list(distribution=distribution,
                                       parameter_one_values=parameters)

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

    @property
    def true_parameters(self) -> list[float]:
        return [bandit.parameter for bandit in self]

    @property
    def estimated_parameters(self) -> list[float]:
        return [bandit.parameter_hat for bandit in self]

    @property
    def simulation_counts(self) -> list[int]:
        return [len(bandit) for bandit in self]

    @property
    def residuals(self) -> list[float]:
        return [bandit.residual for bandit in self]


@dataclass
class TwoParameterBanditCollection(BanditCollection):
    """
    Container class for a collection of armed bandits,
    where the armed bandits are of a two-parameter
    distribution.
    """
    num_parameters: int = 2

    @classmethod
    def from_parameter_list( # noqa
        cls,
        distribution: Distribution,
        parameter_one_values: list[float],
        parameter_two_values: list[float],
    ):
        """
        Constructor using list(s) of parameters and a bandit type.
        """
        bandit_type = distribution_factory(distribution=distribution)

        bandits = [
            bandit_type(parameter_one_values[i], parameter_two_values[i])
            for i in range(len(parameter_one_values))
        ]

        return cls(bandits=bandits)

    @classmethod
    def from_parameter_distribution( #noqa
        cls,
        distribution: Distribution,
        num_bandits: int,
        parameter_one_mean: float,
        parameter_one_std: float,
        parameter_two_mean: float,
        parameter_two_std: float
    ):
        """
        Constructor using and a bandit type,
        the number of bandits to generate, and the
        mean and std of each parameter.
        """
        parameters = normal(loc=parameter_one_mean,
                            scale=parameter_one_std,
                            size=num_bandits).tolist()
        secondary_parameters = normal(loc=parameter_two_mean,
                                        scale=parameter_two_std,
                                        size=num_bandits).tolist()

        return cls.from_parameter_list(distribution=distribution,
                                       parameter_one_values=parameters,
                                       parameter_two_values=secondary_parameters)


    @property
    def true_secondary_parameters(self) -> list[float]:
        return [bandit.secondary_parameter for bandit in self]

    @property
    def estimated_secondary_parameters(self) -> list[float]:
        return [bandit.secondary_parameter_hat for bandit in self]

    @property
    def secondary_residuals(self) -> list[float]:
        return [bandit.secondary_residual for bandit in self]