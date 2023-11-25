from math import inf
from random import randint, choice
from dataclasses import dataclass, field
from functools import cached_property, total_ordering


@total_ordering
@dataclass
class Bandit:
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

        return result

    @cached_property
    def parameter_hat(self):
        """
        The estimated parameter for this bandit.
        Note that the Maximum Likelihood estimator is used.
        """
        if not self._results:
            return inf
        else:
            return sum(self._results) / len(self._results)

    @cached_property
    def num_simulations(self):
        """
        Number of simulations that this armed bandit has performed.
        """
        return len(self._results)

    def __len__(self):
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

    @cached_property
    def optimal_bandit(self) -> Bandit:
        return self.bandits[self.bandits.index(max(self.bandits))]

    @property
    def random_bandit(self) -> Bandit:
        return choice(self.bandits)

    def add_bandit(self, bandit: Bandit):
        self.bandits.append(bandit)
