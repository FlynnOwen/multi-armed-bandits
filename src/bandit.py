import numpy as np

from random import randint
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
            return np.inf
        else:
            return sum(self._results) / len(self._results)

    @cached_property
    def simulations(self):
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
