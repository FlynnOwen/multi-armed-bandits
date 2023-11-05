from random import randint
from dataclasses import dataclass, field


@dataclass
class Bandit:
    """Single bandit, simulating over a Bernoulli distribution."""

    true_parameter: float
    _results: list[int] = field(init=False)

    def generate(self):
        """
        Generates a single result from the pull of the
        armed bandit.
        """
        result = randint(0, 1)
        self._results.append(result)

        return result

    @property
    def parameter_hat(self):
        """
        The estimated parameter for this bandit.
        Note that the Maximum Likelihood estimator is used.
        """
        return sum(self._results)/len(self._results)

    @property
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

    def __le__(self, value: float):
        return self.parameter_hat <= value

    def __gt__(self, value: float):
        return self.parameter_hat > value

    def __ge__(self, value: float):
        return self.parameter_hat >= value
