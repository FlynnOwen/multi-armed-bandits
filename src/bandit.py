from dataclasses import dataclass, field


@dataclass
class Bandit:
    """Single bandit."""

    true_parameter: float
    _results: list[int] = field(init=False)

    @property
    def parameter_hat(self):
        return self.true_parameter

    @property
    def simulations(self):
        pass

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
