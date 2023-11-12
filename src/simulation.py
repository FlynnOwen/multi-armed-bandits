from random import choice, uniform
from functools import cached_property
from dataclasses import dataclass

from src.bandit import Bandit


@dataclass
class BanditCollection:
    """
    Container class for a collection of armed bandits.
    """
    bandits: list[Bandit]

    @cached_property
    def optimal_bandit(self) -> Bandit:
        return self.bandits[self.bandits.index(max(self.bandits))]

    def random_bandit(self) -> Bandit:
        return choice(self.bandits)

    def add_bandit(self, bandit: Bandit):
        self.bandits.append(bandit)


@dataclass
class Simulation:
    """
    Defines simulation parameters.
    """

    bandit_collection: BanditCollection
    exploitation_constant: float = 0.5
    random_bound: float = 0.2

    def __iter__(self):
        self._pull_count = 0
        return self

    def __next__(self):
        self._pull_count += 1
        random_value = self.gen_random_value()
        bandit = self.bandit_strategy(random_value=random_value)
        return bandit.generate()


    def _upper_confidence_bound(self):
        """
        Calculates the upper confidence bound given the current
        bandit collection.
        """
        pass

    def gen_random_value(self):
        return uniform(a=0, b=1)


    def _bandit_strategy(self, random_value: float):
        """
        Strategy for which bandit to generate:
        - If a randomly generate value is less than the defined
            bound, return a random bandit.
        - Otherwise return the current best bandit.
        """
        if random_value <= self.random_bound:
            return self.bandit_collection.random_bandit
        else:
            return self.bandit_collection.optimal_bandit


def simulate(simulation: Simulation, pull_count: int):
    """
    Run a simulation for pull_count iterations.
    """
    pass
