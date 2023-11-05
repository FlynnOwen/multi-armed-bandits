from random import choice
from functools import cached_property
from dataclasses import dataclass

from src.bandit import Bandit


@dataclass
class Simulation:
    """
    Controller class for a full simulation over a collection
    of armed bandits.
    """
    bandits: list[Bandit]

    @cached_property
    def optimal_bandit(self) -> Bandit:
        return self.bandits[self.bandits.index(max(self.bandits))]

    def get_random_bandit(self) -> Bandit:
        return choice(self.bandits)

    def add_bandit(self, bandit: Bandit):
        self.bandits.append(bandit)
