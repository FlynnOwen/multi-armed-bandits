from dataclasses import dataclass

from src.bandit import Bandit


@dataclass
class Simulation:
    bandits: list[Bandit]

    def optimal_bandit(self):
        pass

    def random_bandit(self):
        pass