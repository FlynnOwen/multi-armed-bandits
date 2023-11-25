from random import uniform
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.bandit import Bandit, BanditCollection
from src.utils.utils import ucb


@dataclass
class Simulation(ABC):
    """
    Base class for MAB simulations.
    """

    bandit_collection: BanditCollection
    num_simulations: int
    _simulation_num: int = 0

    @property
    def simulation_num(self):
        return self._simulation_num

    @abstractmethod
    def _bandit_strategy(self):
        """
        Strategy for which bandit to generate
        """
        pass

    @abstractmethod
    def simulate_one(self):
        pass

    def simulation(self) -> None:
        [self.simulate_one() for _ in range(self.num_simulations)]


@dataclass
class EpsilonSimulation(Simulation):
    """
    Simulations that involve 'Epsilon' strategies.
    """
    random_bound: float = 0.2

    def gen_random_value(self) -> float:
        return uniform(a=0, b=1)


    def _bandit_strategy(self, random_value: float) -> Bandit:
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

    def simulate_one(self) -> None:
        self._simulation_num += 1
        random_value = self.gen_random_value()
        bandit = self._bandit_strategy(random_value=random_value)
        bandit.generate()


@dataclass
class UCBSimulation(Simulation):
    """
    Simulation incorporating the Upper Confidence Bound (UCB).

    NOTE: This is incomplete
    """
    exploitation_constant: float = 0.5

    def _ucb(self) -> float:
        """
        Calculates the upper confidence bound given the current
        bandit collection.
        """
        best_bandit = self.bandit_collection.optimal_bandit
        return ucb(q=best_bandit.parameter_hat,
                    t=self.num_simulations,
                    c=self.exploitation_constant,
                    q_t=best_bandit.num_simulations)

    def simulate_one(self):
        pass


def simulate(simulation: Simulation, pull_count: int):
    """
    Run a simulation for pull_count iterations.
    """
    for _ in range(pull_count):
        simulation.simulate()
