from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import uniform

from src.bandit import Bandit, BanditCollection
from src.utils.utils import ucb


@dataclass
class Simulation:
    """
    Base class for MAB simulations.
    """

    bandit_collection: BanditCollection
    strategy: SemiUniformStrategy

    def full_simulation(self) -> None:
        self.strategy.full_simulation(self.bandit_collection)


@dataclass
class SemiUniformStrategy(ABC):
    """
    Implements the strategy pattern for 'Semi-Uniform' strategies
    in the Multi-Armed Bandit problem.

    Semi-uniform strategies were the earliest (and simplest)
    strategies discovered to approximately solve the bandit problem.
    """

    bandit_collection: BanditCollection
    num_simulations: int
    epsilon: float

    @property
    def simulation_num(self) -> int:
        return self.bandit_collection.simulation_num

    def gen_random_uniform(self) -> float:
        return uniform(a=0, b=1)

    @abstractmethod
    def simulate_one(self) -> None:
        pass

    def full_simulation(self) -> BanditCollection:
        """
        Executes self.simulate_one up until num_simulations has been reached.
        Raises an exception if simulation_num > num_simulations.
        """
        if self.simulation_num >= self.num_simulations:
            raise Exception(
                f"There have already been {self.simulation_num} simulations"
                f"Consdier using simulate(desired_number) for "
                f"executing more simulations."
            )
        [self.simulate_one() for _ in range(self.simulation_num, self.num_simulations)]
        return self.bandit_collection

    def simulate(self, num_sims: int) -> BanditCollection:
        """
        Executes exactly num_sims simulations, regardless of how many have occurred
        previously and what the value self.num_simulations is set to.
        """
        [self.simulate_one() for _ in range(num_sims)]
        return self.bandit_collection


@dataclass
class EpsilonGreedyStrategy(SemiUniformStrategy):
    """
    The best lever is selected for 1-epsilon of the trials,
    and a lever is selected at (Uniform) random for a proportion
    epsilon.
    """

    epsilon: float = 0.2

    def _bandit_strategy(self, random_value: float) -> Bandit:
        """
        Strategy for which bandit to generate:
        - If a randomly generate value is less than the defined
            bound, return a random bandit.
        - Otherwise return the current best bandit.
        """
        if random_value <= self.epsilon:
            return self.bandit_collection.random_bandit
        return self.bandit_collection.optimal_bandit

    def simulate_one(self) -> None:
        random_value = self.gen_random_uniform()
        bandit = self._bandit_strategy(random_value=random_value)
        bandit.generate()


@dataclass
class EpsilonDecreasingStrategy(SemiUniformStrategy):
    """
    Similar to EpsilonGreegyStrategy, but epsilon gradually decreases.
    This results in highly explorative behaviour, followed by highly
    exploitative behaviour.

    NOTE: The formula for this strategy is:
          epsilon_curr = e^{epsilon - (decay_rate * simulation_num)}

    Is this therefore necessary that 0 <= decay_rate <= 1.
    Choose lower values of decay_rate for more explorative
    behaviour, and higher values for more exploitative behaviour.

    Note that decay rate is proportional to the number of simulations,
    and epsilon, all parameters should be kept in mind during selection.
    """

    epsilon: float = 0.8
    decay_rate: float = 0.05

    def __post_init__(self):
        if 0 < self.decay_rate < 1:
            raise ValueError("parameter decay_rate must be 0 < decay_rate < 1.")

    @property
    def epsilon_curr(self) -> float:
        return math.exp(self.epsilon - (self.decay_rate * self.simulation_num))

    def _bandit_strategy(self, random_value: float) -> Bandit:
        """
        Strategy for which bandit to generate:
        - If a randomly generate value is less than the defined
            bound, return a random bandit.
        - Otherwise return the current best bandit.
        """
        if random_value <= self.epsilon_curr:
            return self.bandit_collection.random_bandit
        return self.bandit_collection.optimal_bandit

    def simulate_one(self) -> None:
        """
        Generates a single value from a bandit.
        """
        random_value = self.gen_random_uniform()
        bandit = self._bandit_strategy(
            random_value=random_value, bandit_collection=self.bandit_collection
        )
        bandit.generate()


@dataclass
class EpsilonFirstStrategy(SemiUniformStrategy):
    """
    A pure exploration phase occurs for epsilon * num_simulations trials,
    followed by a pure exploitation phase for (1 - epsilon) * num_simulations
    trials.
    """

    epsilon: float = 0.8

    @property
    def exploitation_phase(self) -> bool:
        """
        Boolean: Has the threshold for the exploitation phase passed.
        """
        return self.epsilon * self.num_simulations >= self.simulation_num

    def simulate_one(self) -> None:
        if not self.exploitation_phase:
            bandit = self.bandit_collection.random_bandit
        else:
            bandit = self.bandit_collection.optimal_bandit

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
        return ucb(
            q=best_bandit.parameter_hat,
            t=self.num_simulations,
            c=self.exploitation_constant,
            q_t=len(best_bandit),
        )
