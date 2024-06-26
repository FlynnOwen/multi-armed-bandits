from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import StrEnum
from random import uniform

from src.bandit import Bandit, BanditCollection
from src.metrics import OneParameterMetrics, TwoParameterMetrics
from src.utils.utils import ucb


class SimStrategy(StrEnum):
    epsilon_first = "epsilon_first"
    epsilon_decreasing = "epsilon_decreasing"
    epsilon_greedy = "epsilon_greedy"


def simulation_strategy_factory(strategy: SimStrategy, **kwargs) -> SemiUniformStrategy:  # noqa: ANN003
    strategy_map = {
        SimStrategy.epsilon_first: EpsilonFirstStrategy,
        SimStrategy.epsilon_decreasing: EpsilonDecreasingStrategy,
        SimStrategy.epsilon_greedy: EpsilonGreedyStrategy,
    }

    return strategy_map[strategy](**kwargs)


class SemiUniformStrategy(ABC):
    """
    Implements the strategy pattern for 'Semi-Uniform' strategies
    in the Multi-Armed Bandit problem.

    Semi-uniform strategies were the earliest (and simplest)
    strategies discovered to approximately solve the bandit problem.
    """

    def __init__(
        self,
        bandit_collection: BanditCollection,
        num_simulations: int,
        epsilon: float = 0.2
    ):
        if 0 > epsilon < 1:
            raise ValueError(
                f"epsilon must be 0 <= epsilon <= 1" f"got value {epsilon} instead."
            )
        self.bandit_collection = bandit_collection
        self.num_simulations = num_simulations
        self.epsilon = epsilon

        match bandit_collection.num_parameters:
            case 1:
                self.metrics = OneParameterMetrics(bandit_collection)
            case 2:
                self.metrics = TwoParameterMetrics(bandit_collection)

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
                "Consdier using simulate(desired_number) for "
                "executing more simulations."
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


class EpsilonGreedyStrategy(SemiUniformStrategy):
    """
    The best lever is selected for 1-epsilon of the trials,
    and a lever is selected at (Uniform) random for a proportion
    epsilon.

    Epsilon is suggested to be higher (~0.8) for this strategy.
    """

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
        result = bandit.generate()
        self.bandit_collection.results.append({"id": bandit.id, "value": result})


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

    def __init__(
        self,
        bandit_collection: BanditCollection,
        num_simulations: int,
        epsilon: float = 0.8,
        decay_rate: float = 0.05,
        **kwargs,
    ):
        if 0 < self.decay_rate < 1:
            raise ValueError("parameter decay_rate must be 0 < decay_rate < 1.")
        super().__init__(bandit_collection, num_simulations, epsilon, **kwargs)
        self.decay_rate = decay_rate

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
        result = bandit.generate()
        self.bandit_collection.results.append(result)


class EpsilonFirstStrategy(SemiUniformStrategy):
    """
    A pure exploration phase occurs for epsilon * num_simulations trials,
    followed by a pure exploitation phase for (1 - epsilon) * num_simulations
    trials.
    """

    @property
    def exploitation_phase(self) -> bool:
        """
        Boolean: Has the threshold for the exploitation phase passed.
        """
        return self.epsilon * self.num_simulations <= self.simulation_num

    def simulate_one(self) -> None:
        if not self.exploitation_phase:
            bandit = self.bandit_collection.random_bandit
        else:
            bandit = self.bandit_collection.optimal_bandit

        result = bandit.generate()
        self.bandit_collection.results.append(result)


class UCBSimulation:
    """
    Simulation incorporating the Upper Confidence Bound (UCB).

    NOTE: This is incomplete
    """

    def __init__(self, exploitation_constant: float = 0.5) -> None:
        self.exploitation_constant = exploitation_constant

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
