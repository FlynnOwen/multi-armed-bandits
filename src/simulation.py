from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import uniform

from tabulate import tabulate

from src.bandit import Bandit, BanditCollection
from src.utils.utils import ucb


@dataclass
class Simulation(ABC):
    """
    Base class for MAB simulations.
    """

    bandit_collection: BanditCollection
    num_simulations: int
    strategy: SemiUniformStrategy
    _simulation_num: int = 0

    @property
    def simulation_num(self) -> int:
        return self._simulation_num

    def full_simulation(self) -> None:
        self.strategy.full_simulation(self.num_simulations, self.bandit_collection)


@dataclass
class EpsilonSimulation(Simulation):
    """
    Simulations that involve 'Epsilon' strategies.
    """


class SemiUniformStrategy(ABC):
    """
    Implements the strategy pattern for 'Semi-Uniform' strategies
    in the Multi-Armed Bandit problem.

    Semi-uniform strategies were the earliest (and simplest)
    strategies discovered to approximately solve the bandit problem.
    """

    epsilon: float = 0.2
    random_bound: float = 0.2

    def gen_random_value(self) -> float:
        return uniform(a=0, b=1)

    def _bandit_strategy(
        self, random_value: float, bandit_collection: BanditCollection
    ) -> Bandit:
        """
        Strategy for which bandit to generate:
        - If a randomly generate value is less than the defined
            bound, return a random bandit.
        - Otherwise return the current best bandit.
        """
        if random_value <= self.random_bound:
            return self.bandit_collection.random_bandit
        return self.bandit_collection.optimal_bandit

    def simulate_one(self, bandit_collection: BanditCollection) -> None:
        random_value = self.gen_random_value()
        bandit = self._bandit_strategy(
            random_value=random_value, bandit_collection=bandit_collection
        )
        bandit.generate()

    @abstractmethod
    def full_simulation(  # noqa: ANN201
        self, num_simulations: int, bandit_collection: BanditCollection
    ):
        pass


class EpsilonGreegyStrategy(SemiUniformStrategy):
    """
    The best lever is selected for 1-epsilon of the trials,
    and a lever is selected at (Uniform) random for a proportion
    epsilon.
    """

    def full_simulation(  # noqa ANN201
        self, num_simulations: int, bandit_collection: BanditCollection
    ):
        pass


class EpsilonFirstStrategy(SemiUniformStrategy):
    """
    A pure exploration phase is followed by a pure exploitation phase.
    """

    def full_simulation(  # noqa ANN201
        self, num_simulations: int, bandit_collection: BanditCollection
    ):
        pass


class EpsilonDecreasingStrategy(SemiUniformStrategy):
    """
    Similar to EpsilonGreegyStrategy, but epsilon gradually decreases.
    This results in highly explorative behaviour, followed by highly
    exploitative behaviour.
    """

    decay_rate: float

    def full_simulation(  # noqa ANN201
        self, num_simulations: int, bandit_collection: BanditCollection
    ):
        pass


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
            q_t=best_bandit.num_simulations,
        )


@dataclass
class Metrics:
    """
    Simulation incorporating the Upper Confidence Bound (UCB).

    NOTE: This is incomplete
    """

    bandit_collection: BanditCollection

    @property
    def total_sims(self) -> int:
        return sum([len(bandit) for bandit in self.bandit_collection])

    @property
    def _ae(self) -> float:
        return sum([bandit.residual for bandit in self.bandit_collection])

    @property
    def mae(self) -> float:
        return self._ae / len(self.bandit_collection)

    @property
    def total_reward(self) -> float:
        return sum([bandit.reward for bandit in self.bandit_collection])

    @property
    def mape(self) -> float:
        return sum(
            [
                bandit.residual / bandit.true_parameter
                for bandit in self.bandit_collection
            ],
        ) / len(self.bandit_collection)

    def __str__(self) -> str:
        return tabulate(
            data=[self.total_sims, self.total_reward, self.mape, self.mae],
            headers=["total simulations", "total reward", "mape", "mae"],
        )
