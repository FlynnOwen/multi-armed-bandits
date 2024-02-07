from dataclasses import dataclass

from tabulate import tabulate

from src.bandit import BanditCollection


@dataclass
class Metrics:
    """
    Encapsulates metrics of bandits.
    """

    bandit_collection: BanditCollection
    rounding_dp: int = 2

    @property
    def num_simulations(self) -> int:
        return self.bandit_collection.simulation_num

    @property
    def _ae(self) -> float:
        return round(sum([abs(bandit.residual) for bandit in self.bandit_collection]),
                     self.rounding_dp)

    @property
    def mae(self) -> float:
        return round(self._ae / len(self.bandit_collection), self.rounding_dp)

    @property
    def total_reward(self) -> float:
        return sum([bandit.reward for bandit in self.bandit_collection])

    @property
    def mape(self) -> float:
        return round(
            sum(
            [
                abs(bandit.residual) / bandit.true_parameter
                for bandit in self.bandit_collection
            ],
        ) / len(self.bandit_collection),
        self.rounding_dp)

    def __str__(self) -> str:
        return tabulate(
            [
                ["total simulations", self.num_simulations],
                ["total reward", self.total_reward],
                ["mape", self.mape],
                ["mae", self.mae]
             ],
             headers=["metric", "value"],
             tablefmt="rounded_outline",
             numalign="left"
        )
