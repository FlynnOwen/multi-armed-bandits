from dataclasses import dataclass

from tabulate import tabulate

from src.bandit import BanditCollection


@dataclass
class Metrics:
    """
    Encapsulates metrics of bandits.
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
