from dataclasses import dataclass

from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.simulation import SemiUniformStrategy


@dataclass
class Metrics:
    """
    Encapsulates metrics of bandits.
    """

    simulation: SemiUniformStrategy
    rounding_dp: int = 2

    def __post_init__(self):
        self.bandit_collection = self.simulation.bandit_collection

    @property
    def num_simulations(self) -> int:
        return self.simulation.simulation_num

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

    @property
    def average_reward_timeseries(self) -> list[float]:
        return list(
            (np.cumsum(self.simulation.results)/np.arange(1, self.num_simulations+1))
            .round(2)
            )

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

    def plots(self) -> None:
        """
        Generates plots to stdout of the multiarmed
        bandit simulation process.
        """
        sns.set_style("darkgrid")
        ax = sns.lineplot(data=self.average_reward_timeseries)

        plt.axhline(y=self.bandit_collection.optimal_bandit.parameter_hat,
                    color="r",
                    linestyle="--",
                    label="Optimal Bandit Estimated Parameter")
        plt.axhline(y=0.5,
                    color="g",
                    linestyle="--",
                    label="Optimal Bandit True Parameter")

        ax.set(title="Simulation Reward Over Time",
            xlabel="Simulation Number",
            ylabel="Average Reward")

        plt.legend()
        plt.show()
