from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

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
        return round(
            sum([abs(bandit.residual) for bandit in self.bandit_collection]),
            self.rounding_dp,
        )

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
            )
            / len(self.bandit_collection),
            self.rounding_dp,
        )

    @property
    def average_reward_timeseries(self) -> list[float]:
        return list(
            (
                np.cumsum(self.simulation.results)
                / np.arange(1, self.num_simulations + 1)
            ).round(self.rounding_dp)
        )

    def __str__(self) -> str:
        return tabulate(
            [
                ["best parameter hat",
                 round(self.bandit_collection.optimal_bandit.parameter_hat, 2)],
                 ["best parameter true",
                 self.bandit_collection.optimal_bandit.true_parameter],
                ["total simulations", self.num_simulations],
                ["total reward", self.total_reward],
                ["mape", self.mape],
                ["mae", self.mae],
            ],
            headers=["metric", "value"],
            tablefmt="rounded_outline",
            numalign="left",
        )

    def residual_barplots(self) -> None:
        """
        Barplots showing estimated vs true Parameter
        values for each bandit.
        """
        parameters = [
            bandit.parameter_hat for bandit in self.bandit_collection.bandits
        ] + [bandit.true_parameter for bandit in self.bandit_collection.bandits]

        parameter_types = ["Estimated" for _ in range(len(self.bandit_collection))] + [
            "True" for _ in range(len(self.bandit_collection))
        ]

        bandit_id = list(range(1, len(self.bandit_collection.bandits) + 1)) * 2

        data = pd.DataFrame(
            {
                "Parameter Value": parameters,
                "Parameter Type": parameter_types,
                "Bandit ID": bandit_id,
            }
        )

        sns.set_theme(style="whitegrid")
        ax = sns.catplot(
            data=data,
            x="Bandit ID",
            y="Parameter Value",
            hue="Parameter Type",
            kind="bar",
            alpha=0.8,
        )
        ax.despine(left=True)
        ax.set(title="Simulation Estimated vs True Parameters")

    def pull_residual_scatterplot(self) -> None:
        """
        Scatterplot showing number of bandit pulls vs
        the absolute residual for each bandit.
        """
        simulation_counts = [len(bandit) for bandit in self.bandit_collection]
        abs_residuals = [abs(bandit.residual) for bandit in self.bandit_collection]
        sim_counts_residuals = pd.DataFrame(
            {"Number of Pulls": simulation_counts, "Absolute Resdidual": abs_residuals}
        )

        ax = sns.scatterplot(
            color="r",
            data=sim_counts_residuals,
            x="Absolute Resdidual",
            y="Number of Pulls",
        )
        ax.set(title="Number of Bandit Pulls vs Absolute Residual of Bandit")

    def reward_timeseries_plot(self) -> None:
        """
        Generates a timeseries plot of the average reward
        recieved during each stage of the simulation process.
        """
        sns.set_style("darkgrid")
        ax = sns.lineplot(data=self.average_reward_timeseries)

        plt.axhline(
            y=self.bandit_collection.optimal_bandit.parameter_hat,
            color="r",
            linestyle="--",
            label="Optimal Bandit Estimated Parameter",
        )
        plt.axhline(
            y=0.5, color="g", linestyle="--", label="Optimal Bandit True Parameter"
        )

        ax.set(
            title="Simulation Reward Over Time",
            xlabel="Simulation Number",
            ylabel="Average Reward",
        )

        plt.legend()
        plt.show()

    def generate_plots(self) -> None:
        """
        Generates plots to stdout of the multiarmed
        bandit simulation process.
        """
        self.residual_barplots()
        self.pull_residual_scatterplot()
        self.reward_timeseries_plot()
