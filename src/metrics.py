from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.bandit import BanditCollection


class ParameterType:
    primary = "Primary"
    secondary = "Secondary"


@dataclass
class Metrics(ABC):
    """
    Encapsulates metrics of bandits.
    """

    bandit_collection: BanditCollection
    rounding_dp: int = 2

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        pass

    def __post_init__(cls):  # noqa N801
        if cls.num_parameters != cls.bandit_collection.num_parameters:
            raise ValueError(
                f"You may only utilize bandits with"
                f" {cls.num_parameters} to this class."
                f" Got {cls.bandit_collection.num_parameters} instead."
            )

    @property
    def num_simulations(self) -> int:
        return self.bandit_collection.simulation_num

    def _ae(self, residuals: list[float]) -> float:
        return round(sum(map(abs, residuals)), self.rounding_dp)

    def _mae(self, residuals: list[float]) -> float:
        return round(
            self._ae(residuals) / len(self.bandit_collection), self.rounding_dp
        )

    @property
    def mae(self) -> float:
        return self._mae(residuals=self.bandit_collection.residuals)

    def _mape(self, residuals: list[float], parameters: list[float]) -> float:
        return round(
            sum(
                [
                    abs(residuals[i]) / parameters[i]
                    for i in range(len(self.bandit_collection))
                ],
            )
            / len(self.bandit_collection),
            self.rounding_dp,
        )

    @property
    def mape(self) -> float:
        return self._mape(
            residuals=self.bandit_collection.residuals,
            parameters=self.bandit_collection.true_parameters,
        )

    @property
    def total_reward(self) -> float:
        return sum([bandit.reward for bandit in self.bandit_collection])

    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def average_reward_timeseries(self) -> list[float]:
        return list(
            (
                np.cumsum(
                    [result["value"] for result in self.bandit_collection.results]
                )
                / np.arange(1, self.num_simulations + 1)
            ).round(self.rounding_dp)
        )

    @property
    def bandit_use_timeseries(self) -> list[float]:
        """
        TODO: Implement.
        """
        values = [result["id"] for result in self.bandit_collection.results]

        def find_index_ranges(values: list[int]) -> dict[int, list[tuple[int, int]]]:
            """
            For each bandit id, find the continuous ranges that this
            bandit was selected for.
            FIXME: Not working
            HACK: This could be written better.
            """
            ranges = defaultdict(list)
            idx = 0
            while idx < len(values) - 1:
                starting_idx = idx
                while values[idx] == values[idx + 1]:
                    if idx + 1 == len(values) - 1:
                        break
                    idx += 1
                ranges[values[idx]].append((starting_idx + 1, idx + 2))
                idx += 1

            return ranges

        ranges = find_index_ranges(values)
        _, ax = plt.subplots()
        for k, v in ranges.items():
            ax.broken_barh(v, yrange=(k, 1))
        plt.show()

    def bandit_use_stackplot(self) -> None:
        num_simulations = self.bandit_collection.simulation_num

        results_df = pd.DataFrame(
            {
                "pull_number": range(1, len(self.bandit_collection.results) + 1),
                "id": [result["id"] for result in self.bandit_collection.results],
            }
        )
        results_df["cum_id"] = results_df.groupby("id").cumcount() + 1
        results_df["cum_proportion"] = results_df["cum_id"] / results_df["pull_number"]
        results_df = results_df.pivot_table("cum_id", "pull_number", "id").ffill()
        results_df = results_df.fillna(0)
        results_df[results_df.columns] = results_df[results_df.columns].div(
            results_df.index, axis=0
        )
        results_df = results_df.reset_index().drop(columns=["pull_number"])

        plt.stackplot(
            range(num_simulations),
            results_df.to_dict(orient="list").values(),
            labels=zip(
                self.bandit_collection.bandit_ids(),
                [
                    round(param, self.rounding_dp)
                    for param in self.bandit_collection.true_parameters
                ],
                [
                    round(param, self.rounding_dp)
                    for param in self.bandit_collection.estimated_parameters
                ],
            ),
        )
        # TODO: Implement below if using epsilon_first
        # plt.axvline(
        #    x=100,
        #    color="black",
        #    linestyle="-",
        #    label="Cutoff",
        # )
        plt.legend(title="(ID, True Param, Param Hat)", loc="upper right")
        plt.xlabel("Pull Number")
        plt.ylabel("Proportion of Total Pulls")
        plt.title("Share of Total Pulls by Bandit Over Time")
        plt.show()

    def residual_barplots(
        self,
        estimated_parameters: list[float],
        true_parameters: list[float],
        parameter_type: ParameterType,
    ) -> None:
        """
        Barplots showing estimated vs true Parameter
        values for each bandit.
        """
        parameters = estimated_parameters + true_parameters

        parameter_types = ["Estimated" for _ in range(len(self.bandit_collection))] + [
            "True" for _ in range(len(self.bandit_collection))
        ]

        bandit_id = [bandit.id for bandit in self.bandit_collection] * 2

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
        ax.set(
            title=f"Simulation Estimated {parameter_type} "
            f"vs True {parameter_type} Parameters"
        )
        plt.show()

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
            linestyle="-",
            label="Optimal Bandit Estimated Parameter",
        )
        plt.axhline(
            y=self.bandit_collection.optimal_bandit.parameter,
            color="g",
            linestyle="--",
            label="Optimal Bandit True Parameter",
        )

        ax.set(
            title="Simulation Reward Over Time",
            xlabel="Simulation Number",
            ylabel="Average Reward",
        )

        plt.legend()
        plt.show()

    def cumulative_reward_timeseries_plot(self) -> None:
        """
        Generates a timeseries plot of the cumulative reward
        recieved during each stage of the simulation process.

        This includes lines of expected cumulatove reward over all
        bandits, as well as best and worst parameters.
        """
        best_expected_parameter = np.max(self.bandit_collection.true_parameters)
        worst_expected_parameter = np.min(self.bandit_collection.true_parameters)
        avg_expected_parameter = np.mean(self.bandit_collection.true_parameters)

        cumulative_rewards = pd.DataFrame(
            {
                "True Cumulative Reward": list(
                    np.cumsum(
                        [result["value"] for result in self.bandit_collection.results]
                    )
                ),
                "Expected Average Reward": [
                    avg_expected_parameter * sim for sim in range(self.num_simulations)
                ],
                "Expected Best Reward": [
                    best_expected_parameter * sim for sim in range(self.num_simulations)
                ],
                "Expected Worst Reward": [
                    worst_expected_parameter * sim
                    for sim in range(self.num_simulations)
                ],
            }
        )
        ax = sns.lineplot(data=cumulative_rewards)

        ax.set(
            title="Simulation Cumulative Reward",
            xlabel="Simulation Number",
            ylabel="Cumulative Reward",
        )

        plt.legend()
        plt.show()

    @abstractmethod
    def generate_plots(self) -> None:
        """
        Generates plots to stdout of the multiarmed
        bandit simulation process.
        """


@dataclass
class OneParameterMetrics(Metrics):
    """
    Metrics for a simulation over bandits that are from a
    one-parameter distribution.
    """

    num_parameters = 1

    def __str__(self) -> str:
        return tabulate(
            [
                [
                    "optimal bandit parameter true",
                    round(
                        self.bandit_collection.optimal_bandit.parameter,
                        self.rounding_dp,
                    ),
                ],
                [
                    "optimal bandit parameter hat",
                    round(
                        self.bandit_collection.optimal_bandit.parameter_hat,
                        self.rounding_dp,
                    ),
                ],
                ["total simulations", round(self.num_simulations, self.rounding_dp)],
                ["total reward", round(self.total_reward, self.rounding_dp)],
                ["mape", round(self.mape, self.rounding_dp)],
                ["mae", round(self.mae, self.rounding_dp)],
            ],
            headers=["metric", "value"],
            tablefmt="rounded_outline",
            numalign="left",
        )

    def generate_plots(self) -> None:
        """
        Generates plots to stdout of the multiarmed
        bandit simulation process.
        """
        self.bandit_use_stackplot()
        self.residual_barplots(
            true_parameters=self.bandit_collection.true_parameters,
            estimated_parameters=self.bandit_collection.estimated_parameters,
            parameter_type=ParameterType.primary,
        )
        self.reward_timeseries_plot()
        self.cumulative_reward_timeseries_plot()


@dataclass
class TwoParameterMetrics(Metrics):
    """
    Metrics for a simulation over bandits that are from a
    two-parameter distribution.
    """

    num_parameters = 2

    @property
    def secondary_mae(self) -> float:
        return self._mae(residuals=self.bandit_collection.residuals)

    @property
    def secondary_mape(self) -> float:
        return self._mape(
            residuals=self.bandit_collection.residuals,
            parameters=self.bandit_collection.true_parameters,
        )

    def __str__(self) -> str:
        return tabulate(
            [
                [
                    "optimal bandit parameter true",
                    round(
                        self.bandit_collection.optimal_bandit.parameter,
                        self.rounding_dp,
                    ),
                ],
                [
                    "optimal bandit parameter hat",
                    round(
                        self.bandit_collection.optimal_bandit.parameter_hat,
                        self.rounding_dp,
                    ),
                ],
                [
                    "optimal bandit secondary parameter true",
                    round(
                        self.bandit_collection.optimal_bandit.secondary_parameter,
                        self.rounding_dp,
                    ),
                ],
                [
                    "optimal bandit secondary parameter hat",
                    round(
                        self.bandit_collection.optimal_bandit.secondary_parameter_hat,
                        self.rounding_dp,
                    ),
                ],
                ["total simulations", round(self.num_simulations, self.rounding_dp)],
                ["total reward", round(self.total_reward, self.rounding_dp)],
                ["mape", round(self.mape, self.rounding_dp)],
                ["mae", round(self.mae, self.rounding_dp)],
                ["secondary mape", round(self.secondary_mape, self.rounding_dp)],
                ["secondary mae", round(self.secondary_mae, self.rounding_dp)],
            ],
            headers=["metric", "value"],
            tablefmt="rounded_outline",
            numalign="left",
        )

    def generate_plots(self) -> None:
        """
        Generates plots to stdout of the multiarmed
        bandit simulation process.
        """
        self.bandit_use_stackplot()
        self.residual_barplots(
            true_parameters=self.bandit_collection.true_parameters,
            estimated_parameters=self.bandit_collection.estimated_parameters,
            parameter_type=ParameterType.primary,
        )
        self.residual_barplots(
            true_parameters=self.bandit_collection.true_secondary_parameters,
            estimated_parameters=self.bandit_collection.estimated_secondary_parameters,
            parameter_type=ParameterType.secondary,
        )
        self.reward_timeseries_plot()
        self.cumulative_reward_timeseries_plot()
