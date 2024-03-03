from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated
from abc import ABC, abstractmethod, abstractproperty

import typer

from src.bandit import (
    OneParamDistribution,
    TwoParamDistribution,
    BanditCollection,
    TwoParameterBanditCollection
)
from src.simulation import Strategy, strategy_factory
from src.utils.utils import ExtStrEnum

app = typer.Typer()


class CLICommands(ExtStrEnum):
    ONE_PARAM_FIXED = "simulate-one-param-fixed"
    TWO_PARAM_FIXED = "simulate-two-param-fixed"
    ONE_PARAM_GEN = "simulate-one-param-generate"
    TWO_PARAM_GEN = "simulate-two-param-generate"

    @classmethod
    def command_factory(cls, value: CLICommands) -> CLICommand:
        return {
            cls.ONE_PARAM_FIXED: OneParamFixed,
            cls.TWO_PARAM_FIXED: TwoParamFixed,
            cls.ONE_PARAM_GEN: OneParamGen,
            cls.TWO_PARAM_GEN: TwoParamGen
         }.get(value)


def run_simulation(strategy: Strategy,
                   num_simulations: int,
                   print_metrics: bool,
                   print_plots: bool,
                   epsilon: float,
                   decay_rate: float | None,
                   bandit_collection: BanditCollection) -> None:

        simulation = strategy_factory(
        strategy=strategy,
        bandit_collection=bandit_collection,
        num_simulations=num_simulations,
        epsilon=epsilon,
        decay_rate=decay_rate,  # HACK: This parameter should only be passed to some strategies. # noqa: E501
    )

        simulation.full_simulation()

        if print_plots:
            simulation.metrics.generate_plots()
        if print_metrics:
            print(simulation.metrics)  # noqa: T201

class CLICommand(ABC):

    @abstractproperty
    def command_name(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def cli_command() -> None:
        pass


class OneParamFixed(CLICommand):
    command_name = "simulate-one-param-fixed"

    @staticmethod
    @app.command(command_name)
    def cli_command(strategy: Strategy = Strategy.epsilon_greedy,  # noqa
                    distribution: OneParamDistribution = OneParamDistribution.bernoulli,
                    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
                    print_metrics: bool = False,
                    print_plots: bool = False,
                    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
                    decay_rate: float = 0.05,
                    parameter_one_values: list[float] = None) -> None:
        """
        Run a simulation from a fixed set of bandit
        parameters.
        """
        bandit_collection = BanditCollection.from_parameter_list(
            distribution=distribution, parameter_one_values=parameter_one_values
        )

        run_simulation(
            strategy=strategy,
            num_simulations=num_simulations,
            print_metrics=print_metrics,
            print_plots=print_plots,
            epsilon=epsilon,
            decay_rate=decay_rate,
            bandit_collection=bandit_collection,
        )


class TwoParamFixed(CLICommand):
    command_name = "simulate-two-param-fixed"

    @staticmethod
    @app.command(command_name)
    def cli_command(strategy: Strategy = Strategy.epsilon_greedy, #noqa
                    distribution: TwoParamDistribution = TwoParamDistribution.gaussian,
                    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
                    print_metrics: bool = False,
                    print_plots: bool = False,
                    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
                    decay_rate: float = 0.05,
                    parameter_one_values: list[float] = None,
                    parameter_two_values: list[float] = None,
                ) -> None:
        """
        Run a simulation from a fixed set of bandit
        parameters.
        """
        if parameter_one_values != len(parameter_two_values):
            raise ValueError(
                "Length of parameter 'parameter_one_values' must be equal to "
                f"length of paramter 'parameter_two_values'. Got {len(parameter_one_values)} "
                f"and {len(parameter_two_values)} respectively."
            )

        bandit_collection = TwoParameterBanditCollection.from_parameter_list(
            distribution=distribution,
            parameter_one_values=parameter_one_values,
            parameter_two_values=parameter_two_values,
        )
        run_simulation(
            strategy=strategy,
            num_simulations=num_simulations,
            print_metrics=print_metrics,
            print_plots=print_plots,
            epsilon=epsilon,
            decay_rate=decay_rate,
            bandit_collection=bandit_collection,
        )


class OneParamGen(CLICommand):
    command_name = "simulate-one-param-generate"

    @staticmethod
    @app.command(command_name)
    def cli_command(parameter_one_mean: float,  # noqa
                    parameter_one_std: float,
                    num_bandits: int = typer.Argument(min=2),
                    strategy: Strategy = Strategy.epsilon_greedy,
                    distribution: OneParamDistribution = OneParamDistribution.bernoulli,
                    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
                    print_metrics: bool = False,
                    print_plots: bool = False,
                    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
                    decay_rate: float = 0.05) -> None:
        """
        Run a simulation from a set of bandits
        whose parameters are randomly generated
        according to some distribution.
        """
        if parameter_one_mean is None or parameter_one_std is None:
            raise ValueError(
                "Either 'parameter_one_values' or "
                "'parameter_one_mean' and 'paramater_one_std' "
                "must be passed as args."
            )

        bandit_collection = OneParamDistribution.from_parameter_distribution(
            distribution=distribution,
            num_bandits=num_bandits,
            parameter_one_mean=parameter_one_mean,
            parameter_one_std=parameter_one_std
        )
        run_simulation(
            strategy=strategy,
            num_simulations=num_simulations,
            print_metrics=print_metrics,
            print_plots=print_plots,
            epsilon=epsilon,
            decay_rate=decay_rate,
            bandit_collection=bandit_collection,
        )


class TwoParamGen(CLICommand):
    command_name = "simulate-two-param-generate"

    @staticmethod
    @app.command(command_name)
    def cli_command(parameter_one_mean: float, #noqa
                    parameter_one_std: float,
                    parameter_two_mean: float,
                    parameter_two_std: float,
                    num_bandits: int = typer.Argument(min=2),
                    strategy: Strategy = Strategy.epsilon_greedy,
                    distribution: TwoParamDistribution = TwoParamDistribution.gaussian,
                    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
                    print_metrics: bool = False,
                    print_plots: bool = False,
                    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
                    decay_rate: float = 0.05) -> None:
        """
        Run a simulation from a set of bandits
        whose parameters are randomly generated
        according to some distribution.
        """
        bandit_collection = TwoParameterBanditCollection.from_parameter_distribution(
            distribution=distribution,
            num_bandits=num_bandits,
            parameter_one_mean=parameter_one_mean,
            parameter_one_std=parameter_one_std,
            parameter_two_mean=parameter_two_mean,
            parameter_two_std=parameter_two_std,
        )
        run_simulation(
            strategy=strategy,
            num_simulations=num_simulations,
            print_metrics=print_metrics,
            print_plots=print_plots,
            epsilon=epsilon,
            decay_rate=decay_rate,
            bandit_collection=bandit_collection,
        )


@app.command()
def simulate_from_json(
    command: CLICommands,
    config: str
    ) -> None:
    """
    Runs a multi-armed bandit simulation with
    configuration (arguments) provided via a json
    file.

    TODO: Create example json schema.
    """
    config = Path(config)
    if not config.is_file() and config.suffix != ".json":
        raise ValueError("config must be suffixed with '.json'.")
    with Path.open(config) as config_file:
        simulation_args = json.load(config_file)

    sim_command = CLICommands.command_factory(value=command)
    sim_command().cli_command(**simulation_args)


@app.command()
def list_distributions() -> list[str]:
    print("One Parameter Distributions: \n"  #noqa
          f"{OneParamDistribution.values()} \n \n"
          "Two Parameter Distributions: \n"
          f"{TwoParamDistribution.values()}")


if __name__ == "__main__":
    app()
