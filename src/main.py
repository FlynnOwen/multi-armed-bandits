import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from src.bandit import (
    BanditCollection,
    Distribution,
    TwoParameterBanditCollection,
    distribution_factory,
)
from src.simulation import Strategy, strategy_factory

app = typer.Typer()


class BanditGenMethod(StrEnum):
    from_list = "from_list"
    from_dist = "from_distribution"

    @classmethod
    def values(cls) -> list[str]:
        return [c.value for c in cls]


def main(
    strategy: Strategy,
    num_simulations: int,
    print_metrics: bool,
    print_plots: bool,
    epsilon: float,
    decay_rate: float | None,
    bandit_collection: BanditCollection,
) -> None:
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


@app.command()
def simulate_fixed(  # noqa
    strategy: Strategy = Strategy.epsilon_greedy,
    distribution: Distribution = Distribution.bernoulli,
    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
    print_metrics: bool = False,
    print_plots: bool = False,
    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
    decay_rate: float = 0.05,
    parameter_one_values: list[float] = None,
    parameter_two_values: list[float] = None,
):
    """
    Run a simulation from a fixed set of bandit
    parameters.
    """
    if strategy == Strategy.epsilon_decreasing and decay_rate is None:
        raise ValueError(
            "Arg 'decay_rate' must be passed if using strategy" "'epsilon_first'."
        )

    if (
        distribution in Distribution.two_parameter_family
        and parameter_two_values is None
    ):
        raise ValueError("'parameter_two_values' " "must be passed as args.")

    if distribution in Distribution.two_parameter_family and len(
        parameter_one_values
    ) != len(parameter_two_values):
        raise ValueError(
            "Length of parameter 'parameter_one_values' must be equal to "
            f"length of paramter 'parameter_two_values'. Got {len(parameter_one_values)} "
            f"and {len(parameter_two_values)} respectively."
        )

    num_parameters = distribution_factory(distribution).num_parameters
    if num_parameters == 1:
        bandit_collection = BanditCollection.from_parameter_list(
            distribution=distribution, parameter_one_values=parameter_one_values
        )
    if num_parameters == 2:
        bandit_collection = TwoParameterBanditCollection.from_parameter_list(
            distribution=distribution,
            parameter_one_values=parameter_one_values,
            parameter_two_values=parameter_two_values,
        )
    main(
        strategy=strategy,
        num_simulations=num_simulations,
        print_metrics=print_metrics,
        print_plots=print_plots,
        epsilon=epsilon,
        decay_rate=decay_rate,
        bandit_collection=bandit_collection,
    )


@app.command()
def simulate_generate(  # noqa
    parameter_one_mean: float,
    parameter_one_std: float,
    num_bandits: int = typer.Argument(min=2),
    strategy: Strategy = Strategy.epsilon_greedy,
    distribution: Distribution = Distribution.bernoulli,
    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
    print_metrics: bool = False,
    print_plots: bool = False,
    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
    decay_rate: float = 0.05,
    parameter_two_mean: float = None,
    parameter_two_std: float = None,
):
    """
    Run a simulation from a set of bandits
    whose parameters are randomly generated
    according to some distribution.
    """
    if strategy == Strategy.epsilon_decreasing and decay_rate is None:
        raise ValueError(
            "Arg 'decay_rate' must be passed if using strategy" "'epsilon_first'."
        )

    if distribution in Distribution.two_parameter_family and (
        parameter_two_mean is None or parameter_two_std is None
    ):
        raise ValueError(
            "Either 'parameter_two_values' or "
            "'parameter_two_mean' and 'paramater_two_std' "
            "must be passed as args."
        )

    if parameter_one_mean is None or parameter_one_std is None:
        raise ValueError(
            "Either 'parameter_one_values' or "
            "'parameter_one_mean' and 'paramater_one_std' "
            "must be passed as args."
        )

    num_parameters = distribution_factory(distribution).num_parameters
    if num_parameters == 1:
        bandit_collection = BanditCollection.from_parameter_distribution(
            distribution=distribution,
            num_bandits=num_bandits,
            parameter_one_mean=parameter_one_mean,
            parameter_one_std=parameter_one_std,
        )
    elif num_parameters == 2:
        bandit_collection = TwoParameterBanditCollection.from_parameter_distribution(
            distribution=distribution,
            num_bandits=num_bandits,
            parameter_one_mean=parameter_one_mean,
            parameter_one_std=parameter_one_std,
            parameter_two_mean=parameter_two_mean,
            parameter_two_std=parameter_two_std,
        )
    main(
        strategy=strategy,
        num_simulations=num_simulations,
        print_metrics=print_metrics,
        print_plots=print_plots,
        epsilon=epsilon,
        decay_rate=decay_rate,
        bandit_collection=bandit_collection,
    )


@app.command()
def simulate_from_json(config: str,
                       bandit_gen_method: BanditGenMethod) -> None:
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

    if bandit_gen_method == BanditGenMethod.from_dist.value:
        simulate_generate(**simulation_args)
    elif bandit_gen_method == BanditGenMethod.from_list.value:
        simulate_fixed(**simulation_args)
    else:
        raise ValueError(
            "Arg 'bandit_gen_method' must be one of" f"{BanditGenMethod.values()}."
        )


@app.command()
def list_distributions() -> list[str]:
    one_parameter = {dist.value for dist in Distribution.one_parameter_family}
    two_parameter = {dist.value for dist in Distribution.two_parameter_family}

    print("One Parameter Distributions: \n"  #noqa
          f"{one_parameter} \n \n"
          "Two Parameter Distributions: \n"
          f"{two_parameter}")


if __name__ == "__main__":
    app()
