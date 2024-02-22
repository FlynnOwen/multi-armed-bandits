import json
from pathlib import Path
from typing import Annotated, Optional
from enum import StrEnum

import typer

from src.bandit import BanditCollection, Distribution
from src.simulation import Strategy, strategy_factory

app = typer.Typer()


class BanditGenMethod(StrEnum):
    from_list = "from_list"
    from_dist = "from_distribution"


def main(
    strategy: Strategy,
    distirbution: Distribution,
    num_simulations: int,
    num_bandits: int,
    print_metrics: bool,
    print_plots: bool,
    epsilon: float,
    decay_rate: float | None,
    bandit_gen_method: BanditGenMethod,
    parameter_one_values: list[float] | None,
    parameter_two_values: list[float] | None,
    parameter_one_mean: float | None,
    parameter_one_std: float | None,
    parameter_two_mean: float | None,
    parameter_two_std: float | None

) -> None:
    if bandit_gen_method == BanditGenMethod.from_list:
        bandit_collection = BanditCollection.from_parameter_list(
            distribution=distirbution,
            parameter_one_values=parameter_one_values,
            parameter_two_values=parameter_two_values,
        )
    else:
        bandit_collection = BanditCollection.from_parameter_distribution(
            distribution=distirbution,
            num_bandits=num_bandits,
            parameter_one_mean=parameter_one_mean,
            parameter_two_mean=parameter_two_mean,
            parameter_one_std=parameter_one_std,
            parameter_two_std=parameter_two_std
        )

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


def _validate_args(
    num_bandits: int,
    strategy: Strategy,
    distribution: Distribution,
    decay_rate: float,
    bandit_gen_method: BanditGenMethod,
    parameter_one_values: list[float] | None,
    parameter_two_values: list[float] | None,
    parameter_one_mean: float | None,
    parameter_one_std: float | None,
    parameter_two_mean: float | None,
    parameter_two_std: float | None
) -> None:
    """
    Validate complex arguments pass to simulate().
    """
    if strategy == Strategy.epsilon_decreasing and decay_rate is None:
        raise ValueError(
            "Arg 'decay_rate' must be passed if using strategy" "'epsilon_first'."
        )

    if (distribution in Distribution.two_parameter_family and (
        bandit_gen_method == BanditGenMethod.from_dist and (
            parameter_two_mean is None or parameter_two_std is None
            ))):  # noqa: E501
        raise ValueError("Either 'parameter_two_values' or "
                         "'parameter_two_mean' and 'paramater_two_std' "
                         "must be passed as args.")


    if num_bandits != len(parameter_one_values):
        raise ValueError(
            "Length of parameter 'parameter_one_values' must be equal to "
            f"parameter 'num_bandits'. Got {len(parameter_one_values)} "
            f"and {num_bandits} respectively."
        )

    if distribution in Distribution.two_parameter_family and num_bandits != len(
        parameter_two_values
    ):
        raise ValueError(
            "Length of parameter 'parameter_two_values' must be equal to "
            f"parameter 'num_bandits'. Got {len(parameter_two_values)} "
            f"and {num_bandits} respectively."
        )

    if bandit_gen_method == BanditGenMethod.from_dist and (
        parameter_one_mean is None or parameter_one_std is None
        ):
        raise ValueError("Either 'parameter_one_values' or "
                         "'parameter_one_mean' and 'paramater_one_std' "
                         "must be passed as args.")


@app.command()
def simulate(
    strategy: Strategy = Strategy.epsilon_greedy,
    distribution: Distribution = Distribution.bernoulli,
    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
    num_bandits: Annotated[int, typer.Option(min=2)] = 10,
    print_metrics: bool = False,
    print_plots: bool = False,
    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
    decay_rate: float = 0.05,
    parameter_one_values: list[float] = [
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        500,
    ],
    parameter_two_values: list[float] = None,
    parameter_one_mean: list[float] = None,
    parameter_one_std: list[float] = None,
    parameter_two_mean: list[float] = None,
    parameter_two_std: list[float] = None,

) -> None:  # noqa: ANN003
    """
    Runs a multi-armed bandit simulation.
    """
    bandit_gen_method = BanditGenMethod.from_list \
                        if parameter_one_values is not None \
                        else BanditGenMethod.from_dist
    _validate_args(
        strategy=strategy,
        distribution=distribution,
        num_bandits=num_bandits,
        decay_rate=decay_rate,
        bandit_gen_method=bandit_gen_method,
        parameter_one_values=parameter_one_values,
        parameter_two_values=parameter_two_values,
        parameter_one_mean=parameter_one_mean,
        parameter_two_mean=parameter_two_mean,
        parameter_one_std=parameter_one_std,
        parameter_two_std=parameter_two_std
    )
    main(
        strategy=strategy,
        distirbution=distribution,
        num_simulations=num_simulations,
        num_bandits=num_bandits,
        print_metrics=print_metrics,
        print_plots=print_plots,
        epsilon=epsilon,
        decay_rate=decay_rate,
        bandit_gen_method=bandit_gen_method,
        parameter_one_values=parameter_one_values,
        parameter_two_values=parameter_two_values,
        parameter_one_mean=parameter_one_mean,
        parameter_two_mean=parameter_two_mean,
        parameter_one_std=parameter_one_std,
        parameter_two_std=parameter_two_std
    )


@app.command()
def simulate_from_json(config: Path) -> None:
    """
    Runs a multi-armed bandit simulation with
    configuration (arguments) provided via a json
    file.

    TODO: Create example json schema.
    """
    if not config.is_file() and config.suffix != ".json":
        raise ValueError("config must be suffixed with '.json'.")
    simulation_args = json.load(config)
    simulate(**simulation_args)


if __name__ == "__main__":
    app()
