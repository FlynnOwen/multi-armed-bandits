import json
from pathlib import Path
from typing import Annotated

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


class BanditGenMethod(ExtStrEnum):
    from_list = "from_list"
    from_dist = "from_distribution"


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
def simulate_one_param_fixed(  # noqa
    strategy: Strategy = Strategy.epsilon_greedy,
    distribution: OneParamDistribution = OneParamDistribution.bernoulli,
    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
    print_metrics: bool = False,
    print_plots: bool = False,
    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
    decay_rate: float = 0.05,
    parameter_one_values: list[float] = None,
):
    """
    Run a simulation from a fixed set of bandit
    parameters.
    """
    if strategy == Strategy.epsilon_decreasing and decay_rate is None:
        raise ValueError(
            "Arg 'decay_rate' must be passed if using strategy" "'epsilon_first'."
        )

    bandit_collection = BanditCollection.from_parameter_list(
        distribution=distribution, parameter_one_values=parameter_one_values
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
def simulate_two_param_fixed(  # noqa
    strategy: Strategy = Strategy.epsilon_greedy,
    distribution: TwoParamDistribution = TwoParamDistribution.gaussian,
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
def simulate_one_param_generate(  # noqa
    parameter_one_mean: float,
    parameter_one_std: float,
    num_bandits: int = typer.Argument(min=2),
    strategy: Strategy = Strategy.epsilon_greedy,
    distribution: OneParamDistribution = OneParamDistribution.bernoulli,
    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
    print_metrics: bool = False,
    print_plots: bool = False,
    epsilon: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
    decay_rate: float = 0.05
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
def simulate_two_param_generate(  # noqa
    parameter_one_mean: float,
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
    decay_rate: float = 0.05
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

    num_parameters = 1 if simulation_args.get("distribution") \
        in OneParamDistribution.values() \
        else 2

    if bandit_gen_method == BanditGenMethod.from_dist.value \
        and num_parameters == 1:
        simulate_one_param_generate(**simulation_args)
    elif bandit_gen_method == BanditGenMethod.from_list.value \
        and num_parameters == 1:
        simulate_one_param_fixed(**simulation_args)
    elif bandit_gen_method == BanditGenMethod.from_dist.value \
        and num_parameters == 2:
        simulate_two_param_generate(**simulation_args)
    elif bandit_gen_method == BanditGenMethod.from_list.value \
        and num_parameters == 2:
        simulate_two_param_fixed(**simulation_args)
    else:
        raise ValueError(
            "Arg 'bandit_gen_method' must be one of" f"{BanditGenMethod.values()}."
        )


@app.command()
def list_distributions() -> list[str]:
    print("One Parameter Distributions: \n"  #noqa
          f"{OneParamDistribution.values()} \n \n"
          "Two Parameter Distributions: \n"
          f"{TwoParamDistribution.values()}")


if __name__ == "__main__":
    app()
