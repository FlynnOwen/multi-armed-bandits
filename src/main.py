import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from src.bandit import Bandit, BanditCollection, BernoulliBandit  # noqa: F401
from src.metrics import Metrics  # noqa: F401
from src.simulation import (
    EpsilonDecreasingStrategy,
    EpsilonFirstStrategy,
    EpsilonGreedyStrategy,
    SemiUniformStrategy,
)

app = typer.Typer()


class Distribution(str, Enum):
    bernoulli = "bernoulli"
    gaussian = "gaussian"


def distribution_factory(distribution: Distribution, **kwargs) -> Bandit:  # noqa: ANN003
    distribution_map = {
        Distribution.bernoulli: BernoulliBandit,
    }

    return distribution_map[distribution](**kwargs)


class Strategy(str, Enum):
    epsilon_first = "epsilon_first"
    epsilon_decreasing = "epsilon_decreasing"
    epsilon_greedy = "epsilon_greedy"


def strategy_factory(strategy: Strategy, **kwargs) -> SemiUniformStrategy:  # noqa: ANN003
    strategy_map = {
        Strategy.epsilon_first: EpsilonFirstStrategy,
        Strategy.epsilon_decreasing: EpsilonDecreasingStrategy,
        Strategy.epsilon_greedy: EpsilonGreedyStrategy,
    }

    return strategy_map[strategy](**kwargs)


def main(
    strategy: Strategy,
    distirbution: Distribution,
    num_simulations: int,
    num_bandits: int,
    print_metrics: bool,
    print_plots: bool,
) -> None:
    sel_distribution = distribution_factory(distribution=distirbution)
    sel_strategy = strategy_factory(strategy=strategy)


@app.command()
def simulate(
    strategy: Strategy = Strategy.epsilon_greedy.value,
    distirbution: Distribution = Distribution.bernoulli.value,
    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
    num_bandits: Annotated[int, typer.Option(min=2)] = 10,
    print_metrics: bool = False,
    print_plots: bool = False,
    parameter_one_mean: float = 0.5,
    parameter_two_mean: float = None,
    epsilon: float = 0.2,
    decay_rate: float = 0.05,
    parameter_one_values: list[float] = None,
    parameter_two_values: list[float] = None,
) -> None:  # noqa: ANN003
    """
    Runs a multi-armed bandit simulation.
    """
    pass  # noqa: PIE790


@app.command()
def simulate_from_json(config: Path) -> None:
    """
    Runs a multi-armed bandit simulation with
    configuration (arguments) provided via a json
    file.

    TODO: Create example json schema.
    """
    if not config.is_file() and config.suffix != ".json":
        raise ValueError("config must be a file with .json suffix")
    simulation_args = json.load(config)
    simulate(**simulation_args)


if __name__ == "__main__":
    app()
