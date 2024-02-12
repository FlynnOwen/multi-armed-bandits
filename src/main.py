from enum import Enum
from typing import Annotated

import typer

from src.bandit import BanditCollection, BernoulliBandit  # noqa: F401
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


def distribution_factory(distribution: Distribution, **kwargs) -> SemiUniformStrategy:  # noqa: ANN003
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


@app.command()
def simulate(
    strategy: Strategy = Strategy.epsilon_greedy.value,
    distirbution: Distribution = Distribution.bernoulli.value,
    num_simulations: Annotated[int, typer.Option(min=10)] = 500,
    num_bandits: Annotated[int, typer.Option(min=2)] = 10,
    print_metrics: bool = False,
    print_plots: bool = False,
    **kwargs,
) -> None:  # noqa: ANN003
    """
    Run a multi-armed bandit simulation.
    """
    pass  # noqa: PIE790


if __name__ == "__main__":
    typer.run(simulate)
