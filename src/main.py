from typing import Annotated
from enum import Enum

import typer # noqa: F401

from src.bandit import BernoulliBandit, BanditCollection # noqa: F401
from src.metrics import Metrics # noqa: F401
from src.simulation import (EpsilonFirstStrategy, # noqa: F401
                            EpsilonDecreasingStrategy, # noqa: F401
                            EpsilonGreedyStrategy) # noqa: F401

app = typer.Typer()

class Distribution(str, Enum):
    bernoulli = "bernoulli"
    gaussian = "gaussian"


class Strategy(str, Enum):
    epsilon_first = "epsilon_first"
    epsilon_decreasing = "epsilon_decreasing"
    epsilon_greedy = "epsilon_greedy"


@app.command()
def simulate(strategy: Strategy = Strategy.epsilon_greedy.value,
             distirbution: Distribution = Distribution.bernoulli.value,
             num_simulations: Annotated[int, typer.Option(min=10)] = 500,
             num_bandits: Annotated[int, typer.Option(min=2)] = 10,
             print_metrics: bool = False,
             print_plots: bool = False,
             **kwargs) -> None: # noqa: ANN003
    """
    Run a multi-armed bandit simulation.
    """
    pass  #noqa: PIE790


if __name__ == "__main__":
    typer.run(simulate)
