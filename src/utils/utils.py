from math import log, sqrt
from enum import StrEnum


def ucb(q: float, t: int, c: float, q_t: int) -> float:
    """
    Calculates the upper confidence bound.

    Where UCB = Argmax(a) Q_t(a) + c(sqrt((ln(t)/N_t(a))))

    Note that:
    q: Average reward for the best bandit.
    t: Number of total bandit pulls.
    c: A constant that balances exploration vs exploitation.
    q_t: The number of times the best bandit has been pulled.
    """
    return q + (c * sqrt(log(t) / q_t))


class ExtStrEnum(StrEnum):

    @classmethod
    def values(cls) -> list[str]:
        return {c.value for c in cls}
