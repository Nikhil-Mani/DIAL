"""Gaussian proposals and fixed-temperature MH; no controller or tempering yet."""

from dataclasses import dataclass
from math import exp, isfinite
from random import Random
from typing import Callable, Literal, Sequence

Plan = tuple[float, ...]


def gaussian_proposal(center: Sequence[float], sigma: float, rng: Random) -> Plan:
    """Add independent zero-mean Gaussian noise to every planned torque."""
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not center or not all(isfinite(u) for u in center):
        raise ValueError("center must be nonempty and finite")
    return tuple(u + rng.gauss(0.0, sigma) for u in center)


def acceptance_probability(current_cost: float, proposed_cost: float, temperature: float) -> float:
    """Metropolis rule for a symmetric proposal and target exp(-cost/T)."""
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isfinite(current_cost) or not isfinite(proposed_cost):
        raise ValueError("costs must be finite")
    if proposed_cost <= current_cost:
        return 1.0
    return exp(-(proposed_cost - current_cost) / temperature)


@dataclass(frozen=True)
class SearchResult:
    best_plan: Plan
    best_cost: float
    final_plan: Plan
    final_cost: float
    best_cost_history: tuple[float, ...]
    current_cost_history: tuple[float, ...]
    accepted_moves: int
    evaluations: int


def search(
    initial_plan: Sequence[float], objective: Callable[[Plan], float], *,
    method: Literal["gaussian", "mh"], evaluations: int = 200,
    sigma: float = 0.5, temperature: float = 1.0, seed: int = 0,
) -> SearchResult:
    """Compare independent draws about a fixed plan with an MH random walk.

    Budget includes the initial evaluation. MH proposals center on the current
    chain state; Gaussian baseline proposals always center on the initial plan.
    Rejected MH moves are recorded as repeated states. Best-so-far is reported
    separately from the chain state, which may intentionally move uphill.
    Torques are not clipped: clipping would invalidate symmetric Gaussian MH.
    The supplied objective currently uses soft constraint penalties.
    """
    if method not in ("gaussian", "mh"):
        raise ValueError("method must be gaussian or mh")
    if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations < 1:
        raise ValueError("evaluations must be a positive integer")
    rng = Random(seed)
    # Validate sampling parameters even if the budget only includes initialization.
    initial = tuple(initial_plan)
    gaussian_proposal(initial, sigma, Random(0))
    acceptance_probability(0.0, 0.0, temperature)

    def evaluate(plan: Plan) -> float:
        value = objective(plan)
        if not isfinite(value):
            raise ValueError("objective must return a finite cost")
        return value

    current = best = initial
    current_cost = best_cost = evaluate(initial)
    best_history = [best_cost]
    current_history = [current_cost]
    accepted = 0
    for _ in range(evaluations - 1):
        center = current if method == "mh" else initial
        proposal = gaussian_proposal(center, sigma, rng)
        proposal_cost = evaluate(proposal)
        if proposal_cost < best_cost:
            best, best_cost = proposal, proposal_cost
        if method == "gaussian" or rng.random() < acceptance_probability(
            current_cost, proposal_cost, temperature
        ):
            current, current_cost = proposal, proposal_cost
            accepted += 1
        best_history.append(best_cost)
        current_history.append(current_cost)
    return SearchResult(best, best_cost, current, current_cost,
                        tuple(best_history), tuple(current_history), accepted, evaluations)
