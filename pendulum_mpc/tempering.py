"""One-chain simulated tempering with a finite, equal-visit calibration phase."""

from dataclasses import dataclass
from math import exp, isfinite
from random import Random
from typing import Callable, Sequence

from .sampling import Plan, SearchResult, acceptance_probability, gaussian_proposal


def cold_hot_round_trips(levels: Sequence[int]) -> int:
    """Count completed 0 -> 2 -> 0 excursions, ignoring intermediate repeats."""
    phase = trips = 0
    for level in levels:
        if phase == 0 and level == 0:
            phase = 1
        elif phase == 1 and level == 2:
            phase = 2
        elif phase == 2 and level == 0:
            trips += 1
            phase = 1
    return trips


@dataclass(frozen=True)
class CalibrationRule:
    min_evaluations: int = 120
    window: int = 60
    tolerance: float = 0.10
    require_round_trip: bool = True

    def __post_init__(self) -> None:
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in (self.min_evaluations, self.window)):
            raise ValueError("minimum and window must be positive integers")
        if self.window > self.min_evaluations:
            raise ValueError("window cannot exceed minimum evaluations")
        if not isfinite(self.tolerance) or not 0 <= self.tolerance < 1/3:
            raise ValueError("tolerance must be in [0, 1/3)")

    def ready(self, levels: Sequence[int]) -> bool:
        """Operational mixing diagnostic, not a convergence certificate."""
        if len(levels) < self.min_evaluations:
            return False
        recent = levels[-self.window:]
        balanced = all(abs(recent.count(k) / self.window - 1/3) <= self.tolerance
                       for k in range(3))
        return balanced and (not self.require_round_trip or cold_hot_round_trips(levels) > 0)


def temperature_acceptance(
    cost: float, old_temperature: float, new_temperature: float,
    old_log_weight: float, new_log_weight: float,
) -> float:
    """Keep U fixed; use the joint target exp(-J(U)/T + g).

    Requires symmetric temperature proposals. Our chain proposes +/- one
    level with equal probability and stays put at an out-of-range boundary.
    """
    values = (cost, old_temperature, new_temperature, old_log_weight, new_log_weight)
    if not all(isfinite(value) for value in values):
        raise ValueError("temperature transition inputs must be finite")
    if min(old_temperature, new_temperature) <= 0:
        raise ValueError("temperatures must be positive")
    log_ratio = cost * (1 / old_temperature - 1 / new_temperature)
    log_ratio += new_log_weight - old_log_weight
    return exp(min(0.0, log_ratio))


@dataclass(frozen=True)
class TemperingResult:
    search: SearchResult
    temperatures: tuple[float, float, float]
    level_history: tuple[int, ...]
    log_weight_history: tuple[tuple[float, ...], ...]
    calibration_evaluations: int
    calibration_visits: tuple[int, ...]
    frozen_visits: tuple[int, ...]
    temperature_moves: int
    calibration_level_history: tuple[int, ...]
    calibration_log_weight_history: tuple[tuple[float, ...], ...]
    calibration_stop_reason: str


def tempered_search(
    initial_plan: Sequence[float], objective: Callable[[Plan], float], *,
    evaluations: int = 200, calibration_evaluations: int = 50,
    temperatures: tuple[float, float, float] = (1.0, 3.0, 10.0),
    sigma: float = 0.5, learning_rate: float = 1.0, seed: int = 0,
    calibration_rule: CalibrationRule | None = None,
) -> TemperingResult:
    """Adapt toward equal temperature visits, then freeze the learned weights.

    Calibration and testing have separate budgets, each including initialization.
    Testing restarts from the initial plan and cold level with a fresh RNG;
    only learned weights transfer. Pilot candidates do not affect test scores.
    Total objective calls equal calibration_evaluations + evaluations.
    Temperature changes reuse the current cost and require no extra rollouts.

    Calibration is a stochastic-approximation heuristic, not equilibrium
    sampling or a guarantee of balanced visits. Frozen-phase visits diagnose
    whether the learned weights worked. Gaussian sigma stays fixed across T.
    With a rule, calibration_evaluations is a hard maximum; passing diagnostics
    can end the pilot early. Failure at the cap is reported, not called success.
    """
    if any(isinstance(n, bool) or not isinstance(n, int)
           for n in (evaluations, calibration_evaluations)):
        raise ValueError("evaluation budgets must be integers")
    if calibration_evaluations < 1 or evaluations < 1:
        raise ValueError("calibration and test budgets must both be positive")
    if calibration_rule is not None and calibration_evaluations < calibration_rule.min_evaluations:
        raise ValueError("calibration cap cannot be below the diagnostic minimum")
    if len(temperatures) != 3 or not all(isfinite(t) and t > 0 for t in temperatures):
        raise ValueError("need three finite positive temperatures")
    if not temperatures[0] < temperatures[1] < temperatures[2]:
        raise ValueError("temperatures must be strictly increasing")
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    initial = tuple(initial_plan)
    current = best = initial
    gaussian_proposal(current, sigma, Random(0))  # Validate without using chain RNG.
    rng = Random(f"calibration:{seed}")

    def evaluate(plan: Plan) -> float:
        value = objective(plan)
        if not isfinite(value):
            raise ValueError("objective must return a finite cost")
        return value

    current_cost = best_cost = evaluate(current)
    # Equalize temperature probabilities at the initial cost as a starting
    # heuristic. Calibration then learns corrections from visited plans.
    weights = [current_cost * (1 / t - 1 / temperatures[0]) for t in temperatures]
    level = accepted = temperature_moves = 0
    best_history, current_history = [best_cost], [current_cost]
    levels, weight_history = [level], [tuple(weights)]
    calibration_count = calibration_evaluations
    stop_reason = "fixed_budget" if calibration_rule is None else "max_reached"
    for index in range(1, calibration_evaluations + evaluations):
        if index >= calibration_count + evaluations:
            break
        if index == calibration_count:
            # The independent test starts here. Retain weights only; reset the
            # plan, temperature, RNG, counters and best-so-far test result.
            current = best = initial
            current_cost = best_cost = evaluate(initial)
            rng = Random(seed)
            level = accepted = temperature_moves = 0
            best_history.append(best_cost)
            current_history.append(current_cost)
            levels.append(level)
            weight_history.append(tuple(weights))
            continue
        proposal = gaussian_proposal(current, sigma, rng)
        proposed_cost = evaluate(proposal)
        if proposed_cost < best_cost:
            best, best_cost = proposal, proposed_cost
        if rng.random() < acceptance_probability(current_cost, proposed_cost, temperatures[level]):
            current, current_cost = proposal, proposed_cost
            accepted += 1

        next_level = level + rng.choice((-1, 1))
        if 0 <= next_level < 3 and rng.random() < temperature_acceptance(
            current_cost, temperatures[level], temperatures[next_level],
            weights[level], weights[next_level],
        ):
            level = next_level
            temperature_moves += 1
        if index < calibration_count:
            # Decrease the visited level's attractiveness, with a diminishing
            # gain. Subtracting a common offset keeps g[0]=0 without changing p.
            gain = learning_rate / index**0.6
            weights = [g - gain * (float(k == level) - 1 / 3)
                       for k, g in enumerate(weights)]
            offset = weights[0]
            weights = [g - offset for g in weights]
        best_history.append(best_cost)
        current_history.append(current_cost)
        levels.append(level)
        weight_history.append(tuple(weights))
        if index < calibration_count and calibration_rule is not None and calibration_rule.ready(levels):
            calibration_count = index + 1
            stop_reason = "diagnostics_passed"

    start = calibration_count
    result = SearchResult(best, best_cost, current, current_cost,
                          tuple(best_history[start:]), tuple(current_history[start:]), accepted, evaluations)
    return TemperingResult(
        result, temperatures, tuple(levels[start:]), tuple(weight_history[start:]), calibration_count,
        tuple(levels[:start].count(k) for k in range(3)),
        tuple(levels[start:].count(k) for k in range(3)), temperature_moves,
        tuple(levels[:start]), tuple(weight_history[:start]),
        stop_reason,
    )
