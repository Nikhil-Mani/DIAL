"""Equal-rollout-budget open-loop comparison, not an MPC performance test."""

import argparse
import csv
from pathlib import Path
from statistics import median

from .cost import SwingUpCost
from .dynamics import Pendulum
from .sampling import Plan, search
from .tempering import CalibrationRule, cold_hot_round_trips, tempered_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatures", type=float, nargs=3, default=(1.0, 3.0, 10.0))
    parser.add_argument("--calibration-evaluations", type=int, default=200,
                        help="maximum setup rollouts, separate from test budget")
    parser.add_argument("--calibration-min", type=int, default=120)
    parser.add_argument("--calibration-window", type=int, default=60)
    parser.add_argument("--calibration-tolerance", type=float, default=0.10)
    parser.add_argument("--fixed-calibration", action="store_true",
                        help="disable diagnostic early stopping, for reproducing old runs")
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("experiments/sampling.csv"))
    args = parser.parse_args()
    if args.seeds < 1 or args.horizon < 1:
        parser.error("seeds and horizon must be positive")
    model, cost = Pendulum(), SwingUpCost()
    initial_plan = (0.0,) * args.horizon
    rule = None if args.fixed_calibration else CalibrationRule(
        min_evaluations=args.calibration_min, window=args.calibration_window,
        tolerance=args.calibration_tolerance)

    def objective(plan: Plan) -> float:
        return cost.evaluate(model, model.rollout((0.0, 0.0), plan), plan).total

    rows = []
    temperatures = (args.temperatures[0], args.temperatures[1], args.temperatures[2])
    # Test every fixed temperature, not only cold MH versus the switching chain.
    methods = [("gaussian", args.temperature)]
    methods += [(f"mh_T{t:g}", t) for t in temperatures]
    methods += [("tempered", temperatures[0])]
    for method, temperature in methods:
        scores = []
        frozen_visits = [0, 0, 0]
        pilot_counts = []
        pilot_statuses = []
        for seed in range(args.seeds):
            tempering = None
            if method == "tempered":
                tempering = tempered_search(initial_plan, objective,
                    evaluations=args.evaluations, sigma=args.sigma, seed=seed,
                    temperatures=temperatures, calibration_evaluations=args.calibration_evaluations,
                    learning_rate=args.learning_rate, calibration_rule=rule)
                result = tempering.search
                pilot_counts.append(tempering.calibration_evaluations)
                pilot_statuses.append(tempering.calibration_stop_reason)
                frozen_visits = [a+b for a, b in zip(frozen_visits, tempering.frozen_visits)]
            else:
                result = search(initial_plan, objective,
                    method="gaussian" if method == "gaussian" else "mh",
                    evaluations=args.evaluations, sigma=args.sigma,
                    temperature=temperature, seed=seed)
            scores.append(result.best_cost)
            for evaluation, (best, current) in enumerate(zip(
                result.best_cost_history, result.current_cost_history
            ), start=1):
                index = evaluation - 1
                level = tempering.level_history[index] if tempering else -1
                weights = tempering.log_weight_history[index] if tempering else (0.0, 0.0, 0.0)
                rows.append(dict(method=method, seed=seed, evaluation=evaluation,
                                 best_cost=best, current_cost=current,
                                 horizon=args.horizon, dt=model.dt, sigma=args.sigma,
                                 temperature=temperatures[level] if tempering else temperature,
                                 accepted_moves=result.accepted_moves,
                                 phase="frozen" if tempering else "fixed",
                                 level=level, g0=weights[0], g1=weights[1], g2=weights[2],
                                 temperature_moves=tempering.temperature_moves if tempering else 0,
                                 calibration_evaluations=tempering.calibration_evaluations if tempering else 0,
                                 calibration_status=tempering.calibration_stop_reason if tempering else "none",
                                 calibration_round_trips=cold_hot_round_trips(
                                     tempering.calibration_level_history) if tempering else 0,
                                 calibration_cap=args.calibration_evaluations if tempering else 0,
                                 calibration_min=rule.min_evaluations if tempering and rule else 0,
                                 calibration_window=rule.window if tempering and rule else 0,
                                 calibration_tolerance=rule.tolerance if tempering and rule else 0,
                                 learning_rate=args.learning_rate if tempering else 0,
                                 total_evaluations=(args.evaluations + tempering.calibration_evaluations)
                                    if tempering else args.evaluations,
                                 T0=temperatures[0], T1=temperatures[1], T2=temperatures[2]))
        print(f"{method}: median best cost {median(scores):.3f}; "
              f"range [{min(scores):.3f}, {max(scores):.3f}]")
        if method == "tempered":
            total = sum(frozen_visits)
            print("  Frozen temperature visits: " + ", ".join(
                f"T={t:g}: {n/total:.1%}" for t, n in zip(temperatures, frozen_visits)))
            print(f"  Calibration extra rollouts by seed: {pilot_counts}; "
                  f"test: {args.evaluations} rollouts per seed")
            print(f"  Calibration diagnostics passed: "
                  f"{pilot_statuses.count('diagnostics_passed')}/{args.seeds}; "
                  f"cap reached without passing: {pilot_statuses.count('max_reached')}/{args.seeds}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}; initial cost {rows[0]['best_cost']:.3f}")


if __name__ == "__main__":
    main()
