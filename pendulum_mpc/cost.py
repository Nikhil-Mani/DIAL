"""Configurable swing-up objective; all components are nonnegative costs."""

from dataclasses import dataclass
from math import atan2, cos, isfinite, pi, sin
from typing import Sequence

from .dynamics import Pendulum, State


@dataclass(frozen=True)
class CostBreakdown:
    energy: float
    effort: float
    terminal_angle: float
    terminal_velocity: float
    constraints: float

    @property
    def total(self) -> float:
        return (self.energy + self.effort + self.terminal_angle
                + self.terminal_velocity + self.constraints)


@dataclass(frozen=True)
class SwingUpCost:
    energy_weight: float = 0.01
    effort_weight: float = 0.01
    terminal_angle_weight: float = 10.0
    terminal_velocity_weight: float = 1.0
    constraint_weight: float = 100.0
    torque_limit: float | None = 2.0
    speed_limit: float | None = None

    def __post_init__(self) -> None:
        for name in ("energy_weight", "effort_weight", "terminal_angle_weight",
                     "terminal_velocity_weight", "constraint_weight"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("torque_limit", "speed_limit"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive, or None")

    def evaluate(
        self, model: Pendulum, states: Sequence[State], torques: Sequence[float]
    ) -> CostBreakdown:
        """Score H+1 states and H torques, without changing either sequence.

        Running terms use each post-control state and are multiplied by dt.
        Endpoint terms are applied once. Constraints are soft squared-excess
        penalties, not enforcement; the controller must enforce torque bounds.
        """
        if len(states) != len(torques) + 1:
            raise ValueError("expected H+1 states for H torques")
        if not all(isfinite(v) for state in states for v in state):
            raise ValueError("states must be finite")
        if not all(isfinite(torque) for torque in torques):
            raise ValueError("torques must be finite")

        target_energy = 2 * model.mass * model.gravity * model.length
        energy = self.energy_weight * model.dt * sum(
            (model.energy(state) - target_energy)**2 for state in states[1:]
        )
        effort = self.effort_weight * model.dt * sum(torque**2 for torque in torques)
        torque_excess = 0.0 if self.torque_limit is None else sum(
            max(0.0, abs(torque) - self.torque_limit)**2 for torque in torques
        )
        speed_excess = 0.0 if self.speed_limit is None else sum(
            max(0.0, abs(state[1]) - self.speed_limit)**2 for state in states[1:]
        )
        theta, omega = states[-1]
        angle_error = atan2(sin(theta - pi), cos(theta - pi))
        return CostBreakdown(
            energy=energy,
            effort=effort,
            terminal_angle=self.terminal_angle_weight * angle_error**2,
            terminal_velocity=self.terminal_velocity_weight * omega**2,
            constraints=self.constraint_weight * model.dt * (torque_excess + speed_excess),
        )
