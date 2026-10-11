"""Point-mass pendulum: zero angle is down, pi radians is upright."""

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Iterable

State = tuple[float, float]


@dataclass(frozen=True)
class Pendulum:
    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    damping: float = 0.0
    dt: float = 0.02

    def __post_init__(self) -> None:
        for name in ("mass", "length", "gravity", "damping", "dt"):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name in ("mass", "length", "dt") and value == 0):
                raise ValueError(f"{name} has an invalid value")

    def derivative(self, state: State, torque: float) -> State:
        """Return angular velocity and acceleration in SI units."""
        theta, omega = state
        if not all(isfinite(value) for value in (theta, omega, torque)):
            raise ValueError("state and torque must be finite")
        acceleration = (
            torque - self.damping * omega
            - self.mass * self.gravity * self.length * sin(theta)
        ) / (self.mass * self.length**2)
        return omega, acceleration

    def step(self, state: State, torque: float) -> State:
        """Advance one dt using RK4 and constant torque throughout the step."""
        def offset(slope: State, scale: float) -> State:
            return state[0] + scale * slope[0], state[1] + scale * slope[1]

        k1 = self.derivative(state, torque)
        k2 = self.derivative(offset(k1, self.dt / 2), torque)
        k3 = self.derivative(offset(k2, self.dt / 2), torque)
        k4 = self.derivative(offset(k3, self.dt), torque)
        return (
            state[0] + self.dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6,
            state[1] + self.dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6,
        )

    def rollout(self, initial_state: State, torques: Iterable[float]) -> list[State]:
        """Return H+1 states including the initial state for H controls."""
        self.derivative(initial_state, 0.0)  # Validate even an empty rollout.
        states: list[State] = [(initial_state[0], initial_state[1])]
        for torque in torques:
            states.append(self.step(states[-1], torque))
        return states

    def energy(self, state: State) -> float:
        """Mechanical energy, with potential energy zero at the bottom."""
        theta, omega = state
        return (
            0.5 * self.mass * self.length**2 * omega**2
            + self.mass * self.gravity * self.length * (1 - cos(theta))
        )
