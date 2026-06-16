from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTrajectory:
    times: np.ndarray
    positions: tuple[np.ndarray, ...]
    max_joint_velocity: float
    max_joint_acceleration: float

    @property
    def duration(self) -> float:
        if len(self.times) == 0:
            return 0.0
        return float(self.times[-1])

    @property
    def dof(self) -> int:
        if not self.positions:
            return 0
        return int(self.positions[0].shape[0])

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.positions:
            raise ValueError("Cannot sample an empty trajectory.")
        if len(self.positions) == 1:
            zeros = np.zeros_like(self.positions[0])
            return self.positions[0].copy(), zeros, zeros

        t = float(np.clip(time_s, 0.0, self.duration))
        index = int(np.searchsorted(self.times, t, side="right") - 1)
        index = min(max(index, 0), len(self.positions) - 2)
        t0 = float(self.times[index])
        t1 = float(self.times[index + 1])
        dt = t1 - t0
        if dt <= 1e-12:
            zeros = np.zeros_like(self.positions[index])
            return self.positions[index].copy(), zeros, zeros

        u = (t - t0) / dt
        q0 = self.positions[index]
        dq = self.positions[index + 1] - q0
        s = 3.0 * u**2 - 2.0 * u**3
        ds_dt = (6.0 * u * (1.0 - u)) / dt
        d2s_dt2 = (6.0 - 12.0 * u) / (dt**2)
        return q0 + s * dq, ds_dt * dq, d2s_dt2 * dq


def parameterize_joint_path(
    path: tuple[np.ndarray, ...],
    *,
    max_joint_velocity: float = 0.8,
    max_joint_acceleration: float = 1.6,
    min_segment_duration: float = 0.12,
    duplicate_tolerance: float = 1e-9,
) -> JointTrajectory:
    if max_joint_velocity <= 0.0:
        raise ValueError("max_joint_velocity must be positive.")
    if max_joint_acceleration <= 0.0:
        raise ValueError("max_joint_acceleration must be positive.")
    if min_segment_duration <= 0.0:
        raise ValueError("min_segment_duration must be positive.")

    waypoints = remove_duplicate_waypoints(path, tolerance=duplicate_tolerance)
    if not waypoints:
        raise ValueError("Path must contain at least one waypoint.")
    if len(waypoints) == 1:
        return JointTrajectory(
            times=np.array([0.0], dtype=float),
            positions=tuple(waypoints),
            max_joint_velocity=float(max_joint_velocity),
            max_joint_acceleration=float(max_joint_acceleration),
        )

    times = [0.0]
    for q_from, q_to in zip(waypoints[:-1], waypoints[1:]):
        delta = np.abs(q_to - q_from)
        velocity_dt = 1.5 * float(np.max(delta / max_joint_velocity))
        acceleration_dt = float(np.sqrt(np.max(6.0 * delta / max_joint_acceleration)))
        duration = max(min_segment_duration, velocity_dt, acceleration_dt)
        times.append(times[-1] + duration)

    return JointTrajectory(
        times=np.array(times, dtype=float),
        positions=tuple(waypoints),
        max_joint_velocity=float(max_joint_velocity),
        max_joint_acceleration=float(max_joint_acceleration),
    )


def make_roundtrip_path(path: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    waypoints = remove_duplicate_waypoints(path)
    if len(waypoints) <= 1:
        return tuple(waypoints)
    return tuple(waypoints + list(reversed(waypoints[:-1])))


def remove_duplicate_waypoints(path: tuple[np.ndarray, ...], *, tolerance: float = 1e-9) -> list[np.ndarray]:
    cleaned: list[np.ndarray] = []
    for waypoint in path:
        q = np.asarray(waypoint, dtype=float).copy()
        if cleaned and np.linalg.norm(q - cleaned[-1]) <= tolerance:
            continue
        cleaned.append(q)
    return cleaned


def trajectory_limits_are_satisfied(
    trajectory: JointTrajectory,
    *,
    velocity_tolerance: float = 1e-9,
    acceleration_tolerance: float = 1e-9,
    samples_per_segment: int = 9,
) -> bool:
    if len(trajectory.positions) <= 1:
        return True
    for t0, t1 in zip(trajectory.times[:-1], trajectory.times[1:]):
        for alpha in np.linspace(0.0, 1.0, samples_per_segment):
            _, qd, qdd = trajectory.sample((1.0 - alpha) * t0 + alpha * t1)
            if np.max(np.abs(qd)) > trajectory.max_joint_velocity + velocity_tolerance:
                return False
            if np.max(np.abs(qdd)) > trajectory.max_joint_acceleration + acceleration_tolerance:
                return False
    return True

