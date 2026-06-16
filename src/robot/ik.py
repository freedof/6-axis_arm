from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import forward_kinematics, geometric_jacobian, pose_error
from .model import SerialRobotModel


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray
    iterations: int
    position_error: float
    rotation_error: float


def solve_ik_dls(
    robot: SerialRobotModel,
    target_pose: np.ndarray,
    initial_q: np.ndarray | None = None,
    *,
    max_iterations: int = 200,
    position_tolerance: float = 1e-4,
    rotation_tolerance: float = 1e-3,
    damping: float = 1e-2,
    step_scale: float = 0.7,
) -> IKResult:
    q = np.zeros(robot.dof, dtype=float) if initial_q is None else np.asarray(initial_q, dtype=float).copy()
    q = robot.clamp(q)

    pos_norm = np.inf
    rot_norm = np.inf

    for iteration in range(1, max_iterations + 1):
        fk = forward_kinematics(robot, q)
        err = pose_error(fk.pose, target_pose)
        pos_norm = float(np.linalg.norm(err[:3]))
        rot_norm = float(np.linalg.norm(err[3:]))

        if pos_norm <= position_tolerance and rot_norm <= rotation_tolerance:
            return IKResult(True, q, iteration, pos_norm, rot_norm)

        jac = geometric_jacobian(robot, q)
        lhs = jac @ jac.T + (damping**2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(lhs, err)
        q = robot.clamp(q + step_scale * dq)

    return IKResult(False, q, max_iterations, pos_norm, rot_norm)


def solve_ik_multi_start(
    robot: SerialRobotModel,
    target_pose: np.ndarray,
    seeds: list[np.ndarray] | None = None,
    *,
    random_starts: int = 16,
    rng_seed: int = 7,
    **kwargs,
) -> IKResult:
    rng = np.random.default_rng(rng_seed)
    all_seeds: list[np.ndarray] = []
    if seeds:
        all_seeds.extend(np.asarray(seed, dtype=float) for seed in seeds)
    all_seeds.append(np.zeros(robot.dof, dtype=float))

    lower = robot.lower_limits
    upper = robot.upper_limits
    all_seeds.extend(rng.uniform(lower, upper) for _ in range(random_starts))

    best: IKResult | None = None
    for seed in all_seeds:
        result = solve_ik_dls(robot, target_pose, seed, **kwargs)
        if result.success:
            return result
        if best is None or (result.position_error + result.rotation_error) < (best.position_error + best.rotation_error):
            best = result

    if best is None:
        raise RuntimeError("IK solver did not run any seed.")
    return best


def solve_position_ik_dls(
    robot: SerialRobotModel,
    target_xyz: np.ndarray,
    initial_q: np.ndarray | None = None,
    *,
    max_iterations: int = 200,
    position_tolerance: float = 1e-4,
    damping: float = 1e-2,
    step_scale: float = 0.7,
) -> IKResult:
    q = np.zeros(robot.dof, dtype=float) if initial_q is None else np.asarray(initial_q, dtype=float).copy()
    q = robot.clamp(q)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if target_xyz.shape != (3,):
        raise ValueError(f"Expected target_xyz shape (3,), got {target_xyz.shape}")

    pos_norm = np.inf
    for iteration in range(1, max_iterations + 1):
        fk = forward_kinematics(robot, q)
        err = target_xyz - fk.position
        pos_norm = float(np.linalg.norm(err))
        if pos_norm <= position_tolerance:
            return IKResult(True, q, iteration, pos_norm, 0.0)

        jac = geometric_jacobian(robot, q)[:3, :]
        lhs = jac @ jac.T + (damping**2) * np.eye(3)
        dq = jac.T @ np.linalg.solve(lhs, err)
        q = robot.clamp(q + step_scale * dq)

    return IKResult(False, q, max_iterations, pos_norm, 0.0)


def solve_position_ik_multi_start(
    robot: SerialRobotModel,
    target_xyz: np.ndarray,
    seeds: list[np.ndarray] | None = None,
    *,
    random_starts: int = 24,
    rng_seed: int = 11,
    **kwargs,
) -> IKResult:
    rng = np.random.default_rng(rng_seed)
    all_seeds: list[np.ndarray] = []
    if seeds:
        all_seeds.extend(np.asarray(seed, dtype=float) for seed in seeds)
    all_seeds.extend(
        [
            np.zeros(robot.dof, dtype=float),
            np.array([0.0, -0.6, 0.85, 0.0, -0.8, 0.0], dtype=float),
        ]
    )

    lower = robot.lower_limits
    upper = robot.upper_limits
    all_seeds.extend(rng.uniform(lower, upper) for _ in range(random_starts))

    best: IKResult | None = None
    for seed in all_seeds:
        result = solve_position_ik_dls(robot, target_xyz, seed, **kwargs)
        if result.success:
            return result
        if best is None or result.position_error < best.position_error:
            best = result

    if best is None:
        raise RuntimeError("Position IK solver did not run any seed.")
    return best
