from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.collision import MujocoCollisionChecker
from src.planning.rrt_connect import RRTConnectConfig, path_length, plan_joint_rrt_connect, shortcut_path
from src.planning.singularity import SingularityChecker
from src.planning.trajectory import make_roundtrip_path, parameterize_joint_path, trajectory_limits_are_satisfied
from src.robot.model import dobot_cr5_simplified
from src.sim.planning_cases import planning_cases, solve_planning_case
from src.sim.planning_model import write_planning_model


MAX_JOINT_VELOCITY = 0.8
MAX_JOINT_ACCELERATION = 1.6


def sampled_trajectory_is_valid(
    checker: MujocoCollisionChecker,
    singularity: SingularityChecker,
    path: tuple[np.ndarray, ...],
    *,
    sample_dt: float = 0.02,
) -> tuple[bool, float, int]:
    trajectory = parameterize_joint_path(
        make_roundtrip_path(path),
        max_joint_velocity=MAX_JOINT_VELOCITY,
        max_joint_acceleration=MAX_JOINT_ACCELERATION,
    )
    if not trajectory_limits_are_satisfied(trajectory):
        return False, trajectory.duration, 0

    samples = max(2, int(np.ceil(trajectory.duration / sample_dt)) + 1)
    for time_s in np.linspace(0.0, trajectory.duration, samples):
        q, _, _ = trajectory.sample(float(time_s))
        if not checker.is_state_valid(q):
            return False, trajectory.duration, samples
        if not singularity.is_state_valid(q):
            return False, trajectory.duration, samples
    return True, trajectory.duration, samples


def main() -> None:
    robot = dobot_cr5_simplified()
    checker = MujocoCollisionChecker(write_planning_model(), robot)
    singularity = SingularityChecker(robot)
    state_valid = lambda q: checker.is_state_valid(q) and singularity.is_state_valid(q)
    config = RRTConnectConfig(
        max_iterations=8000,
        step_size=0.16,
        edge_resolution=0.02,
        rng_seed=47,
        try_direct=False,
    )

    for case in planning_cases():
        solved = solve_planning_case(case)
        result = plan_joint_rrt_connect(robot, solved.a.q, solved.b.q, state_valid, config=config)
        if not result.success:
            raise RuntimeError(f"{case.case_id} RRT-Connect failed: {result.reason}")

        shortened = shortcut_path(result.path, state_valid, attempts=50, edge_resolution=0.02)
        for label, path in (("raw", result.path), ("shortcut", shortened)):
            valid, duration, samples = sampled_trajectory_is_valid(checker, singularity, path)
            singularity_summary = singularity.path_summary(path, resolution=0.02)
            print(
                f"{case.case_id} {label}:",
                f"waypoints={len(path)}",
                f"length={path_length(path):.6f}",
                f"duration={duration:.3f}s",
                f"samples={samples}",
                f"max_cond={singularity_summary.max_condition_number:.3f}",
                f"min_smin={singularity_summary.min_singular_value:.6f}",
                f"valid={valid}",
            )
            if not valid:
                raise RuntimeError(f"{case.case_id} {label} trajectory is invalid.")

    print("status: OK")


if __name__ == "__main__":
    main()
