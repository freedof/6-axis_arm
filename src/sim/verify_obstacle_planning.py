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
from src.sim.obstacle_scene import ObstaclePlanningCase, base_planning_case, obstacle_cases, write_obstacle_model
from src.sim.planning_cases import solve_planning_case
from src.sim.render_planned_roundtrip_gif import render_case_gif
from src.sim.verify_render_gifs import TEST_FPS, TEST_FRAMES, TEST_HEIGHT, TEST_WIDTH, validate_gif


def path_edges_are_valid(checker: MujocoCollisionChecker, path: tuple[np.ndarray, ...], resolution: float) -> bool:
    return all(
        checker.is_edge_valid(q_from, q_to, resolution=resolution)
        for q_from, q_to in zip(path[:-1], path[1:])
    )


def trajectory_states_are_valid(checker: MujocoCollisionChecker, path: tuple[np.ndarray, ...]) -> bool:
    trajectory = parameterize_joint_path(make_roundtrip_path(path), max_joint_velocity=0.8, max_joint_acceleration=1.6)
    if not trajectory_limits_are_satisfied(trajectory):
        return False
    sample_count = max(2, int(np.ceil(trajectory.duration / 0.02)))
    for time_s in np.linspace(0.0, trajectory.duration, sample_count):
        q, _, _ = trajectory.sample(float(time_s))
        if not checker.is_state_valid(q):
            return False
    return True


def trajectory_singularity_is_valid(singularity: SingularityChecker, path: tuple[np.ndarray, ...]) -> bool:
    trajectory = parameterize_joint_path(make_roundtrip_path(path), max_joint_velocity=0.8, max_joint_acceleration=1.6)
    sample_count = max(2, int(np.ceil(trajectory.duration / 0.02)))
    for time_s in np.linspace(0.0, trajectory.duration, sample_count):
        q, _, _ = trajectory.sample(float(time_s))
        if not singularity.is_state_valid(q):
            return False
    return True


def verify_obstacle_case(obstacle_case: ObstaclePlanningCase) -> None:
    robot = dobot_cr5_simplified()
    case = base_planning_case(obstacle_case)
    solved = solve_planning_case(case)
    model_path = write_obstacle_model(obstacle_case=obstacle_case)
    checker = MujocoCollisionChecker(model_path, robot)
    singularity = SingularityChecker(robot)
    state_valid = lambda q: checker.is_state_valid(q) and singularity.is_state_valid(q)

    print(f"case: {obstacle_case.case_id} {obstacle_case.name}")
    print(f"base_case: {case.case_id} {case.name}")
    for label, q in (("A", solved.a.q), ("B", solved.b.q)):
        result = checker.check(q)
        singularity_metrics = singularity.metrics(q)
        print(
            f"{label}:",
            f"valid={result.valid}",
            f"contacts={len(result.contacts)}",
            f"jac_cond={singularity_metrics.condition_number:.3f}",
            f"jac_smin={singularity_metrics.min_singular_value:.6f}",
        )
        if not result.valid:
            raise RuntimeError(f"Obstacle endpoint {label} is invalid: {result.contacts}")
        if not singularity.is_state_valid(q):
            raise RuntimeError(f"Obstacle endpoint {label} is too close to a singularity.")

    direct_valid = checker.is_edge_valid(solved.a.q, solved.b.q, resolution=0.02)
    print(f"direct_path_valid={direct_valid}")
    if obstacle_case.expect_direct_collision and direct_valid:
        raise RuntimeError("Expected the direct joint-space path to collide with the obstacle.")

    config = RRTConnectConfig(
        max_iterations=8000,
        step_size=0.14,
        edge_resolution=0.015,
        goal_sample_rate=0.12,
        rng_seed=77 + int(obstacle_case.case_id.rsplit("-", maxsplit=1)[-1]),
        try_direct=False,
    )
    result = plan_joint_rrt_connect(robot, solved.a.q, solved.b.q, state_valid, config=config)
    print(
        "rrt_connect:",
        f"success={result.success}",
        f"iterations={result.iterations}",
        f"waypoints={len(result.path)}",
        f"length={path_length(result.path):.6f}",
        f"reason={result.reason}",
    )
    if not result.success:
        raise RuntimeError(f"RRT-Connect failed in obstacle scene: {result.reason}")
    if not path_edges_are_valid(checker, result.path, resolution=0.015):
        raise RuntimeError("RRT-Connect returned an invalid obstacle-avoidance path edge.")
    singularity_summary = singularity.path_summary(result.path, resolution=0.015)
    print(
        "singularity:",
        f"max_cond={singularity_summary.max_condition_number:.3f}",
        f"min_smin={singularity_summary.min_singular_value:.6f}",
        f"min_manip={singularity_summary.min_manipulability:.6f}",
        f"valid={singularity_summary.valid}",
    )
    if not singularity_summary.valid:
        raise RuntimeError("RRT-Connect obstacle path is too close to a singularity.")
    if not trajectory_states_are_valid(checker, result.path):
        raise RuntimeError("Time-parameterized obstacle trajectory is invalid.")
    if not trajectory_singularity_is_valid(singularity, result.path):
        raise RuntimeError("Time-parameterized obstacle trajectory is too close to a singularity.")

    shortened = shortcut_path(result.path, state_valid, attempts=80, edge_resolution=0.015, rng_seed=91)
    print(
        "shortcut:",
        f"waypoints={len(shortened)}",
        f"length={path_length(shortened):.6f}",
    )
    if not path_edges_are_valid(checker, shortened, resolution=0.015):
        raise RuntimeError("Shortcut returned an invalid obstacle-avoidance path edge.")
    if not singularity.path_summary(shortened, resolution=0.015).valid:
        raise RuntimeError("Shortcut obstacle path is too close to a singularity.")

    rendered = render_case_gif(
        model_path,
        obstacle_case.test_gif_output,
        case,
        TEST_WIDTH,
        TEST_HEIGHT,
        TEST_FRAMES,
        TEST_FPS,
        shortcut=True,
    )
    if not path_edges_are_valid(checker, rendered.path, resolution=0.015):
        raise RuntimeError("Rendered obstacle GIF path contains an invalid edge.")
    if not trajectory_states_are_valid(checker, rendered.path):
        raise RuntimeError("Rendered obstacle GIF trajectory violates limits or collisions.")
    if not trajectory_singularity_is_valid(singularity, rendered.path):
        raise RuntimeError("Rendered obstacle GIF trajectory is too close to a singularity.")
    validate_gif(obstacle_case.test_gif_output, expected_frames=TEST_FRAMES, expected_size=(TEST_WIDTH, TEST_HEIGHT))


def main() -> None:
    for obstacle_case in obstacle_cases():
        verify_obstacle_case(obstacle_case)

    print("status: OK")


if __name__ == "__main__":
    main()
