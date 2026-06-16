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
from src.robot.model import dobot_cr5_simplified
from src.sim.planning_model import write_planning_model
from src.sim.planning_cases import SolvedPlanningCase, planning_cases, solve_planning_case


def path_edges_are_valid(checker: MujocoCollisionChecker, path: tuple[np.ndarray, ...]) -> bool:
    return all(
        checker.is_edge_valid(q_from, q_to, resolution=0.02)
        for q_from, q_to in zip(path[:-1], path[1:])
    )


def verify_case(
    solved: SolvedPlanningCase,
    checker: MujocoCollisionChecker,
    singularity: SingularityChecker,
    *,
    rng_seed: int,
) -> None:
    case = solved.case
    print(f"{case.case_id}: {case.name}")
    print(f"purpose: {case.purpose}")

    for solved_target in (solved.a, solved.b):
        label = solved_target.target.name
        q = solved_target.q
        result = checker.check(q)
        singularity_metrics = singularity.metrics(q)
        print(
            f"  {label}:",
            f"valid={result.valid}",
            f"contacts={len(result.contacts)}",
            f"ik_pos_error={solved_target.position_error:.8e}",
            f"tool_z_angle_deg={solved_target.tool_z_angle_deg:.6f}",
            f"jac_cond={singularity_metrics.condition_number:.3f}",
            f"jac_smin={singularity_metrics.min_singular_value:.6f}",
            f"manip={singularity_metrics.manipulability:.6f}",
        )
        if not result.valid:
            raise RuntimeError(f"{case.case_id} endpoint {label} is invalid: {result.contacts}")
        if not singularity.is_state_valid(q):
            raise RuntimeError(f"{case.case_id} endpoint {label} is too close to a singularity.")

    config = RRTConnectConfig(
        max_iterations=8000,
        step_size=0.16,
        edge_resolution=0.02,
        rng_seed=rng_seed,
        try_direct=False,
    )
    result = plan_joint_rrt_connect(
        dobot_cr5_simplified(),
        solved.a.q,
        solved.b.q,
        lambda q: checker.is_state_valid(q) and singularity.is_state_valid(q),
        config=config,
    )
    print(
        "  rrt_connect:",
        f"success={result.success}",
        f"iterations={result.iterations}",
        f"waypoints={len(result.path)}",
        f"length={path_length(result.path):.6f}",
        f"reason={result.reason}",
    )
    if not result.success:
        raise RuntimeError(f"{case.case_id} RRT-Connect failed: {result.reason}")
    if not np.allclose(result.path[0], solved.a.q) or not np.allclose(result.path[-1], solved.b.q):
        raise RuntimeError(f"{case.case_id} RRT-Connect path endpoints do not match the requested endpoints.")
    if not path_edges_are_valid(checker, result.path):
        raise RuntimeError(f"{case.case_id} RRT-Connect returned an invalid path edge.")
    singularity_summary = singularity.path_summary(result.path, resolution=0.02)
    print(
        "  singularity:",
        f"max_cond={singularity_summary.max_condition_number:.3f}",
        f"min_smin={singularity_summary.min_singular_value:.6f}",
        f"min_manip={singularity_summary.min_manipulability:.6f}",
        f"samples={singularity_summary.samples}",
        f"valid={singularity_summary.valid}",
    )
    if not singularity_summary.valid:
        raise RuntimeError(f"{case.case_id} RRT-Connect path is too close to a singularity.")

    state_valid = lambda q: checker.is_state_valid(q) and singularity.is_state_valid(q)
    shortened = shortcut_path(result.path, state_valid, attempts=50, edge_resolution=0.02)
    print(
        "  shortcut:",
        f"waypoints={len(shortened)}",
        f"length={path_length(shortened):.6f}",
    )
    if not path_edges_are_valid(checker, shortened):
        raise RuntimeError(f"{case.case_id} shortcut returned an invalid path edge.")
    shortcut_singularity = singularity.path_summary(shortened, resolution=0.02)
    if not shortcut_singularity.valid:
        raise RuntimeError(f"{case.case_id} shortcut path is too close to a singularity.")


def main() -> None:
    planning_model = write_planning_model()
    robot = dobot_cr5_simplified()
    checker = MujocoCollisionChecker(planning_model, robot)
    singularity = SingularityChecker(robot)
    cases = planning_cases()
    for index, case in enumerate(cases):
        solved = solve_planning_case(case)
        verify_case(solved, checker, singularity, rng_seed=31 + index)

    print(f"cases={len(cases)}")
    print("status: OK")


if __name__ == "__main__":
    main()
