from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.rrt_connect import path_length
from src.planning.trajectory import make_roundtrip_path, parameterize_joint_path
from src.sim.demo_xyz_joint_roundtrip import set_target_marker
from src.sim.obstacle_scene import DEFAULT_OBSTACLE_MODEL, base_planning_case, default_obstacle_case, obstacle_case_by_id, write_obstacle_model
from src.sim.planning_model import DEFAULT_PLANNING_MODEL, write_planning_model
from src.sim.render_planned_roundtrip_gif import case_by_id, plan_case_roundtrip


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a time-parameterized planned CR5 roundtrip.")
    parser.add_argument("--case-id", default="TC-RRT-MULTI-004")
    parser.add_argument("--obstacle-case-id", default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_PLANNING_MODEL)
    parser.add_argument("--obstacle", action="store_true", help="Use the default obstacle-avoidance scene and case.")
    parser.add_argument("--shortcut", action="store_true", help="Use shortcut-smoothed path.")
    parser.add_argument("--max-joint-velocity", type=float, default=0.8)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.6)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    if args.obstacle:
        obstacle_case = default_obstacle_case() if args.obstacle_case_id is None else obstacle_case_by_id(args.obstacle_case_id)
        model_path = write_obstacle_model(DEFAULT_OBSTACLE_MODEL, obstacle_case=obstacle_case)
        case = base_planning_case(obstacle_case)
    else:
        model_path = write_planning_model(args.model)
        case = case_by_id(args.case_id)
    planned = plan_case_roundtrip(model_path, case, shortcut=args.shortcut)
    trajectory = parameterize_joint_path(
        make_roundtrip_path(planned.path),
        max_joint_velocity=args.max_joint_velocity,
        max_joint_acceleration=args.max_joint_acceleration,
    )

    print(f"case: {case.case_id} {case.name}")
    print(f"model: {model_path}")
    print(f"path_waypoints: {len(planned.path)}")
    print(f"path_length: {path_length(planned.path):.6f}")
    print(f"trajectory_duration_s: {trajectory.duration:.3f}")
    print(f"max_joint_velocity_rad_s: {trajectory.max_joint_velocity:.3f}")
    print(f"max_joint_acceleration_rad_s2: {trajectory.max_joint_acceleration:.3f}")

    if args.no_viewer:
        print("status: OK")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    set_target_marker(model, data, "target_marker", planned.xyz_a)
    set_target_marker(model, data, "target_marker_b", planned.xyz_b)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            elapsed = (time.time() - start) % max(trajectory.duration, 1e-9)
            q, _, _ = trajectory.sample(elapsed)
            data.qpos[:] = q
            data.ctrl[:] = q
            set_target_marker(model, data, "target_marker", planned.xyz_a)
            set_target_marker(model, data, "target_marker_b", planned.xyz_b)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
