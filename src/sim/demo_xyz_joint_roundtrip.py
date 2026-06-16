from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robot.ik import solve_ik_multi_start
from src.robot.kinematics import forward_kinematics
from src.robot.model import dobot_cr5_simplified


DEFAULT_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_simplified.xml"
READY_Q = np.array([0.0, -0.6, 0.85, 0.0, -0.8, 0.0], dtype=float)
DEFAULT_XYZ_A = np.array([0.35, -0.55, 0.20], dtype=float)
DEFAULT_XYZ_B = np.array([0.20, -0.60, 0.30], dtype=float)
DEFAULT_TARGET_RADIUS = 0.014
DEFAULT_TOOL_RADIUS = 0.020
DEFAULT_APPROACH = np.array([0.0, 0.0, 1.0], dtype=float)
TOOL_DOWN_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
TOOL_HORIZONTAL_APPROACH_B = np.array([-1.0, 0.0, 0.0], dtype=float)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("Approach direction must be non-zero.")
    return vector / norm


def tool_target_from_sphere_center(
    sphere_center: np.ndarray,
    approach: np.ndarray,
    target_radius: float,
    tool_radius: float,
) -> np.ndarray:
    return sphere_center + normalize(approach) * (target_radius + tool_radius)


def sphere_center_from_tool_target(
    tool_target: np.ndarray,
    approach: np.ndarray,
    target_radius: float,
    tool_radius: float,
) -> np.ndarray:
    return tool_target - normalize(approach) * (target_radius + tool_radius)


def rotation_from_tool_z(tool_z: np.ndarray, preferred_x: np.ndarray | None = None) -> np.ndarray:
    z_axis = normalize(tool_z)
    x_seed = np.array([1.0, 0.0, 0.0], dtype=float) if preferred_x is None else normalize(preferred_x)
    x_axis = x_seed - np.dot(x_seed, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_seed = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = x_seed - np.dot(x_seed, z_axis) * z_axis
    x_axis = normalize(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def make_tool_pose(tool_target: np.ndarray, target_rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = target_rotation
    pose[:3, 3] = tool_target
    return pose


def make_tool_down_pose(tool_target: np.ndarray) -> np.ndarray:
    return make_tool_pose(tool_target, TOOL_DOWN_ROTATION)


def set_target_marker(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, xyz: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    mocap_id = model.body_mocapid[body_id]
    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = xyz


def print_roundtrip(
    label: str,
    q: np.ndarray,
    sphere_center: np.ndarray,
    tool_target: np.ndarray,
    target_radius: float,
    tool_radius: float,
    desired_tool_z: np.ndarray | None = None,
) -> None:
    robot = dobot_cr5_simplified()
    fk = forward_kinematics(robot, q)
    fk_xyz = fk.position
    tool_z = fk.rotation[:, 2]
    down_alignment = float(np.clip(np.dot(tool_z, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
    down_angle_deg = float(np.rad2deg(np.arccos(down_alignment)))
    desired_tool_z = np.array([0.0, 0.0, -1.0], dtype=float) if desired_tool_z is None else normalize(desired_tool_z)
    desired_alignment = float(np.clip(np.dot(tool_z, desired_tool_z), -1.0, 1.0))
    desired_angle_deg = float(np.rad2deg(np.arccos(desired_alignment)))
    center_distance = float(np.linalg.norm(fk_xyz - sphere_center))
    tangent_gap = center_distance - (target_radius + tool_radius)
    print(f"{label}) xyz -> joint angles")
    print("sphere_center_xyz_m:", np.round(sphere_center, 6))
    print("tool0_target_xyz_m:", np.round(tool_target, 6))
    print("joint_rad:", np.round(q, 6))
    print("joint_deg:", np.round(np.rad2deg(q), 3))
    print(f"{label}) joint angles -> xyz")
    print("fk_tool0_xyz_m:", np.round(fk_xyz, 6))
    print("tool0_position_error_m:", f"{np.linalg.norm(fk_xyz - tool_target):.8e}")
    print("surface_tangent_gap_m:", f"{tangent_gap:.8e}")
    print("tool_down_angle_deg:", f"{down_angle_deg:.6f}")
    print("desired_tool_z_angle_deg:", f"{desired_angle_deg:.6f}")
    print()


def solve_from_pose(tool_target: np.ndarray, target_rotation: np.ndarray | None = None) -> np.ndarray:
    robot = dobot_cr5_simplified()
    result = solve_ik_multi_start(
        robot,
        make_tool_pose(tool_target, TOOL_DOWN_ROTATION if target_rotation is None else target_rotation),
        seeds=[READY_Q],
        position_tolerance=1e-5,
        rotation_tolerance=1e-4,
        random_starts=48,
    )
    if not result.success:
        raise RuntimeError(
            "Tool-down IK failed. "
            f"Best position error: {result.position_error:.6f} m, "
            f"rotation error: {result.rotation_error:.6f}"
        )
    return result.q


def visualize(q_goal: np.ndarray, xyz: np.ndarray, model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    q_start = READY_Q.copy()
    data.qpos[:] = q_start
    data.ctrl[:] = q_start
    set_target_marker(model, data, "target_marker", xyz)
    set_target_marker(model, data, "target_marker_b", xyz)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            elapsed = time.time() - start
            alpha = min(elapsed / 3.0, 1.0)
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            q_des = (1.0 - alpha) * q_start + alpha * q_goal
            data.qpos[:] = q_des
            data.ctrl[:] = q_des
            set_target_marker(model, data, "target_marker", xyz)
            set_target_marker(model, data, "target_marker_b", xyz)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def visualize_roundtrip(q_a: np.ndarray, xyz_a: np.ndarray, q_b: np.ndarray, xyz_b: np.ndarray, model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = q_a
    data.ctrl[:] = q_a
    set_target_marker(model, data, "target_marker", xyz_a)
    set_target_marker(model, data, "target_marker_b", xyz_b)
    mujoco.mj_forward(model, data)

    segment_seconds = 3.0
    dwell_seconds = 0.5
    cycle_seconds = 2.0 * (segment_seconds + dwell_seconds)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = (time.time() - start) % cycle_seconds
            if t < segment_seconds:
                alpha = t / segment_seconds
                q_from, q_to = q_a, q_b
            elif t < segment_seconds + dwell_seconds:
                alpha = 1.0
                q_from, q_to = q_a, q_b
            elif t < 2.0 * segment_seconds + dwell_seconds:
                alpha = (t - segment_seconds - dwell_seconds) / segment_seconds
                q_from, q_to = q_b, q_a
            else:
                alpha = 1.0
                q_from, q_to = q_b, q_a

            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            q_des = (1.0 - alpha) * q_from + alpha * q_to
            data.qpos[:] = q_des
            data.ctrl[:] = q_des
            set_target_marker(model, data, "target_marker", xyz_a)
            set_target_marker(model, data, "target_marker_b", xyz_b)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize xyz->joint IK and joint->xyz FK in MuJoCo."
    )
    parser.add_argument(
        "--xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Single target sphere center in meters.",
    )
    parser.add_argument(
        "--xyz-a",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_XYZ_A.tolist(),
        help="First roundtrip target sphere center in meters.",
    )
    parser.add_argument(
        "--xyz-b",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_XYZ_B.tolist(),
        help="Second roundtrip target sphere center in meters.",
    )
    parser.add_argument(
        "--q",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Optional joint angles in radians. If omitted, IK solves from --xyz.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS, help="Target sphere radius in meters.")
    parser.add_argument("--tool-radius", type=float, default=DEFAULT_TOOL_RADIUS, help="Spherical radius of the MuJoCo tool capsule end in meters.")
    parser.add_argument(
        "--approach",
        nargs=3,
        type=float,
        default=DEFAULT_APPROACH.tolist(),
        metavar=("X", "Y", "Z"),
        help="World direction from sphere center to tool0 capsule-center target.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Print results without opening MuJoCo viewer.")
    args = parser.parse_args()
    approach = normalize(np.array(args.approach, dtype=float))

    if args.q is None and args.xyz is None:
        xyz_a = np.array(args.xyz_a, dtype=float)
        xyz_b = np.array(args.xyz_b, dtype=float)
        tool_target_a = tool_target_from_sphere_center(xyz_a, approach, args.target_radius, args.tool_radius)
        tool_target_b = tool_target_from_sphere_center(xyz_b, approach, args.target_radius, args.tool_radius)
        q_a = solve_from_pose(tool_target_a)
        q_b = solve_from_pose(tool_target_b)
        print_roundtrip("A", q_a, xyz_a, tool_target_a, args.target_radius, args.tool_radius)
        print_roundtrip("B", q_b, xyz_b, tool_target_b, args.target_radius, args.tool_radius)
        if not args.no_viewer:
            visualize_roundtrip(q_a, xyz_a, q_b, xyz_b, args.model)
        return

    xyz = np.array(DEFAULT_XYZ_A if args.xyz is None else args.xyz, dtype=float)
    if args.q is None:
        tool_target = tool_target_from_sphere_center(xyz, approach, args.target_radius, args.tool_radius)
        q_goal = solve_from_pose(tool_target)
    else:
        q_goal = dobot_cr5_simplified().clamp(np.array(args.q, dtype=float))
        tool_target = forward_kinematics(dobot_cr5_simplified(), q_goal).position
        xyz = sphere_center_from_tool_target(tool_target, approach, args.target_radius, args.tool_radius)

    print_roundtrip("1", q_goal, xyz, tool_target, args.target_radius, args.tool_radius)
    if not args.no_viewer:
        visualize(q_goal, xyz, args.model)


if __name__ == "__main__":
    main()
