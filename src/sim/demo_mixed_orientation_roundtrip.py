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

from src.sim.demo_xyz_joint_roundtrip import (
    DEFAULT_MODEL,
    DEFAULT_TARGET_RADIUS,
    DEFAULT_TOOL_RADIUS,
    DEFAULT_XYZ_A,
    DEFAULT_XYZ_B,
    TOOL_DOWN_ROTATION,
    TOOL_HORIZONTAL_APPROACH_B,
    print_roundtrip,
    rotation_from_tool_z,
    set_target_marker,
    solve_from_pose,
    tool_target_from_sphere_center,
)


def smooth_alpha(value: float) -> float:
    return 0.5 - 0.5 * np.cos(np.pi * value)


def interpolate_roundtrip(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
    if t < 0.5:
        alpha = smooth_alpha(t / 0.5)
        return (1.0 - alpha) * q_a + alpha * q_b
    alpha = smooth_alpha((t - 0.5) / 0.5)
    return (1.0 - alpha) * q_b + alpha * q_a


def solve_demo(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    target_radius: float,
    tool_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    approach_a = np.array([0.0, 0.0, 1.0], dtype=float)
    approach_b = TOOL_HORIZONTAL_APPROACH_B
    rotation_a = TOOL_DOWN_ROTATION
    rotation_b = rotation_from_tool_z(-approach_b, preferred_x=np.array([0.0, 0.0, 1.0]))
    tool_target_a = tool_target_from_sphere_center(xyz_a, approach_a, target_radius, tool_radius)
    tool_target_b = tool_target_from_sphere_center(xyz_b, approach_b, target_radius, tool_radius)
    q_a = solve_from_pose(tool_target_a, rotation_a)
    q_b = solve_from_pose(tool_target_b, rotation_b)
    return q_a, q_b, tool_target_a, tool_target_b, rotation_a, rotation_b


def visualize(
    q_a: np.ndarray,
    q_b: np.ndarray,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    model_path: Path,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = q_a
    data.ctrl[:] = q_a
    set_target_marker(model, data, "target_marker", xyz_a)
    set_target_marker(model, data, "target_marker_b", xyz_b)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = ((time.time() - start) % 6.0) / 6.0
            q = interpolate_roundtrip(q_a, q_b, t)
            data.qpos[:] = q
            data.ctrl[:] = q
            set_target_marker(model, data, "target_marker", xyz_a)
            set_target_marker(model, data, "target_marker_b", xyz_b)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 2: red target uses downward contact, blue target uses horizontal contact."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--xyz-a", nargs=3, type=float, default=DEFAULT_XYZ_A.tolist())
    parser.add_argument("--xyz-b", nargs=3, type=float, default=DEFAULT_XYZ_B.tolist())
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS)
    parser.add_argument("--tool-radius", type=float, default=DEFAULT_TOOL_RADIUS)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    xyz_a = np.array(args.xyz_a, dtype=float)
    xyz_b = np.array(args.xyz_b, dtype=float)
    q_a, q_b, tool_target_a, tool_target_b, rotation_a, rotation_b = solve_demo(
        xyz_a, xyz_b, args.target_radius, args.tool_radius
    )
    print_roundtrip("A red/down", q_a, xyz_a, tool_target_a, args.target_radius, args.tool_radius, rotation_a[:, 2])
    print_roundtrip("B blue/horizontal", q_b, xyz_b, tool_target_b, args.target_radius, args.tool_radius, rotation_b[:, 2])

    if not args.no_viewer:
        visualize(q_a, q_b, xyz_a, xyz_b, args.model)


if __name__ == "__main__":
    main()
