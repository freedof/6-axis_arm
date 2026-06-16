from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robot.kinematics import forward_kinematics
from src.robot.model import dobot_cr5_simplified
from src.sim.demo_xyz_joint_roundtrip import (
    DEFAULT_MODEL,
    DEFAULT_TARGET_RADIUS,
    DEFAULT_TOOL_RADIUS,
    DEFAULT_XYZ_A,
    DEFAULT_XYZ_B,
    set_target_marker,
    solve_from_pose,
    tool_target_from_sphere_center,
)
from src.sim.render_timing import DEFAULT_TARGET_DWELL_SECONDS, roundtrip_motion_alpha


DEFAULT_OUTPUT = ROOT / "outputs" / "roundtrip_touch_tool_down.gif"


def interpolate_roundtrip(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
    segment = 0.5
    if t < segment:
        alpha = t / segment
        q_from, q_to = q_a, q_b
    else:
        alpha = (t - segment) / segment
        q_from, q_to = q_b, q_a
    alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
    return (1.0 - alpha) * q_from + alpha * q_to


def configure_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.22, -0.58, 0.30])
    camera.distance = 1.18
    camera.azimuth = 145.0
    camera.elevation = -18.0
    return camera


def render_gif(
    model_path: Path,
    output_path: Path,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    width: int,
    height: int,
    frames: int,
    fps: int,
    target_radius: float,
    tool_radius: float,
    target_dwell_seconds: float = DEFAULT_TARGET_DWELL_SECONDS,
) -> None:
    approach = np.array([0.0, 0.0, 1.0], dtype=float)
    tool_target_a = tool_target_from_sphere_center(xyz_a, approach, target_radius, tool_radius)
    tool_target_b = tool_target_from_sphere_center(xyz_b, approach, target_radius, tool_radius)
    q_a = solve_from_pose(tool_target_a)
    q_b = solve_from_pose(tool_target_b)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = configure_camera(model)

    images: list[Image.Image] = []
    for frame_index in range(frames):
        alpha, reverse = roundtrip_motion_alpha(
            frame_index,
            frames,
            fps,
            target_dwell_seconds=target_dwell_seconds,
        )
        t = 0.5 + 0.5 * alpha if reverse else 0.5 * alpha
        q = interpolate_roundtrip(q_a, q_b, t)
        data.qpos[:] = q
        data.ctrl[:] = q
        set_target_marker(model, data, "target_marker", xyz_a)
        set_target_marker(model, data, "target_marker_b", xyz_b)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        images.append(Image.fromarray(rgb))

    renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    robot = dobot_cr5_simplified()
    for label, q, sphere_center, tool_target in (
        ("A", q_a, xyz_a, tool_target_a),
        ("B", q_b, xyz_b, tool_target_b),
    ):
        fk = forward_kinematics(robot, q)
        tangent_gap = np.linalg.norm(fk.position - sphere_center) - (target_radius + tool_radius)
        down_angle = np.rad2deg(np.arccos(np.clip(np.dot(fk.rotation[:, 2], [0.0, 0.0, -1.0]), -1.0, 1.0)))
        print(
            f"{label}: tool0_error={np.linalg.norm(fk.position - tool_target):.8e} m, "
            f"surface_tangent_gap={tangent_gap:.8e} m, "
            f"tool_down_angle={down_angle:.6f} deg"
        )
    print(f"gif: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the two-target CR5 roundtrip demo to a GIF.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xyz-a", nargs=3, type=float, default=DEFAULT_XYZ_A.tolist())
    parser.add_argument("--xyz-b", nargs=3, type=float, default=DEFAULT_XYZ_B.tolist())
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS)
    parser.add_argument("--tool-radius", type=float, default=DEFAULT_TOOL_RADIUS)
    parser.add_argument("--target-dwell-seconds", type=float, default=DEFAULT_TARGET_DWELL_SECONDS)
    args = parser.parse_args()

    render_gif(
        args.model,
        args.output,
        np.array(args.xyz_a, dtype=float),
        np.array(args.xyz_b, dtype=float),
        args.width,
        args.height,
        args.frames,
        args.fps,
        args.target_radius,
        args.tool_radius,
        target_dwell_seconds=args.target_dwell_seconds,
    )


if __name__ == "__main__":
    main()
