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
from src.sim.demo_mixed_orientation_roundtrip import interpolate_roundtrip, solve_demo
from src.sim.demo_xyz_joint_roundtrip import (
    DEFAULT_MODEL,
    DEFAULT_TARGET_RADIUS,
    DEFAULT_TOOL_RADIUS,
    DEFAULT_XYZ_A,
    DEFAULT_XYZ_B,
    set_target_marker,
)
from src.sim.render_timing import DEFAULT_TARGET_DWELL_SECONDS, roundtrip_motion_alpha


DEFAULT_OUTPUT = ROOT / "outputs" / "roundtrip_mixed_orientation.gif"


def configure_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.24, -0.58, 0.28])
    camera.distance = 1.18
    camera.azimuth = 140.0
    camera.elevation = -16.0
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
    q_a, q_b, tool_target_a, tool_target_b, rotation_a, rotation_b = solve_demo(
        xyz_a, xyz_b, target_radius, tool_radius
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = configure_camera()

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
        images.append(Image.fromarray(renderer.render()))

    renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )

    robot = dobot_cr5_simplified()
    for label, q, sphere_center, tool_target, rotation in (
        ("A red/down", q_a, xyz_a, tool_target_a, rotation_a),
        ("B blue/horizontal", q_b, xyz_b, tool_target_b, rotation_b),
    ):
        fk = forward_kinematics(robot, q)
        tangent_gap = np.linalg.norm(fk.position - sphere_center) - (target_radius + tool_radius)
        desired_angle = np.rad2deg(np.arccos(np.clip(np.dot(fk.rotation[:, 2], rotation[:, 2]), -1.0, 1.0)))
        print(
            f"{label}: tool0_error={np.linalg.norm(fk.position - tool_target):.8e} m, "
            f"surface_tangent_gap={tangent_gap:.8e} m, "
            f"desired_tool_z_angle={desired_angle:.6f} deg"
        )
    print(f"gif: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render demo 2: red target downward, blue target horizontal.")
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
