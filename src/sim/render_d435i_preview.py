from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.d435i_camera import D435iCamera, D435iParams, depth_stats
from src.sim.d435i_model import DEFAULT_D435I_PICK_MODEL, write_d435i_pick_scene_model
from src.sim.gripper_model import GRIPPER_OPEN_QPOS
from src.robot.model import dobot_cr5_simplified
from src.sim.gripper_pick_motion import GRIPPER_DOF, ROBOT_DOF, solve_pick_trajectory
from src.sim.gripper_pick_motion import _solve_gripper_center_pose
from src.sim.gripper_pick_scene import CUBE_CENTER


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "d435i_preview"


def render_preview(
    model_path: Path,
    output_dir: Path,
    *,
    width: int = 424,
    height: int = 240,
    seed: int = 7,
    pose: str = "above",
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    trajectory = solve_pick_trajectory()
    q_robot = _trajectory_pose(trajectory, pose)

    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = q_robot
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = q_robot
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    camera = D435iCamera(model, data, width=width, height=height, seed=seed)
    try:
        rgb, raw_depth = camera.render(apply_noise=False)
        _, noisy_depth = camera.render(apply_noise=True)
        intrinsics = camera.get_intrinsics()
        extrinsic = camera.get_pose_matrix()
    finally:
        camera.close()
    context_rgb = _render_context_rgb(model, data, width=max(width, 640), height=max(height, 480))

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "d435i_rgb.png"
    context_rgb_path = output_dir / "d435i_external_context.png"
    raw_depth_path = output_dir / "d435i_raw_depth.npy"
    noisy_depth_path = output_dir / "d435i_noisy_depth.npy"
    raw_vis_path = output_dir / "d435i_raw_depth_vis.png"
    noisy_vis_path = output_dir / "d435i_noisy_depth_vis.png"

    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(context_rgb).save(context_rgb_path)
    np.save(raw_depth_path, raw_depth)
    np.save(noisy_depth_path, noisy_depth)
    _depth_visual(raw_depth).save(raw_vis_path)
    _depth_visual(noisy_depth).save(noisy_vis_path)

    return {
        "rgb_path": str(rgb_path),
        "context_rgb_path": str(context_rgb_path),
        "raw_depth_path": str(raw_depth_path),
        "noisy_depth_path": str(noisy_depth_path),
        "raw_depth_vis_path": str(raw_vis_path),
        "noisy_depth_vis_path": str(noisy_vis_path),
        "intrinsics": intrinsics.tolist(),
        "extrinsic_world_to_camera": extrinsic.tolist(),
        "raw_depth_stats": depth_stats(raw_depth),
        "noisy_depth_stats": depth_stats(noisy_depth),
        "pose": pose,
    }


def _render_context_rgb(model: mujoco.MjModel, data: mujoco.MjData, *, width: int, height: int) -> np.ndarray:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.35, -0.55, 0.085], dtype=float)
    camera.distance = 0.42
    camera.azimuth = 35.0
    camera.elevation = -18.0

    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.sitegroup[:] = 0

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        return renderer.render().copy()
    finally:
        renderer.close()


def _trajectory_pose(trajectory, pose: str) -> np.ndarray:
    if pose == "scan":
        robot = dobot_cr5_simplified()
        scan_center = np.array(CUBE_CENTER, dtype=float) + np.array([0.0, 0.0, 0.240], dtype=float)
        return _solve_gripper_center_pose(robot, scan_center, [trajectory.q_lift, trajectory.q_above, trajectory.q_ready])
    poses = {
        "ready": trajectory.q_ready,
        "above": trajectory.q_above,
        "grasp": trajectory.q_grasp,
        "lift": trajectory.q_lift,
    }
    if pose not in poses:
        raise ValueError(f"Unknown D435i preview pose '{pose}'. Expected one of: {', '.join(poses)}")
    return poses[pose]


def _depth_visual(depth: np.ndarray, params: D435iParams | None = None) -> Image.Image:
    p = D435iParams() if params is None else params
    valid = (depth >= p.min_z_m) & (depth <= p.max_z_m)
    clipped = np.clip(depth, p.min_z_m, min(p.max_z_m, 2.0))
    norm = (clipped - p.min_z_m) / (min(p.max_z_m, 2.0) - p.min_z_m)
    gray = (255.0 * (1.0 - norm)).astype(np.uint8)
    gray[~valid] = 0
    return Image.fromarray(gray, mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RGB/depth previews from the gripper-mounted D435i camera.")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_D435I_PICK_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=424)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pose", choices=("ready", "above", "grasp", "lift", "scan"), default="above")
    args = parser.parse_args()

    model_path = write_d435i_pick_scene_model(args.model_output)
    result = render_preview(model_path, args.output_dir, width=args.width, height=args.height, seed=args.seed, pose=args.pose)
    print("model:", model_path)
    print("pose:", result["pose"])
    print("rgb:", result["rgb_path"])
    print("context_rgb:", result["context_rgb_path"])
    print("raw_depth_vis:", result["raw_depth_vis_path"])
    print("noisy_depth_vis:", result["noisy_depth_vis_path"])
    print("raw_depth_stats:", result["raw_depth_stats"])
    print("noisy_depth_stats:", result["noisy_depth_stats"])


if __name__ == "__main__":
    main()
