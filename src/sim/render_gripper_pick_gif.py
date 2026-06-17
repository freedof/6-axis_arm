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

from src.sim.gripper_model import GRIPPER_OPEN_QPOS
from src.sim.gripper_pick_motion import (
    GRIPPER_DOF,
    ROBOT_DOF,
    PickTrajectory,
    PlannedPickTrajectory,
    command_at_time,
    planned_command_at_time,
    solve_pick_trajectory,
)
from src.sim.gripper_pick_scene import DEFAULT_PICK_MODEL, write_pick_scene_model


DEFAULT_OUTPUT = ROOT / "outputs" / "gripper_pick" / "simplified_gripper_pick_cube.gif"


def configure_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.35, -0.55, 0.16], dtype=float)
    camera.distance = 0.82
    camera.azimuth = 138.0
    camera.elevation = -24.0
    return camera


def render_gif(
    model_path: Path,
    output_path: Path,
    *,
    width: int = 960,
    height: int = 720,
    frames: int = 160,
    fps: int = 20,
    show_sites: bool = False,
    trajectory: PickTrajectory | None = None,
    planned_trajectory: PlannedPickTrajectory | None = None,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = configure_camera()
    if trajectory is not None and planned_trajectory is not None:
        raise ValueError("Pass either trajectory or planned_trajectory, not both.")
    if trajectory is None and planned_trajectory is None:
        trajectory = solve_pick_trajectory()
    q_ready = planned_trajectory.poses.q_ready if planned_trajectory is not None else trajectory.q_ready

    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = q_ready
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    images: list[Image.Image] = []
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    if not show_sites:
        scene_option.sitegroup[:] = 0

    for frame_index in range(frames):
        for step_index in range(steps_per_frame):
            sim_time = (frame_index * steps_per_frame + step_index) * model.opt.timestep
            if planned_trajectory is None:
                q_des, gripper_des = command_at_time(trajectory, sim_time)
            else:
                q_des, gripper_des = planned_command_at_time(planned_trajectory, sim_time)
            data.ctrl[:ROBOT_DOF] = q_des
            data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper_des
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        images.append(Image.fromarray(renderer.render()))

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
    print(f"gif: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a simplified-gripper cube pick demo to GIF.")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_PICK_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--show-sites", action="store_true", help="Show MuJoCo site markers such as gripper_tcp.")
    args = parser.parse_args()

    model_path = write_pick_scene_model(args.model_output)
    render_gif(
        model_path,
        args.output,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        show_sites=args.show_sites,
    )


if __name__ == "__main__":
    main()
