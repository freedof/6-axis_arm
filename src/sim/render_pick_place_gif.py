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
from src.sim.gripper_pick_motion import GRIPPER_DOF, ROBOT_DOF
from src.sim.gripper_pick_scene import DEFAULT_MULTI_OBJECT_MODEL, write_multi_object_scene_model
from src.sim.pick_place_motion import PlannedPickPlaceTrajectory, pick_place_command_at_time


DEFAULT_OUTPUT = ROOT / "outputs" / "pick_place" / "multi_object_pick_place.gif"


def configure_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.35, -0.55, 0.16], dtype=float)
    camera.distance = 0.82
    camera.azimuth = 132.0
    camera.elevation = -26.0
    return camera


def render_pick_place_gif(
    model_path: Path,
    output_path: Path,
    *,
    planned_trajectory: PlannedPickPlaceTrajectory,
    width: int = 960,
    height: int = 720,
    frames: int = 360,
    fps: int = 20,
    show_sites: bool = False,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = configure_camera()

    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = planned_trajectory.poses.q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = planned_trajectory.poses.q_ready
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
            q_des, gripper_des = pick_place_command_at_time(planned_trajectory, sim_time)
            data.ctrl[:ROBOT_DOF] = q_des
            data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper_des
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
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
    print(f"gif: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a planned multi-object pick-and-place GIF.")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MULTI_OBJECT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--show-sites", action="store_true")
    args = parser.parse_args()

    raise SystemExit("Use src.mcp_robot.skills.language_multi_view_pick_and_place or a verification script to build the planned trajectory first.")


if __name__ == "__main__":
    main()
