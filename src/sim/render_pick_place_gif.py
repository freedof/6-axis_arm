from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.gripper_model import GRIPPER_OPEN_QPOS
from src.sim.gripper_pick_motion import GRIPPER_DOF, ROBOT_DOF
from src.sim.gripper_pick_scene import DEFAULT_MULTI_OBJECT_MODEL, write_multi_object_scene_model
from src.sim.pick_place_motion import PlannedPickPlaceTrajectory, PickPlaceSequenceItem, SEQUENCE_BRIDGE_SECONDS, pick_place_command_at_time, pick_place_required_frames, pick_place_sequence_required_frames, sequence_bridge_command_at_time, sequence_bridge_waypoints


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
    _save_gif(images, output_path, fps)
    print(f"gif: {output_path}")


def render_pick_place_sequence_gif(
    model_path: Path,
    output_path: Path,
    *,
    items: tuple[PickPlaceSequenceItem, ...] | list[PickPlaceSequenceItem],
    width: int = 960,
    height: int = 720,
    frames: int = 0,
    fps: int = 20,
    show_sites: bool = False,
    bridge_seconds: float = SEQUENCE_BRIDGE_SECONDS,
) -> None:
    sequence = tuple(items)
    if not sequence:
        raise ValueError("items must contain at least one pick-and-place task.")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = configure_camera()

    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = sequence[0].planned_trajectory.poses.q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = sequence[0].planned_trajectory.poses.q_ready
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    images: list[Image.Image] = []
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    if not show_sites:
        scene_option.sitegroup[:] = 0

    total_frames = pick_place_sequence_required_frames(sequence, frames=frames, fps=fps, bridge_seconds=bridge_seconds)
    frame_count = 0
    for index, item in enumerate(sequence):
        if index > 0:
            bridge_frames = int(np.ceil(max(0.0, bridge_seconds) * fps))
            waypoints = sequence_bridge_waypoints(
                data.qpos[:ROBOT_DOF].copy(),
                sequence[index - 1].planned_trajectory,
                item.planned_trajectory,
            )
            for frame_index in range(bridge_frames):
                for step_index in range(steps_per_frame):
                    step = frame_index * steps_per_frame + step_index + 1
                    q_des = sequence_bridge_command_at_time(waypoints, step * model.opt.timestep, bridge_seconds)
                    data.ctrl[:ROBOT_DOF] = q_des
                    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
                    mujoco.mj_step(model, data)
                _append_frame(renderer, data, camera, scene_option, images)
                frame_count += 1

        task_frames = pick_place_required_frames(item.planned_trajectory, frames=0, fps=fps)
        for frame_index in range(task_frames):
            for step_index in range(steps_per_frame):
                sim_time = (frame_index * steps_per_frame + step_index) * model.opt.timestep
                q_des, gripper_des = pick_place_command_at_time(item.planned_trajectory, sim_time)
                data.ctrl[:ROBOT_DOF] = q_des
                data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper_des
                mujoco.mj_step(model, data)
            _append_frame(renderer, data, camera, scene_option, images)
            frame_count += 1

    while frame_count < total_frames:
        data.ctrl[:ROBOT_DOF] = sequence[-1].planned_trajectory.poses.q_retreat
        data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        _append_frame(renderer, data, camera, scene_option, images)
        frame_count += 1

    renderer.close()
    _save_gif(images, output_path, fps)
    print(f"gif: {output_path}")



def play_pick_place_viewer(
    model_path: Path,
    *,
    planned_trajectory: PlannedPickPlaceTrajectory,
    frames: int = 0,
    fps: int = 20,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = planned_trajectory.poses.q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = planned_trajectory.poses.q_ready
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    total_frames = pick_place_required_frames(planned_trajectory, frames=frames, fps=fps)
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        for frame_index in range(total_frames):
            if not viewer.is_running():
                break
            for step_index in range(steps_per_frame):
                sim_time = (frame_index * steps_per_frame + step_index) * model.opt.timestep
                q_des, gripper_des = pick_place_command_at_time(planned_trajectory, sim_time)
                data.ctrl[:ROBOT_DOF] = q_des
                data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper_des
                mujoco.mj_step(model, data)
            viewer.sync()
            target_time = start + (frame_index + 1) / max(float(fps), 1.0)
            time.sleep(max(0.0, target_time - time.time()))

def play_pick_place_sequence_viewer(
    model_path: Path,
    *,
    items: tuple[PickPlaceSequenceItem, ...] | list[PickPlaceSequenceItem],
    frames: int = 0,
    fps: int = 20,
    bridge_seconds: float = SEQUENCE_BRIDGE_SECONDS,
) -> None:
    sequence = tuple(items)
    if not sequence:
        raise ValueError("items must contain at least one pick-and-place task.")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = sequence[0].planned_trajectory.poses.q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = sequence[0].planned_trajectory.poses.q_ready
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    total_frames = pick_place_sequence_required_frames(sequence, frames=frames, fps=fps, bridge_seconds=bridge_seconds)
    frame_count = 0
    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for index, item in enumerate(sequence):
            if not viewer.is_running():
                break
            if index > 0:
                bridge_frames = int(np.ceil(max(0.0, bridge_seconds) * fps))
                waypoints = sequence_bridge_waypoints(
                    data.qpos[:ROBOT_DOF].copy(),
                    sequence[index - 1].planned_trajectory,
                    item.planned_trajectory,
                )
                for frame_index in range(bridge_frames):
                    if not viewer.is_running():
                        break
                    for step_index in range(steps_per_frame):
                        step = frame_index * steps_per_frame + step_index + 1
                        q_des = sequence_bridge_command_at_time(waypoints, step * model.opt.timestep, bridge_seconds)
                        data.ctrl[:ROBOT_DOF] = q_des
                        data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
                        mujoco.mj_step(model, data)
                    viewer.sync()
                    frame_count += 1
                    _sleep_to_frame(start, frame_count, fps)

            task_frames = pick_place_required_frames(item.planned_trajectory, frames=0, fps=fps)
            for frame_index in range(task_frames):
                if not viewer.is_running():
                    break
                for step_index in range(steps_per_frame):
                    sim_time = (frame_index * steps_per_frame + step_index) * model.opt.timestep
                    q_des, gripper_des = pick_place_command_at_time(item.planned_trajectory, sim_time)
                    data.ctrl[:ROBOT_DOF] = q_des
                    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper_des
                    mujoco.mj_step(model, data)
                viewer.sync()
                frame_count += 1
                _sleep_to_frame(start, frame_count, fps)

        while viewer.is_running() and frame_count < total_frames:
            data.ctrl[:ROBOT_DOF] = sequence[-1].planned_trajectory.poses.q_retreat
            data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            viewer.sync()
            frame_count += 1
            _sleep_to_frame(start, frame_count, fps)


def _sleep_to_frame(start: float, frame_count: int, fps: int) -> None:
    target_time = start + frame_count / max(float(fps), 1.0)
    time.sleep(max(0.0, target_time - time.time()))

def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _append_frame(renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera, scene_option: mujoco.MjvOption, images: list[Image.Image]) -> None:
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    images.append(Image.fromarray(renderer.render()))


def _save_gif(images: list[Image.Image], output_path: Path, fps: int) -> None:
    if not images:
        raise RuntimeError("No frames rendered.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
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
