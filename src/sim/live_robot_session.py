from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp_robot import skills
from src.sim.d435i_model import DEFAULT_D435I_MULTI_OBJECT_MODEL, write_d435i_multi_object_scene_model
from src.sim.gripper_model import GRIPPER_OPEN_QPOS
from src.sim.gripper_pick_motion import GRIPPER_CLOSED_QPOS, GRIPPER_DOF, ROBOT_DOF, solve_pick_trajectory
from src.sim.pick_place_motion import (
    PLACE_OPEN_SECONDS,
    PLACE_RETREAT_DWELL_SECONDS,
    PLANNED_PICK_ABOVE_DWELL_SECONDS,
    PLANNED_PICK_CLOSE_SECONDS,
    POST_GRASP_SETTLE_SECONDS,
    PickPlaceSequenceItem,
    sequence_bridge_command_at_time,
    sequence_bridge_waypoints,
)
from src.sim.render_d435i_preview import _trajectory_pose


DEFAULT_COMMAND_PATH = ROOT / "outputs" / "live_session" / "command.json"
DEFAULT_STATUS_PATH = ROOT / "outputs" / "live_session" / "status.json"
DEFAULT_POSES = ("scan_high", "scan_front_high", "scan_left_high", "scan_right_high")
DEFAULT_DESTINATION = {"type": "tray", "region": "tray", "world_xy_m": [0.64, -0.37]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live MuJoCo robot session controlled by Codex command files.")
    parser.add_argument("--command-path", type=Path, default=DEFAULT_COMMAND_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--provider", default="color_fixture", choices=list(skills.VL_PROVIDERS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--max-parallel-vl", type=int, default=4)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    args = parser.parse_args()

    command_path = _absolute(args.command_path)
    status_path = _absolute(args.status_path)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if command_path.exists():
        command_path.unlink()

    session_info = {
        "pid": os.getpid(),
        "provider": args.provider,
        "vl_model": args.model,
        "config_path": str(args.config_path) if args.config_path else None,
        "command_path": str(command_path),
        "status_path": str(status_path),
    }
    startup_start = time.perf_counter()
    _write_status(status_path, {**session_info, "status": "starting", "phase": "loading_model"})
    model_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    _write_status(status_path, {**session_info, "status": "starting", "phase": "solving_ready_pose", "model_path": str(model_path)})
    ready_q = solve_pick_trajectory().q_ready
    _set_robot_qpos(model, data, ready_q, GRIPPER_OPEN_QPOS)
    _write_robot_state(command_path.parent / "robot_state.json", model, data, phase="startup_ready")
    _write_status(status_path, {**session_info, "status": "starting", "phase": "opening_viewer", "model_path": str(model_path)})

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _write_status(
            status_path,
            {
                **session_info,
                "status": "waiting",
                "model_path": str(model_path),
                "startup_elapsed_s": round(time.perf_counter() - startup_start, 3),
            },
        )
        while viewer.is_running():
            viewer.sync()
            command = _read_command(command_path)
            if command is None:
                time.sleep(0.05)
                continue
            if command.get("action") == "shutdown":
                _write_status(status_path, {**session_info, "status": "shutdown_requested"})
                break
            try:
                _write_status(status_path, {**session_info, "status": "running", "instruction": command.get("instruction")})
                _run_command(
                    model,
                    data,
                    viewer,
                    command,
                    output_dir=command_path.parent,
                    provider=str(command.get("provider") or args.provider),
                    model_name=args.model,
                    config_path=args.config_path,
                    camera_width=args.camera_width,
                    camera_height=args.camera_height,
                    max_parallel_vl=args.max_parallel_vl,
                    fps=args.fps,
                )
                wait_pose = _first_photo_pose(command)
                _write_status(
                    status_path,
                    {
                        **session_info,
                        "status": "running",
                        "phase": "returning_to_photo_pose_1",
                        "waiting_pose": wait_pose,
                        "last_completed_instruction": command.get("instruction"),
                    },
                )
                _return_to_photo_pose_1(model, data, viewer, wait_pose=wait_pose, fps=args.fps)
                _write_robot_state(command_path.parent / "robot_state.json", model, data, phase="waiting_photo_pose_1")
                _write_status(
                    status_path,
                    {
                        **session_info,
                        "status": "waiting",
                        "phase": "waiting_photo_pose_1",
                        "waiting_pose": wait_pose,
                        "last_completion_status": "completed",
                        "last_completed_instruction": command.get("instruction"),
                    },
                )
            except Exception as exc:
                _write_status(status_path, {**session_info, "status": "waiting", "last_error": str(exc), "recoverable": True})
                continue


def _run_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    command: dict[str, Any],
    *,
    output_dir: Path,
    provider: str,
    model_name: str | None,
    config_path: Path | None,
    camera_width: int,
    camera_height: int,
    max_parallel_vl: int,
    fps: int,
) -> None:
    poses = tuple(command.get("poses") or DEFAULT_POSES)
    language_goal = _normalize_language_goal(command)
    target_names = _target_names(language_goal)
    destination = language_goal.get("destination") or DEFAULT_DESTINATION
    sequence_items: list[PickPlaceSequenceItem] = []
    localized_targets: list[dict[str, Any]] = []
    used_tray_slots: list[int] = []
    trace_path = output_dir / "waypoint_trace.json"
    trace_records: list[dict[str, Any]] = []
    _write_waypoint_trace(trace_path, trace_records)

    use_fast_multi_target = bool(command.get("fast_multi_target_vl", provider == "openrouter_vision" and len(target_names) > 1))
    if use_fast_multi_target:
        located_all = _scan_and_localize_multi_target(
            model,
            data,
            viewer,
            target_names,
            output_dir=output_dir / "multi_target_vl",
            provider=provider,
            vl_model_name=model_name,
            config_path=config_path,
            camera_width=camera_width,
            camera_height=camera_height,
            poses=poses,
            max_parallel_vl=max_parallel_vl,
            min_accepted_views=int(command.get("min_accepted_views", 1)),
            depth_variant=str(command.get("depth_variant", "raw")),
            fps=fps,
        )
        for index, object_name in enumerate(target_names):
            located = located_all.get("results", {}).get(object_name)
            if not isinstance(located, dict) or located.get("status") != "ok":
                raise RuntimeError(f"VL localization failed for {object_name}: {located or located_all}")
            localized_targets.append(
                {
                    "index": index,
                    "object_name": object_name,
                    "located": located,
                }
            )
    else:
        _play_scan_motion(model, data, viewer, poses=poses, fps=fps)
        for index, object_name in enumerate(target_names):
            target_object = skills._object_by_name(object_name)
            sub_goal = _single_target_goal(language_goal, target_object, destination)
            sub_instruction = str(sub_goal.get("instruction") or command.get("instruction") or object_name)
            located = skills.multi_object_vl_locate(
                sub_instruction,
                output_dir / f"task_{index + 1:02d}_{object_name}",
                provider=provider,
                model=model_name,
                config_path=config_path,
                camera_width=camera_width,
                camera_height=camera_height,
                poses=poses,
                max_parallel_vl=max_parallel_vl,
                min_accepted_views=int(command.get("min_accepted_views", 1)),
                depth_variant=str(command.get("depth_variant", "raw")),
                language_goal=sub_goal,
            )
            if located.get("status") != "ok":
                raise RuntimeError(f"VL localization failed for {object_name}: {located.get('perception') or located}")
            localized_targets.append(
                {
                    "index": index,
                    "object_name": object_name,
                    "located": located,
                }
            )

    for target in localized_targets:
        index = int(target["index"])
        object_name = str(target["object_name"])
        located = target["located"]
        target_surface_world = np.asarray(located["perception"]["fusion"]["fused_target_surface_world_m"], dtype=float)
        object_half_height = float(located["object_half_height_m"])
        planned, _branch_selection = skills._plan_collection_pick_place_item(
            Path(model.xml_path) if getattr(model, "xml_path", None) else DEFAULT_D435I_MULTI_OBJECT_MODEL,
            sequence_items,
            object_name=object_name,
            target_surface_world=target_surface_world,
            place_xy=skills._destination_place_xy(destination, index),
            place_xy_candidates=skills._destination_next_place_xy_candidate(destination, used_tray_slots),
            object_half_height=object_half_height,
            placement_surface_z=skills._destination_surface_z(destination),
            fps=fps,
            preview_sequence=bool(command.get("preview_sequence", False)),
        )
        selected_slot = _branch_selection.get("selected", {}).get("place_slot_index")
        if isinstance(selected_slot, int) and selected_slot >= 0:
            used_tray_slots.append(selected_slot)
        item = PickPlaceSequenceItem(object_name=object_name, planned_trajectory=planned)
        _play_pick_place_item(model, data, viewer, sequence_items, item, fps=fps, trace_records=trace_records, trace_path=trace_path)
        sequence_items.append(item)


def _play_scan_motion(model: mujoco.MjModel, data: mujoco.MjData, viewer, *, poses: tuple[str, ...], fps: int) -> None:
    trajectory = solve_pick_trajectory()
    current = data.qpos[:ROBOT_DOF].copy()
    for pose in poses:
        target = _trajectory_pose(trajectory, pose)
        _play_joint_interpolation(model, data, viewer, current, target, gripper=GRIPPER_OPEN_QPOS, duration_s=0.65, fps=fps)
        current = target.copy()
        _hold(model, data, viewer, duration_s=0.15, fps=fps)


def _first_photo_pose(command: dict[str, Any]) -> str:
    poses = tuple(command.get("poses") or DEFAULT_POSES)
    return str(poses[0] if poses else DEFAULT_POSES[0])


def _return_to_photo_pose_1(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    *,
    wait_pose: str,
    fps: int,
) -> None:
    trajectory = solve_pick_trajectory()
    target = _trajectory_pose(trajectory, wait_pose)
    current = data.qpos[:ROBOT_DOF].copy()
    _play_joint_interpolation(
        model,
        data,
        viewer,
        current,
        target,
        gripper=GRIPPER_OPEN_QPOS,
        duration_s=0.9,
        fps=fps,
    )
    _settle_robot_q(model, data, viewer, target, gripper=GRIPPER_OPEN_QPOS, duration_s=0.8, fps=fps)


def _scan_and_localize_multi_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    target_names: list[str],
    *,
    output_dir: Path,
    provider: str,
    vl_model_name: str | None,
    config_path: Path | None,
    camera_width: int,
    camera_height: int,
    poses: tuple[str, ...],
    max_parallel_vl: int,
    min_accepted_views: int,
    depth_variant: str,
    fps: int,
) -> dict[str, Any]:
    if provider != "openrouter_vision":
        raise ValueError("fast multi-target live localization currently requires provider='openrouter_vision'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_objects = [skills._object_by_name(name) for name in target_names]
    observations: dict[str, dict[str, Any]] = {}
    regions_by_pose: dict[str, dict[str, dict[str, Any]]] = {}
    trajectory = solve_pick_trajectory()
    current = data.qpos[:ROBOT_DOF].copy()
    workers = max(1, min(int(max_parallel_vl), len(poses)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, pose in enumerate(poses):
            target_q = _trajectory_pose(trajectory, pose)
            _play_joint_interpolation(model, data, viewer, current, target_q, gripper=GRIPPER_OPEN_QPOS, duration_s=0.65, fps=fps)
            current = target_q.copy()
            _hold(model, data, viewer, duration_s=0.15, fps=fps)
            view_dir = output_dir / f"{index:02d}_{pose}"
            observation = skills.render_d435i_preview(
                view_dir,
                width=camera_width,
                height=camera_height,
                seed=7 + index,
                pose=pose,
                scene_id="gripper_multi_object_d435i",
            )
            observations[pose] = observation
            rgb_path = Path(observation["files"]["rgb"])
            if not rgb_path.is_absolute():
                rgb_path = ROOT / rgb_path
            futures[
                executor.submit(
                    skills.locate_openrouter_vision_regions,
                    rgb_path,
                    targets=target_objects,
                    output_path=view_dir / f"{pose}_multi_target_vl_overlay.png",
                    model=vl_model_name,
                    config_path=config_path,
                )
            ] = pose
        for future in as_completed(futures):
            pose = futures[future]
            try:
                regions_by_pose[pose] = future.result()
            except Exception as exc:
                regions_by_pose[pose] = {"__error__": {"error": str(exc)}}
    return skills.multi_target_vl_results_from_observations(
        target_names,
        observations=observations,
        regions_by_pose=regions_by_pose,
        provider=provider,
        depth_variant=depth_variant,
        poses=poses,
        min_accepted_views=min_accepted_views,
    )


def _play_pick_place_item(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    previous_items: list[PickPlaceSequenceItem],
    item: PickPlaceSequenceItem,
    *,
    fps: int,
    trace_records: list[dict[str, Any]] | None = None,
    trace_path: Path | None = None,
) -> None:
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    item_index = len(previous_items)
    pick_above_label, grasp_label, place_above_label, place_label = _item_waypoint_labels(item_index)
    if previous_items:
        previous_retreat = previous_items[-1].planned_trajectory.poses.q_retreat
        _settle_robot_q(model, data, viewer, previous_retreat, gripper=GRIPPER_OPEN_QPOS, duration_s=1.6, fps=fps)
        waypoints = sequence_bridge_waypoints(
            previous_retreat,
            previous_items[-1].planned_trajectory,
            item.planned_trajectory,
        )
        bridge_frames = int(np.ceil(max(0.0, skills.SEQUENCE_BRIDGE_SECONDS) * fps))
        for frame_index in range(bridge_frames):
            for step_index in range(steps_per_frame):
                step = frame_index * steps_per_frame + step_index + 1
                q_des = sequence_bridge_command_at_time(waypoints, step * model.opt.timestep, skills.SEQUENCE_BRIDGE_SECONDS)
                _set_control_targets(model, data, q_des, GRIPPER_OPEN_QPOS)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1.0 / max(float(fps), 1.0))
        _settle_robot_q(model, data, viewer, item.planned_trajectory.poses.q_pick_above, gripper=GRIPPER_OPEN_QPOS, duration_s=0.8, fps=fps)
    else:
        current = data.qpos[:ROBOT_DOF].copy()
        _play_joint_interpolation(
            model,
            data,
            viewer,
            current,
            item.planned_trajectory.poses.q_pick_above,
            gripper=GRIPPER_OPEN_QPOS,
            duration_s=1.0,
            fps=fps,
        )
        _settle_robot_q(model, data, viewer, item.planned_trajectory.poses.q_pick_above, gripper=GRIPPER_OPEN_QPOS, duration_s=0.8, fps=fps)
    _record_waypoint(trace_records, trace_path, model, data, item, item_index, pick_above_label, "pick_above_before_grasp", item.planned_trajectory.poses.q_pick_above)

    _hold_q(model, data, viewer, item.planned_trajectory.poses.q_pick_above, GRIPPER_OPEN_QPOS, PLANNED_PICK_ABOVE_DWELL_SECONDS, fps)
    _play_planned_segment(model, data, viewer, item.planned_trajectory.pick_above_to_grasp, GRIPPER_OPEN_QPOS, fps=fps)
    _settle_robot_q(model, data, viewer, item.planned_trajectory.poses.q_pick_grasp, gripper=GRIPPER_OPEN_QPOS, duration_s=0.8, fps=fps)
    _record_waypoint(trace_records, trace_path, model, data, item, item_index, grasp_label, "grasp_before_close", item.planned_trajectory.poses.q_pick_grasp)

    _play_gripper_transition(model, data, viewer, item.planned_trajectory.poses.q_pick_grasp, GRIPPER_OPEN_QPOS, GRIPPER_CLOSED_QPOS, PLANNED_PICK_CLOSE_SECONDS, fps)
    _hold_q(model, data, viewer, item.planned_trajectory.poses.q_pick_grasp, GRIPPER_CLOSED_QPOS, POST_GRASP_SETTLE_SECONDS, fps)
    _play_planned_segment(model, data, viewer, item.planned_trajectory.pick_grasp_to_lift, GRIPPER_CLOSED_QPOS, fps=fps)
    _settle_robot_q(model, data, viewer, item.planned_trajectory.poses.q_pick_lift, gripper=GRIPPER_CLOSED_QPOS, duration_s=0.8, fps=fps)
    _record_waypoint(trace_records, trace_path, model, data, item, item_index, pick_above_label, "pick_above_after_grasp", item.planned_trajectory.poses.q_pick_lift)

    _play_planned_segment(model, data, viewer, item.planned_trajectory.lift_to_place_above, GRIPPER_CLOSED_QPOS, fps=fps)
    _settle_robot_q(model, data, viewer, item.planned_trajectory.poses.q_place_above, gripper=GRIPPER_CLOSED_QPOS, duration_s=0.8, fps=fps)
    _record_waypoint(trace_records, trace_path, model, data, item, item_index, place_above_label, "place_above_before_place", item.planned_trajectory.poses.q_place_above)

    _hold_q(model, data, viewer, item.planned_trajectory.poses.q_place_above, GRIPPER_CLOSED_QPOS, PLANNED_PICK_ABOVE_DWELL_SECONDS, fps)
    _play_planned_segment(model, data, viewer, item.planned_trajectory.place_above_to_place, GRIPPER_CLOSED_QPOS, fps=fps)
    _settle_robot_q(model, data, viewer, item.planned_trajectory.poses.q_place, gripper=GRIPPER_CLOSED_QPOS, duration_s=0.8, fps=fps)
    _record_waypoint(trace_records, trace_path, model, data, item, item_index, place_label, "place_before_open", item.planned_trajectory.poses.q_place)

    _play_gripper_transition(model, data, viewer, item.planned_trajectory.poses.q_place, GRIPPER_CLOSED_QPOS, GRIPPER_OPEN_QPOS, PLACE_OPEN_SECONDS, fps)
    _play_planned_segment(model, data, viewer, item.planned_trajectory.place_to_retreat, GRIPPER_OPEN_QPOS, fps=fps)
    _settle_robot_q(model, data, viewer, item.planned_trajectory.poses.q_retreat, gripper=GRIPPER_OPEN_QPOS, duration_s=1.6, fps=fps)
    _record_waypoint(trace_records, trace_path, model, data, item, item_index, place_above_label, "place_above_after_place", item.planned_trajectory.poses.q_retreat)

    _hold_q(model, data, viewer, item.planned_trajectory.poses.q_retreat, GRIPPER_OPEN_QPOS, PLACE_RETREAT_DWELL_SECONDS, fps)


def _play_planned_segment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    segment,
    gripper: float,
    *,
    fps: int,
) -> None:
    frames = max(1, int(np.ceil(float(segment.trajectory.duration) * fps)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    for frame_index in range(frames):
        if not viewer.is_running():
            return
        for step_index in range(steps_per_frame):
            sim_time = (frame_index * steps_per_frame + step_index) * model.opt.timestep
            q_des, _, _ = segment.trajectory.sample(sim_time)
            _set_control_targets(model, data, q_des, gripper)
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1.0 / max(float(fps), 1.0))


def _hold_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    q_target: np.ndarray,
    gripper: float,
    duration_s: float,
    fps: int,
) -> None:
    frames = max(1, int(np.ceil(max(0.0, float(duration_s)) * fps)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    target = np.asarray(q_target, dtype=float)
    gripper_start = _current_gripper_command(data)
    for frame_index in range(frames):
        alpha = _smoothstep((frame_index + 1) / frames)
        gripper_cmd = float((1.0 - alpha) * gripper_start + alpha * gripper)
        for _ in range(steps_per_frame):
            _set_control_targets(model, data, target, gripper_cmd)
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1.0 / max(float(fps), 1.0))


def _play_gripper_transition(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    q_target: np.ndarray,
    gripper_start: float,
    gripper_end: float,
    duration_s: float,
    fps: int,
) -> None:
    frames = max(1, int(np.ceil(max(0.0, float(duration_s)) * fps)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    target = np.asarray(q_target, dtype=float)
    actual_start = _current_gripper_command(data)
    for frame_index in range(frames):
        alpha = _smoothstep((frame_index + 1) / frames)
        gripper = float((1.0 - alpha) * actual_start + alpha * gripper_end)
        for _ in range(steps_per_frame):
            _set_control_targets(model, data, target, gripper)
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1.0 / max(float(fps), 1.0))


def _play_joint_interpolation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    q_start: np.ndarray,
    q_end: np.ndarray,
    *,
    gripper: float,
    duration_s: float,
    fps: int,
) -> None:
    frames = max(1, int(np.ceil(duration_s * fps)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    for frame_index in range(frames):
        alpha = _smoothstep((frame_index + 1) / frames)
        q_des = (1.0 - alpha) * q_start + alpha * q_end
        for _ in range(steps_per_frame):
            _set_control_targets(model, data, q_des, gripper)
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1.0 / max(float(fps), 1.0))


def _settle_robot_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    q_target: np.ndarray,
    *,
    gripper: float,
    duration_s: float,
    fps: int,
    min_duration_s: float = 0.35,
    joint_tolerance_rad: float = 0.035,
) -> None:
    frames = max(1, int(np.ceil(duration_s * fps)))
    min_frames = max(1, int(np.ceil(min_duration_s * fps)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    target = np.asarray(q_target, dtype=float)
    gripper_start = _current_gripper_command(data)
    for frame_index in range(frames):
        alpha = _smoothstep((frame_index + 1) / frames)
        gripper_cmd = float((1.0 - alpha) * gripper_start + alpha * gripper)
        for _ in range(steps_per_frame):
            _set_control_targets(model, data, target, gripper_cmd)
            mujoco.mj_step(model, data)
        viewer.sync()
        if frame_index + 1 >= min_frames:
            joint_error = float(np.linalg.norm(data.qpos[:ROBOT_DOF] - target))
            if joint_error <= joint_tolerance_rad:
                break
        time.sleep(1.0 / max(float(fps), 1.0))
    _set_control_targets(model, data, target, gripper)


def _hold(model: mujoco.MjModel, data: mujoco.MjData, viewer, *, duration_s: float, fps: int) -> None:
    frames = max(1, int(np.ceil(duration_s * fps)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    for _ in range(frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1.0 / max(float(fps), 1.0))


def _hold_then_close(viewer, hold_seconds: float) -> None:
    end = time.time() + max(0.0, float(hold_seconds))
    while viewer.is_running() and time.time() < end:
        viewer.sync()
        time.sleep(0.05)


def _set_control_targets(model: mujoco.MjModel, data: mujoco.MjData, q_robot: np.ndarray, gripper: float) -> None:
    q_target = np.asarray(q_robot, dtype=float)
    kp = np.asarray(model.actuator_gainprm[:ROBOT_DOF, 0], dtype=float)
    safe_kp = np.where(np.abs(kp) > 1e-9, kp, 1.0)
    robot_ctrl = q_target + np.asarray(data.qfrc_bias[:ROBOT_DOF], dtype=float) / safe_kp
    for index in range(ROBOT_DOF):
        if int(model.actuator_ctrllimited[index]):
            low, high = model.actuator_ctrlrange[index]
            robot_ctrl[index] = float(np.clip(robot_ctrl[index], low, high))
    data.ctrl[:ROBOT_DOF] = robot_ctrl
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper


def _set_robot_qpos(model: mujoco.MjModel, data: mujoco.MjData, q_robot: np.ndarray, gripper: float) -> None:
    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = q_robot
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = q_robot
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper
    mujoco.mj_forward(model, data)


def _current_gripper_command(data: mujoco.MjData) -> float:
    ctrl = data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF]
    if ctrl.size:
        return float(np.mean(ctrl))
    qpos = data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF]
    return float(np.mean(qpos)) if qpos.size else GRIPPER_OPEN_QPOS


def _normalize_language_goal(command: dict[str, Any]) -> dict[str, Any]:
    goal = dict(command.get("language_goal") or {})
    goal.setdefault("status", "ok")
    goal.setdefault("instruction", command.get("instruction", "Codex live robot command"))
    goal.setdefault("action", command.get("action", "pick_and_place"))
    goal.setdefault("destination", command.get("destination") or DEFAULT_DESTINATION)
    target = dict(goal.get("target") or command.get("target") or {})
    if not target:
        raise ValueError("Command must include language_goal.target or target.")
    if target.get("object_name") and not target.get("object_names"):
        target["object_names"] = [target["object_name"]]
    target.setdefault("quantifier", "one")
    goal["target"] = target
    goal.setdefault("vl_prompt", _vl_prompt(target))
    return skills._resolve_language_goal(str(goal["instruction"]), goal)


def _single_target_goal(language_goal: dict[str, Any], target_object: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    goal = skills._single_object_language_goal(target_object, destination)
    goal["instruction"] = str(language_goal.get("instruction") or goal["instruction"])
    return goal


def _target_names(language_goal: dict[str, Any]) -> list[str]:
    target = language_goal.get("target", {})
    names = [str(name) for name in target.get("object_names") or []]
    if target.get("object_name") and str(target["object_name"]) not in names:
        names.insert(0, str(target["object_name"]))
    if names and target.get("quantifier") == "all":
        return names
    if names:
        return names[:1] if target.get("quantifier") != "all" else names

    color = target.get("color")
    shape = target.get("shape")
    matches = []
    for item in skills.object_specs_to_dicts(skills.DEFAULT_MULTI_OBJECT_SPECS):
        if color and item.get("color") != color:
            continue
        if shape and item.get("shape") != shape:
            continue
        matches.append(str(item["name"]))
    if not matches:
        raise ValueError(f"No scene objects match target constraints: {target}")
    return matches if target.get("quantifier") == "all" else matches[:1]


def _vl_prompt(target: dict[str, Any]) -> str:
    color = target.get("color") or "specified"
    shape = target.get("shape") or "object"
    shape_text = "box/cube" if shape == "box" else shape
    return f"Locate the {color} {shape_text} on the tabletop. Return one tight bbox around only the target object."


def _item_waypoint_labels(item_index: int) -> tuple[str, str, str, str]:
    pick_above = ("a", "b", "c", "d")
    grasp = ("A", "B", "C", "D")
    place_above = ("f", "g", "h", "i")
    place = ("F", "G", "H", "I")
    if 0 <= item_index < len(pick_above):
        return pick_above[item_index], grasp[item_index], place_above[item_index], place[item_index]
    suffix = str(item_index + 1)
    return f"pick_above_{suffix}", f"grasp_{suffix}", f"place_above_{suffix}", f"place_{suffix}"


def _record_waypoint(
    trace_records: list[dict[str, Any]] | None,
    trace_path: Path | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    item: PickPlaceSequenceItem,
    item_index: int,
    label: str,
    phase: str,
    q_target: np.ndarray,
) -> None:
    if trace_records is None or trace_path is None:
        return
    target = np.asarray(q_target, dtype=float)
    object_pos = None
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.object_name)
    if body_id >= 0:
        object_pos = [round(float(value), 6) for value in data.xpos[body_id]]
    trace_records.append(
        {
            "time_s": round(time.time(), 3),
            "item_index": int(item_index),
            "object_name": item.object_name,
            "label": label,
            "phase": phase,
            "q_target_deg": [round(float(value), 3) for value in np.rad2deg(target)],
            "q_actual_deg": [round(float(value), 3) for value in np.rad2deg(data.qpos[:ROBOT_DOF])],
            "joint_error_rad": round(float(np.linalg.norm(data.qpos[:ROBOT_DOF] - target)), 6),
            "gripper_ctrl": [round(float(value), 6) for value in data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF]],
            "object_pos_world_m": object_pos,
        }
    )
    _write_waypoint_trace(trace_path, trace_records)
    _write_robot_state(trace_path.with_name("robot_state.json"), model, data, phase=phase, waypoint_label=label, item_index=item_index, object_name=item.object_name)


def _write_waypoint_trace(trace_path: Path, records: list[dict[str, Any]]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = trace_path.with_name(f"{trace_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, trace_path)


def _write_robot_state(
    state_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    phase: str,
    waypoint_label: str | None = None,
    item_index: int | None = None,
    object_name: str | None = None,
) -> None:
    tcp_pos = None
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    if tcp_site >= 0:
        tcp_pos = [round(float(value), 6) for value in data.site_xpos[tcp_site]]
    objects: dict[str, list[float]] = {}
    for spec in skills.DEFAULT_MULTI_OBJECT_SPECS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.name)
        if body_id >= 0:
            objects[spec.name] = [round(float(value), 6) for value in data.xpos[body_id]]
    state = {
        "time_s": round(time.time(), 3),
        "phase": phase,
        "waypoint_label": waypoint_label,
        "item_index": item_index,
        "object_name": object_name,
        "qpos_deg": [round(float(value), 3) for value in np.rad2deg(data.qpos[:ROBOT_DOF])],
        "qpos_rad": [round(float(value), 6) for value in data.qpos[:ROBOT_DOF]],
        "ctrl_deg": [round(float(value), 3) for value in np.rad2deg(data.ctrl[:ROBOT_DOF])],
        "gripper_qpos": [round(float(value), 6) for value in data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF]],
        "gripper_ctrl": [round(float(value), 6) for value in data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF]],
        "tcp_world_m": tcp_pos,
        "objects_world_m": objects,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_name(f"{state_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, state_path)


def _read_command(command_path: Path) -> dict[str, Any] | None:
    if not command_path.exists():
        return None
    try:
        with command_path.open("r", encoding="utf-8-sig") as handle:
            command = json.load(handle)
        if not _remove_command_file(command_path):
            return None
        return command
    except json.JSONDecodeError:
        return None


def _remove_command_file(command_path: Path) -> bool:
    for _ in range(10):
        try:
            command_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            time.sleep(0.05)
    return False


def _write_status(status_path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = time.time()
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


if __name__ == "__main__":
    main()
