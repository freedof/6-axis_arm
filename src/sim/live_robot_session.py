from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
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
    target_query = str(command.get("target_query") or command.get("open_vl_query") or "").strip()
    language_goal = {} if target_query else _normalize_language_goal(command)
    target_names = [] if target_query else _target_names(language_goal)
    destination = command.get("destination") or language_goal.get("destination") or DEFAULT_DESTINATION
    sequence_items: list[PickPlaceSequenceItem] = []
    localized_targets: list[dict[str, Any]] = []
    used_tray_slots: list[int] = []
    trace_path = output_dir / "waypoint_trace.json"
    trace_records: list[dict[str, Any]] = []
    _write_waypoint_trace(trace_path, trace_records)

    if target_query:
        located_all = _scan_and_localize_open_query(
            model,
            data,
            viewer,
            target_query,
            output_dir=output_dir / "open_query_vl",
            provider=provider,
            vl_model_name=model_name,
            config_path=config_path,
            camera_width=camera_width,
            camera_height=camera_height,
            poses=poses,
            max_parallel_vl=max_parallel_vl,
            min_accepted_views=int(command.get("min_accepted_views", 2)),
            depth_variant=str(command.get("depth_variant", "raw")),
            fps=fps,
        )
        if located_all.get("status") != "ok":
            raise RuntimeError(f"Open-query VL localization failed: {located_all.get('reason') or located_all}")
        for index, located in enumerate(located_all.get("targets", [])):
            localized_targets.append(
                {
                    "index": index,
                    "object_name": str(located.get("object_name") or f"vl_target_{index + 1:02d}"),
                    "located": located,
                }
            )
    elif bool(command.get("fast_multi_target_vl", provider == "openrouter_vision" and len(target_names) > 1)):
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

    source_model_path = Path(model.xml_path) if getattr(model, "xml_path", None) else DEFAULT_D435I_MULTI_OBJECT_MODEL
    if bool(command.get("parallel_fixed_seed_planning", False)) and len(localized_targets) > 1:
        if destination.get("type") == "tray" and bool(command.get("pipeline_planning", True)):
            _run_pipelined_tray_pick_place(
                model,
                data,
                viewer,
                localized_targets,
                source_model_path=source_model_path,
                destination=destination,
                command=command,
                output_dir=output_dir,
                fps=fps,
                sequence_items=sequence_items,
                trace_records=trace_records,
                trace_path=trace_path,
            )
            return
        planned_items, _branch_selections = _plan_parallel_fixed_seed_items(
            localized_targets,
            source_model_path=source_model_path,
            destination=destination,
            command=command,
            output_dir=output_dir,
            fps=fps,
        )
        for item in planned_items:
            _play_pick_place_item(model, data, viewer, sequence_items, item, fps=fps, trace_records=trace_records, trace_path=trace_path)
            sequence_items.append(item)
        return

    for target in localized_targets:
        index = int(target["index"])
        object_name = str(target["object_name"])
        located = target["located"]
        target_surface_world = np.asarray(located["perception"]["fusion"]["fused_target_surface_world_m"], dtype=float)
        object_half_height = float(located["object_half_height_m"])
        planned, _branch_selection = skills._plan_collection_pick_place_item(
            source_model_path,
            sequence_items,
            object_name=object_name,
            target_surface_world=target_surface_world,
            place_xy=skills._destination_place_xy(destination, index),
            place_xy_candidates=skills._destination_place_xy_candidates(destination, used_tray_slots),
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


def _plan_parallel_fixed_seed_items(
    localized_targets: list[dict[str, Any]],
    *,
    source_model_path: Path,
    destination: dict[str, Any],
    command: dict[str, Any],
    output_dir: Path,
    fps: int,
) -> tuple[list[PickPlaceSequenceItem], list[dict[str, Any]]]:
    fixed_seed_pose = str(command.get("fixed_seed_pose") or _first_photo_pose(command))
    fixed_seed_q = _trajectory_pose(solve_pick_trajectory(), fixed_seed_pose)
    workers = max(1, min(int(command.get("max_parallel_planning", len(localized_targets))), len(localized_targets)))
    planning_dir = output_dir / "parallel_fixed_seed_planning"
    planning_dir.mkdir(parents=True, exist_ok=True)
    ordered_targets = sorted(localized_targets, key=lambda item: int(item["index"]))

    def plan_one(target: dict[str, Any]) -> tuple[int, PickPlaceSequenceItem, dict[str, Any]]:
        index = int(target["index"])
        object_name = str(target["object_name"])
        located = target["located"]
        target_surface_world = np.asarray(located["perception"]["fusion"]["fused_target_surface_world_m"], dtype=float)
        object_half_height = float(located["object_half_height_m"])
        place_xy = skills._destination_place_xy(destination, index)
        planned, branch_selection = skills._plan_collection_pick_place_item(
            source_model_path,
            [],
            object_name=object_name,
            target_surface_world=target_surface_world,
            place_xy=place_xy,
            place_xy_candidates=[(-1, place_xy)],
            object_half_height=object_half_height,
            placement_surface_z=skills._destination_surface_z(destination),
            fps=fps,
            preview_sequence=False,
            fixed_seed_q=fixed_seed_q,
            fixed_seed_name=f"fixed:{fixed_seed_pose}",
            planning_model_path=planning_dir / f"{index:02d}_{object_name}_planning.xml",
        )
        item = PickPlaceSequenceItem(object_name=object_name, planned_trajectory=planned)
        return index, item, branch_selection

    if destination.get("type") == "tray":
        return _plan_parallel_unique_tray_slot_items(
            ordered_targets,
            source_model_path=source_model_path,
            destination=destination,
            command=command,
            output_dir=output_dir,
            planning_dir=planning_dir,
            fixed_seed_pose=fixed_seed_pose,
            fixed_seed_q=fixed_seed_q,
            fps=fps,
        )

    results: list[tuple[int, PickPlaceSequenceItem, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(plan_one, target) for target in ordered_targets]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item[0])
    planned_items = [item for _, item, _ in results]
    branch_selections = [selection for _, _, selection in results]
    _write_json_atomic(
        planning_dir / "summary.json",
        {
            "mode": "parallel_fixed_seed_planning",
            "fixed_seed_pose": fixed_seed_pose,
            "workers": workers,
            "items": [
                {
                    "index": index,
                    "object_name": item.object_name,
                    "place_center_m": [round(float(value), 6) for value in item.planned_trajectory.place_center],
                    "total_playback_duration_s": round(float(item.planned_trajectory.total_playback_duration), 3),
                    "transfer_reason": item.planned_trajectory.lift_to_place_above.reason,
                    "branch_selection": branch,
                }
                for (index, item, branch) in results
            ],
        },
    )
    return planned_items, branch_selections


def _run_pipelined_tray_pick_place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    localized_targets: list[dict[str, Any]],
    *,
    source_model_path: Path,
    destination: dict[str, Any],
    command: dict[str, Any],
    output_dir: Path,
    fps: int,
    sequence_items: list[PickPlaceSequenceItem],
    trace_records: list[dict[str, Any]],
    trace_path: Path,
) -> None:
    fixed_seed_pose = str(command.get("fixed_seed_pose") or _first_photo_pose(command))
    fixed_seed_q = _trajectory_pose(solve_pick_trajectory(), fixed_seed_pose)
    planning_dir = output_dir / "pipeline_tray_planning"
    planning_dir.mkdir(parents=True, exist_ok=True)
    ordered_targets = sorted(localized_targets, key=lambda item: int(item["index"]))
    slot_indices = list(range(len(skills.TRAY_PLACE_SLOTS)))
    if len(ordered_targets) > len(slot_indices):
        raise RuntimeError(f"Tray has {len(slot_indices)} place slots but VL produced {len(ordered_targets)} targets.")

    workers = max(1, min(int(command.get("max_parallel_planning", len(ordered_targets))), len(ordered_targets)))
    used_slots: list[int] = []
    plan_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    _write_pipeline_summary(
        planning_dir,
        status="planning_first",
        fixed_seed_pose=fixed_seed_pose,
        workers=workers,
        target_count=len(ordered_targets),
        used_slots=used_slots,
        plan_records=plan_records,
        failed_records=failed_records,
    )

    def plan_target(target: dict[str, Any], used_slots_snapshot: list[int]) -> dict[str, Any]:
        return _plan_tray_target_with_slot_fallback(
            target,
            used_slots=used_slots_snapshot,
            source_model_path=source_model_path,
            destination=destination,
            planning_dir=planning_dir,
            fixed_seed_pose=fixed_seed_pose,
            fixed_seed_q=fixed_seed_q,
            fps=fps,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future = executor.submit(plan_target, ordered_targets[0], list(used_slots))
        for item_index, target in enumerate(ordered_targets):
            result = future.result()
            if result.get("status") != "ok":
                failed_records.append(_pipeline_plan_record(result))
                _write_pipeline_summary(
                    planning_dir,
                    status="failed",
                    fixed_seed_pose=fixed_seed_pose,
                    workers=workers,
                    target_count=len(ordered_targets),
                    used_slots=used_slots,
                    plan_records=plan_records,
                    failed_records=failed_records,
                    reason=str(result.get("reason", "planning failed")),
                )
                raise RuntimeError(f"Pipeline planning failed for {target['object_name']}: {result.get('reason')}")

            selected_slot = int(result["slot_index"])
            used_slots.append(selected_slot)
            plan_records.append(_pipeline_plan_record(result))

            next_future = None
            if item_index + 1 < len(ordered_targets):
                next_target = ordered_targets[item_index + 1]
                next_future = executor.submit(plan_target, next_target, list(used_slots))

            _write_pipeline_summary(
                planning_dir,
                status="executing_with_background_planning" if next_future is not None else "executing_last",
                fixed_seed_pose=fixed_seed_pose,
                workers=workers,
                target_count=len(ordered_targets),
                used_slots=used_slots,
                plan_records=plan_records,
                failed_records=failed_records,
                executing_index=item_index,
                background_planning_index=item_index + 1 if next_future is not None else None,
            )
            _play_pick_place_item(model, data, viewer, sequence_items, result["planned_item"], fps=fps, trace_records=trace_records, trace_path=trace_path)
            sequence_items.append(result["planned_item"])

            if next_future is not None:
                future = next_future

    _write_pipeline_summary(
        planning_dir,
        status="completed",
        fixed_seed_pose=fixed_seed_pose,
        workers=workers,
        target_count=len(ordered_targets),
        used_slots=used_slots,
        plan_records=plan_records,
        failed_records=failed_records,
    )


def _plan_tray_target_with_slot_fallback(
    target: dict[str, Any],
    *,
    used_slots: list[int],
    source_model_path: Path,
    destination: dict[str, Any],
    planning_dir: Path,
    fixed_seed_pose: str,
    fixed_seed_q: np.ndarray,
    fps: int,
) -> dict[str, Any]:
    index = int(target["index"])
    object_name = str(target["object_name"])
    located = target["located"]
    target_surface_world = np.asarray(located["perception"]["fusion"]["fused_target_surface_world_m"], dtype=float)
    object_half_height = float(located["object_half_height_m"])
    attempts: list[dict[str, Any]] = []
    used = set(int(slot) for slot in used_slots)
    preferred = index % len(skills.TRAY_PLACE_SLOTS)
    slot_order = [slot for slot in [preferred, *range(len(skills.TRAY_PLACE_SLOTS))] if slot not in used]
    slot_order = list(dict.fromkeys(slot_order))
    if not slot_order:
        return {
            "status": "failed",
            "target_index": index,
            "object_name": object_name,
            "reason": "no remaining tray slots",
            "attempts": attempts,
        }

    for slot_index in slot_order:
        place_xy = skills._destination_place_xy(destination, slot_index)
        try:
            planned, branch_selection = skills._plan_collection_pick_place_item(
                source_model_path,
                [],
                object_name=object_name,
                target_surface_world=target_surface_world,
                place_xy=place_xy,
                place_xy_candidates=[(slot_index, place_xy)],
                object_half_height=object_half_height,
                placement_surface_z=skills._destination_surface_z(destination),
                fps=fps,
                preview_sequence=False,
                fixed_seed_q=fixed_seed_q,
                fixed_seed_name=f"fixed:{fixed_seed_pose}",
                planning_model_path=planning_dir / f"{index:02d}_{object_name}_slot_{slot_index}_planning.xml",
            )
        except Exception as exc:
            attempts.append(
                {
                    "status": "failed",
                    "slot_index": int(slot_index),
                    "place_xy_m": [round(float(value), 6) for value in place_xy],
                    "reason": str(exc),
                }
            )
            continue
        attempts.append(
            {
                "status": "ok",
                "slot_index": int(slot_index),
                "place_xy_m": [round(float(value), 6) for value in place_xy],
                "branch_selection": branch_selection,
                "playback_duration_s": round(float(planned.total_playback_duration), 3),
                "transfer_reason": planned.lift_to_place_above.reason,
            }
        )
        return {
            "status": "ok",
            "target_index": index,
            "object_name": object_name,
            "slot_index": int(slot_index),
            "place_xy_m": [round(float(value), 6) for value in place_xy],
            "planned_item": PickPlaceSequenceItem(object_name=object_name, planned_trajectory=planned),
            "branch_selection": branch_selection,
            "playback_duration_s": round(float(planned.total_playback_duration), 3),
            "transfer_reason": planned.lift_to_place_above.reason,
            "attempts": attempts,
        }

    return {
        "status": "failed",
        "target_index": index,
        "object_name": object_name,
        "reason": "no feasible remaining tray slot",
        "attempts": attempts,
    }


def _pipeline_plan_record(result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "status": result.get("status"),
        "index": int(result["target_index"]),
        "object_name": str(result["object_name"]),
        "attempts": result.get("attempts", []),
    }
    if result.get("status") == "ok":
        record.update(
            {
                "assigned_slot_index": int(result["slot_index"]),
                "place_xy_m": result.get("place_xy_m"),
                "place_center_m": [round(float(value), 6) for value in result["planned_item"].planned_trajectory.place_center],
                "total_playback_duration_s": result.get("playback_duration_s"),
                "transfer_reason": result.get("transfer_reason"),
                "branch_selection": result.get("branch_selection"),
            }
        )
    else:
        record["reason"] = result.get("reason")
    return record


def _write_pipeline_summary(
    planning_dir: Path,
    *,
    status: str,
    fixed_seed_pose: str,
    workers: int,
    target_count: int,
    used_slots: list[int],
    plan_records: list[dict[str, Any]],
    failed_records: list[dict[str, Any]],
    executing_index: int | None = None,
    background_planning_index: int | None = None,
    reason: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "mode": "pipelined_tray_planning",
        "status": status,
        "fixed_seed_pose": fixed_seed_pose,
        "workers": int(workers),
        "target_count": int(target_count),
        "used_slots": [int(slot) for slot in used_slots],
        "slots_unique": len(used_slots) == len(set(used_slots)),
        "items": plan_records,
        "failed_items": failed_records,
    }
    if executing_index is not None:
        payload["executing_index"] = int(executing_index)
    if background_planning_index is not None:
        payload["background_planning_index"] = int(background_planning_index)
    if reason:
        payload["reason"] = reason
    _write_json_atomic(planning_dir / "summary.json", payload)


def _plan_parallel_unique_tray_slot_items(
    localized_targets: list[dict[str, Any]],
    *,
    source_model_path: Path,
    destination: dict[str, Any],
    command: dict[str, Any],
    output_dir: Path,
    planning_dir: Path,
    fixed_seed_pose: str,
    fixed_seed_q: np.ndarray,
    fps: int,
) -> tuple[list[PickPlaceSequenceItem], list[dict[str, Any]]]:
    slot_indices = list(range(len(skills.TRAY_PLACE_SLOTS)))
    if len(localized_targets) > len(slot_indices):
        raise RuntimeError(f"Tray has {len(slot_indices)} place slots but VL produced {len(localized_targets)} targets.")

    workers = max(1, min(int(command.get("max_parallel_planning", len(localized_targets))), len(localized_targets) * len(slot_indices)))

    def plan_pair(target: dict[str, Any], slot_index: int) -> dict[str, Any]:
        index = int(target["index"])
        object_name = str(target["object_name"])
        located = target["located"]
        target_surface_world = np.asarray(located["perception"]["fusion"]["fused_target_surface_world_m"], dtype=float)
        object_half_height = float(located["object_half_height_m"])
        place_xy = skills._destination_place_xy(destination, slot_index)
        try:
            planned, branch_selection = skills._plan_collection_pick_place_item(
                source_model_path,
                [],
                object_name=object_name,
                target_surface_world=target_surface_world,
                place_xy=place_xy,
                place_xy_candidates=[(slot_index, place_xy)],
                object_half_height=object_half_height,
                placement_surface_z=skills._destination_surface_z(destination),
                fps=fps,
                preview_sequence=False,
                fixed_seed_q=fixed_seed_q,
                fixed_seed_name=f"fixed:{fixed_seed_pose}",
                planning_model_path=planning_dir / f"{index:02d}_{object_name}_slot_{slot_index}_planning.xml",
            )
        except Exception as exc:
            return {
                "status": "failed",
                "target_index": index,
                "object_name": object_name,
                "slot_index": int(slot_index),
                "place_xy_m": [round(float(value), 6) for value in place_xy],
                "reason": str(exc),
            }
        cost = _parallel_pair_cost(planned, branch_selection, target_index=index, slot_index=slot_index)
        return {
            "status": "ok",
            "target_index": index,
            "object_name": object_name,
            "slot_index": int(slot_index),
            "place_xy_m": [round(float(value), 6) for value in place_xy],
            "cost": round(float(cost), 6),
            "planned_item": PickPlaceSequenceItem(object_name=object_name, planned_trajectory=planned),
            "branch_selection": branch_selection,
            "playback_duration_s": round(float(planned.total_playback_duration), 3),
            "transfer_reason": planned.lift_to_place_above.reason,
        }

    pair_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(plan_pair, target, slot_index) for target in localized_targets for slot_index in slot_indices]
        for future in as_completed(futures):
            pair_results.append(future.result())

    ok_by_pair = {
        (int(result["target_index"]), int(result["slot_index"])): result
        for result in pair_results
        if result.get("status") == "ok"
    }
    best_assignment: list[dict[str, Any]] | None = None
    best_cost = float("inf")
    target_indices = [int(target["index"]) for target in localized_targets]
    for slots in itertools.permutations(slot_indices, len(localized_targets)):
        assignment: list[dict[str, Any]] = []
        total_cost = 0.0
        for target_index, slot_index in zip(target_indices, slots):
            result = ok_by_pair.get((target_index, int(slot_index)))
            if result is None:
                assignment = []
                break
            assignment.append(result)
            total_cost += float(result["cost"])
        if assignment and total_cost < best_cost:
            best_assignment = assignment
            best_cost = total_cost

    pair_summaries = [_parallel_pair_summary(result) for result in sorted(pair_results, key=lambda item: (int(item["target_index"]), int(item["slot_index"])))]
    if best_assignment is None:
        _write_json_atomic(
            planning_dir / "summary.json",
            {
                "mode": "parallel_fixed_seed_unique_tray_slot_planning",
                "status": "failed",
                "fixed_seed_pose": fixed_seed_pose,
                "workers": workers,
                "target_count": len(localized_targets),
                "slot_count": len(slot_indices),
                "pair_candidates": pair_summaries,
                "reason": "no feasible unique tray-slot assignment",
            },
        )
        raise RuntimeError("No feasible unique tray-slot assignment for VL target set.")

    best_assignment.sort(key=lambda item: int(item["target_index"]))
    planned_items = [result["planned_item"] for result in best_assignment]
    branch_selections = [
        {
            "mode": "parallel_unique_tray_slot_assignment",
            "selected": {
                **dict(result["branch_selection"].get("selected", {})),
                "target_index": int(result["target_index"]),
                "object_name": str(result["object_name"]),
                "assigned_slot_index": int(result["slot_index"]),
                "assignment_cost": round(float(result["cost"]), 6),
            },
            "pair_branch_selection": result["branch_selection"],
        }
        for result in best_assignment
    ]
    _write_json_atomic(
        planning_dir / "summary.json",
        {
            "mode": "parallel_fixed_seed_unique_tray_slot_planning",
            "status": "ok",
            "fixed_seed_pose": fixed_seed_pose,
            "workers": workers,
            "target_count": len(localized_targets),
            "slot_count": len(slot_indices),
            "assignment_total_cost": round(float(best_cost), 6),
            "items": [
                {
                    "index": int(result["target_index"]),
                    "object_name": str(result["object_name"]),
                    "assigned_slot_index": int(result["slot_index"]),
                    "place_center_m": [round(float(value), 6) for value in result["planned_item"].planned_trajectory.place_center],
                    "total_playback_duration_s": result["playback_duration_s"],
                    "transfer_reason": result["transfer_reason"],
                    "branch_selection": branch,
                }
                for result, branch in zip(best_assignment, branch_selections)
            ],
            "pair_candidates": pair_summaries,
        },
    )
    return planned_items, branch_selections


def _parallel_pair_cost(planned, branch_selection: dict[str, Any], *, target_index: int, slot_index: int) -> float:
    selected = branch_selection.get("selected", {}) if isinstance(branch_selection, dict) else {}
    score = selected.get("score")
    if isinstance(score, (int, float)):
        base = float(score)
    else:
        base = 0.03 * float(planned.total_playback_duration)
    return base + 0.02 * abs(int(slot_index) - int(target_index))


def _parallel_pair_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": result.get("status"),
        "target_index": int(result["target_index"]),
        "object_name": str(result["object_name"]),
        "slot_index": int(result["slot_index"]),
        "place_xy_m": result.get("place_xy_m"),
    }
    if result.get("status") == "ok":
        summary.update(
            {
                "cost": result.get("cost"),
                "playback_duration_s": result.get("playback_duration_s"),
                "transfer_reason": result.get("transfer_reason"),
                "branch_selection": result.get("branch_selection"),
            }
        )
    else:
        summary["reason"] = result.get("reason")
    return summary


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


def _scan_and_localize_open_query(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    target_query: str,
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
        raise ValueError("open-query live localization currently requires provider='openrouter_vision'.")
    output_dir.mkdir(parents=True, exist_ok=True)
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
                seed=17 + index,
                pose=pose,
                scene_id="gripper_multi_object_d435i",
            )
            observations[pose] = observation
            rgb_path = Path(observation["files"]["rgb"])
            if not rgb_path.is_absolute():
                rgb_path = ROOT / rgb_path
            futures[
                executor.submit(
                    skills.locate_openrouter_vision_category_regions,
                    rgb_path,
                    category="target",
                    target_query=target_query,
                    output_path=view_dir / f"{pose}_open_query_vl_overlay.png",
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
    return skills.open_query_vl_results_from_observations(
        target_query,
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
    _write_json_atomic(trace_path, records)


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
    _write_json_atomic(state_path, state)


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
    _write_json_atomic(status_path, status)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


if __name__ == "__main__":
    main()
