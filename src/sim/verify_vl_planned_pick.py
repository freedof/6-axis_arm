from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp_robot import skills
from src.sim.gripper_pick_motion import plan_pick_trajectory_from_target_3d, simulate_pick
from src.sim.gripper_pick_scene import DEFAULT_PICK_MODEL, write_pick_scene_model
from src.sim.render_gripper_pick_gif import render_gif
from src.sim.verify_render_gifs import validate_gif


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "end_to_end" / "vl_planned_pick"
DEFAULT_GIF = DEFAULT_OUTPUT_DIR / "vl_target_planned_pick_cube.gif"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the full D435i/VL/depth target_3d -> grasp pose -> "
            "RRT-Connect planned pick pipeline."
        )
    )
    parser.add_argument("--prompt", default="pick the red block")
    parser.add_argument(
        "--provider",
        default="color_fixture",
        choices=list(skills.VL_PROVIDERS),
    )
    parser.add_argument(
        "--manual-region-json",
        default=None,
        help="JSON bbox/point region for manual_region or codex_vision providers.",
    )
    parser.add_argument(
        "--manual-region-file",
        type=Path,
        default=None,
        help="Path to a JSON bbox/point region for manual_region or codex_vision providers.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=0, help="0 means derive frames from planned playback duration.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pose", default="scan", choices=list(skills.POSE_CHOICES))
    parser.add_argument("--depth-variant", default="raw", choices=list(skills.DEPTH_VARIANTS))
    parser.add_argument("--no-shortcut", action="store_true")
    args = parser.parse_args()

    output_dir = _resolve_path(args.output_dir)
    gif_path = _resolve_path(args.gif)
    manual_region = _load_manual_region(args.manual_region_json, args.manual_region_file)
    located = skills.vl_locate_object_3d(
        args.prompt,
        output_dir,
        provider=args.provider,
        manual_region=manual_region,
        model=args.model,
        config_path=args.config_path,
        width=args.camera_width,
        height=args.camera_height,
        seed=args.seed,
        pose=args.pose,
        depth_variant=args.depth_variant,
    )
    target_surface_world = np.asarray(located["target_3d"]["center_world_m"], dtype=float)
    planned = plan_pick_trajectory_from_target_3d(target_surface_world, shortcut=not args.no_shortcut)
    frames = args.frames
    if frames <= 0:
        frames = int(math.ceil(planned.total_playback_duration * args.fps))

    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    simulation = simulate_pick(model_path, frames=frames, fps=args.fps, planned_trajectory=planned)
    render_gif(
        model_path,
        gif_path,
        width=args.width,
        height=args.height,
        frames=frames,
        fps=args.fps,
        planned_trajectory=planned,
    )
    validate_gif(gif_path, expected_frames=frames, expected_size=(args.width, args.height), expected_fps=args.fps)

    status = "OK" if simulation.lifted else "FAILED"
    print("provider:", args.provider)
    print("depth_variant:", args.depth_variant)
    print("prompt:", args.prompt)
    print("depth_file:", located["depth_file"])
    print("rgb:", located["observation"]["files"]["rgb"])
    print("overlay:", located["region"].get("overlay_path"))
    print("bbox_xyxy:", located["region"].get("bbox_xyxy"))
    print("target_surface_world_m:", _round_vector(target_surface_world))
    print("planned_cube_center_m:", _round_vector(planned.cube_center))
    print("target_3d_depth_m:", located["target_3d"]["depth_m"])
    print("target_3d_valid_pixels:", located["target_3d"]["valid_pixel_count"])
    print("segments:", [_segment_summary(segment) for segment in _segments(planned)])
    print("playback_duration_s:", round(float(planned.total_playback_duration), 3))
    print("frames:", frames)
    print("gif:", gif_path)
    print("initial_cube_pos_m:", _round_vector(simulation.initial_cube_pos))
    print("final_cube_pos_m:", _round_vector(simulation.final_cube_pos))
    print("max_cube_z_m:", round(float(simulation.max_cube_z), 6))
    print("lifted:", simulation.lifted)
    print("status:", status)
    if not simulation.lifted:
        raise RuntimeError("Planned VL pick did not lift the cube.")


def _segments(planned) -> tuple[Any, Any, Any]:
    return (planned.ready_to_above, planned.above_to_grasp, planned.grasp_to_lift)


def _segment_summary(segment) -> dict[str, Any]:
    return {
        "name": segment.name,
        "reason": segment.reason,
        "iterations": int(segment.iterations),
        "raw_waypoints": len(segment.raw_path),
        "waypoints": len(segment.path),
        "duration_s": round(float(segment.trajectory.duration), 3),
    }


def _round_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_manual_region(region_json: str | None, region_file: Path | None) -> dict[str, Any] | None:
    if region_json and region_file:
        raise ValueError("Pass either --manual-region-json or --manual-region-file, not both.")
    if region_json:
        return json.loads(region_json)
    if region_file:
        path = _resolve_path(region_file)
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return None


if __name__ == "__main__":
    main()
