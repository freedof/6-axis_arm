from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.obstacle_scene import (
    DEFAULT_OBSTACLE_GIF,
    DEFAULT_OBSTACLE_MODEL,
    base_planning_case,
    default_obstacle_case,
    obstacle_case_by_id,
    obstacle_cases,
    write_obstacle_model,
)
from src.sim.render_planned_roundtrip_gif import render_case_gif


def render_obstacle_case(args: argparse.Namespace, case_id: str | None = None) -> None:
    obstacle_case = default_obstacle_case() if case_id is None else obstacle_case_by_id(case_id)
    model_output = args.model_output if case_id is None and args.model_output != DEFAULT_OBSTACLE_MODEL else obstacle_case.model_output
    gif_output = args.output if case_id is None and args.output != DEFAULT_OBSTACLE_GIF else obstacle_case.gif_output
    model_path = write_obstacle_model(model_output, obstacle_case=obstacle_case)
    render_case_gif(
        model_path,
        gif_output,
        base_planning_case(obstacle_case),
        args.width,
        args.height,
        args.frames,
        args.fps,
        shortcut=not args.raw_rrt,
        max_joint_velocity=args.max_joint_velocity,
        max_joint_acceleration=args.max_joint_acceleration,
        target_dwell_seconds=args.target_dwell_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the RRT-Connect obstacle-avoidance planning demo.")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_OBSTACLE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OBSTACLE_GIF)
    parser.add_argument("--case-id", help="Render one obstacle case, for example TC-RRT-OBSTACLE-001.")
    parser.add_argument("--all", action="store_true", help="Render all predefined obstacle cases.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-joint-velocity", type=float, default=0.8)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.6)
    parser.add_argument("--target-dwell-seconds", type=float, default=1.0)
    parser.add_argument("--shortcut", action="store_true", help="Deprecated: obstacle rendering uses shortcut paths by default.")
    parser.add_argument("--raw-rrt", action="store_true", help="Render the raw RRT path for planner debugging.")
    args = parser.parse_args()

    if args.all:
        for obstacle_case in obstacle_cases():
            render_obstacle_case(args, obstacle_case.case_id)
    else:
        render_obstacle_case(args, args.case_id)


if __name__ == "__main__":
    main()
