from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.collision import MujocoCollisionChecker
from src.planning.rrt_connect import RRTConnectConfig, path_length, plan_joint_rrt_connect, shortcut_path
from src.planning.singularity import SingularityChecker
from src.planning.trajectory import make_roundtrip_path, parameterize_joint_path
from src.robot.model import dobot_cr5_simplified
from src.sim.demo_xyz_joint_roundtrip import (
    DEFAULT_TARGET_RADIUS,
    DEFAULT_TOOL_RADIUS,
    DEFAULT_XYZ_A,
    DEFAULT_XYZ_B,
    set_target_marker,
)
from src.sim.planning_model import DEFAULT_PLANNING_MODEL, write_planning_model
from src.sim.planning_cases import (
    TOOL_Z_DOWN,
    TOOL_Z_PLUS_X,
    ContactTarget,
    PlanningCase,
    planning_cases,
    solve_planning_case,
)
from src.sim.render_mixed_orientation_gif import configure_camera
from src.sim.render_timing import DEFAULT_TARGET_DWELL_SECONDS, roundtrip_motion_alpha


DEFAULT_OUTPUT = ROOT / "outputs" / "roundtrip_planned_mixed_orientation.gif"
DEFAULT_CASE_ID = "TC-RRT-MULTI-004"


@dataclass(frozen=True)
class PlannedRoundtrip:
    case: PlanningCase
    q_a: np.ndarray
    q_b: np.ndarray
    xyz_a: np.ndarray
    xyz_b: np.ndarray
    path: tuple[np.ndarray, ...]
    raw_waypoints: int
    raw_length: float
    planned_iterations: int
    planned_reason: str


def plan_mixed_orientation_roundtrip(
    model_path: Path,
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    target_radius: float,
    tool_radius: float,
    *,
    shortcut: bool,
) -> PlannedRoundtrip:
    case = PlanningCase(
        case_id="CUSTOM-MIXED-ORIENTATION",
        name="自定义竖直到水平 +X",
        purpose="从命令行 xyz 输入构造的 mixed-orientation 规划渲染用例。",
        target_a=ContactTarget("A", xyz_a, TOOL_Z_DOWN),
        target_b=ContactTarget("B", xyz_b, TOOL_Z_PLUS_X, preferred_x=np.array([0.0, 0.0, 1.0], dtype=float)),
        render_gif=True,
    )
    return plan_case_roundtrip(model_path, case, shortcut=shortcut)


def plan_case_roundtrip(
    model_path: Path,
    case: PlanningCase,
    *,
    shortcut: bool,
) -> PlannedRoundtrip:
    robot = dobot_cr5_simplified()
    solved = solve_planning_case(case)
    q_a = solved.a.q
    q_b = solved.b.q
    checker = MujocoCollisionChecker(model_path, robot)
    singularity = SingularityChecker(robot)
    state_valid = lambda q: checker.is_state_valid(q) and singularity.is_state_valid(q)
    config = RRTConnectConfig(
        max_iterations=8000,
        step_size=0.16,
        edge_resolution=0.02,
        rng_seed=31,
        try_direct=False,
    )
    result = plan_joint_rrt_connect(robot, q_a, q_b, state_valid, config=config)
    if not result.success:
        raise RuntimeError(f"RRT-Connect failed: {result.reason}")

    path = result.path
    raw_waypoints = len(path)
    raw_length = path_length(path)
    if shortcut:
        path = shortcut_path(path, state_valid, attempts=80, edge_resolution=0.02)

    return PlannedRoundtrip(
        case=case,
        q_a=q_a,
        q_b=q_b,
        xyz_a=case.target_a.sphere_center,
        xyz_b=case.target_b.sphere_center,
        path=path,
        raw_waypoints=raw_waypoints,
        raw_length=raw_length,
        planned_iterations=result.iterations,
        planned_reason=result.reason,
    )


def interpolate_path(path: tuple[np.ndarray, ...], t: float) -> np.ndarray:
    if not path:
        raise ValueError("Path must contain at least one waypoint.")
    if len(path) == 1:
        return path[0].copy()

    t = float(np.clip(t, 0.0, 1.0))
    segment_lengths = np.array(
        [np.linalg.norm(q_to - q_from) for q_from, q_to in zip(path[:-1], path[1:])],
        dtype=float,
    )
    total = float(np.sum(segment_lengths))
    if total <= 1e-12:
        return path[0].copy()

    target = t * total
    covered = 0.0
    for q_from, q_to, length in zip(path[:-1], path[1:], segment_lengths):
        next_covered = covered + float(length)
        if target <= next_covered or q_to is path[-1]:
            alpha = 0.0 if length <= 1e-12 else (target - covered) / float(length)
            return (1.0 - alpha) * q_from + alpha * q_to
        covered = next_covered
    return path[-1].copy()


def interpolate_roundtrip_path(path: tuple[np.ndarray, ...], t: float) -> np.ndarray:
    t = t % 1.0
    if t < 0.5:
        alpha = 0.5 - 0.5 * np.cos(np.pi * (t / 0.5))
        return interpolate_path(path, alpha)
    alpha = 0.5 - 0.5 * np.cos(np.pi * ((t - 0.5) / 0.5))
    return interpolate_path(tuple(reversed(path)), alpha)


def sample_roundtrip_path(
    path: tuple[np.ndarray, ...],
    t: float,
    *,
    max_joint_velocity: float,
    max_joint_acceleration: float,
) -> np.ndarray:
    trajectory = parameterize_joint_path(
        make_roundtrip_path(path),
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )
    q, _, _ = trajectory.sample((t % 1.0) * trajectory.duration)
    return q


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
    *,
    shortcut: bool,
    max_joint_velocity: float = 0.8,
    max_joint_acceleration: float = 1.6,
    target_dwell_seconds: float = DEFAULT_TARGET_DWELL_SECONDS,
) -> PlannedRoundtrip:
    if Path(model_path) == DEFAULT_PLANNING_MODEL:
        write_planning_model(DEFAULT_PLANNING_MODEL)
    planned = plan_mixed_orientation_roundtrip(
        model_path,
        xyz_a,
        xyz_b,
        target_radius,
        tool_radius,
        shortcut=shortcut,
    )
    return render_planned_roundtrip(
        model_path,
        output_path,
        planned,
        width,
        height,
        frames,
        fps,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
        target_dwell_seconds=target_dwell_seconds,
    )


def render_case_gif(
    model_path: Path,
    output_path: Path,
    case: PlanningCase,
    width: int,
    height: int,
    frames: int,
    fps: int,
    *,
    shortcut: bool,
    max_joint_velocity: float = 0.8,
    max_joint_acceleration: float = 1.6,
    target_dwell_seconds: float = DEFAULT_TARGET_DWELL_SECONDS,
) -> PlannedRoundtrip:
    if Path(model_path) == DEFAULT_PLANNING_MODEL:
        write_planning_model(DEFAULT_PLANNING_MODEL)
    planned = plan_case_roundtrip(model_path, case, shortcut=shortcut)
    return render_planned_roundtrip(
        model_path,
        output_path,
        planned,
        width,
        height,
        frames,
        fps,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
        target_dwell_seconds=target_dwell_seconds,
    )


def render_planned_roundtrip(
    model_path: Path,
    output_path: Path,
    planned: PlannedRoundtrip,
    width: int,
    height: int,
    frames: int,
    fps: int,
    *,
    max_joint_velocity: float = 0.8,
    max_joint_acceleration: float = 1.6,
    target_dwell_seconds: float = DEFAULT_TARGET_DWELL_SECONDS,
) -> PlannedRoundtrip:

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = configure_camera()
    trajectory = parameterize_joint_path(
        planned.path,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )

    images: list[Image.Image] = []
    for frame_index in range(frames):
        alpha, reverse = roundtrip_motion_alpha(
            frame_index,
            frames,
            fps,
            target_dwell_seconds=target_dwell_seconds,
        )
        sample_time = (1.0 - alpha) * trajectory.duration if reverse else alpha * trajectory.duration
        q, _, _ = trajectory.sample(sample_time)
        data.qpos[:] = q
        data.ctrl[:] = q
        set_target_marker(model, data, "target_marker", planned.xyz_a)
        set_target_marker(model, data, "target_marker_b", planned.xyz_b)
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

    print(
        f"{planned.case.case_id}: {planned.case.name}",
    )
    print(
        "planned:",
        f"iterations={planned.planned_iterations}",
        f"raw_waypoints={planned.raw_waypoints}",
        f"raw_length={planned.raw_length:.6f}",
        f"render_waypoints={len(planned.path)}",
        f"render_length={path_length(planned.path):.6f}",
        f"motion_duration={trajectory.duration:.3f}s",
        f"gif_duration={frames / fps:.3f}s",
        f"target_dwell={target_dwell_seconds:.3f}s",
        f"reason={planned.planned_reason}",
    )
    print(f"gif: {output_path}")
    return planned


def case_by_id(case_id: str) -> PlanningCase:
    for case in planning_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in planning_cases())
    raise ValueError(f"Unknown case_id {case_id!r}. Known cases: {known}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the mixed-orientation roundtrip using an RRT-Connect path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_PLANNING_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xyz-a", nargs=3, type=float, default=DEFAULT_XYZ_A.tolist())
    parser.add_argument("--xyz-b", nargs=3, type=float, default=DEFAULT_XYZ_B.tolist())
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS)
    parser.add_argument("--tool-radius", type=float, default=DEFAULT_TOOL_RADIUS)
    parser.add_argument("--case-id", help=f"Render one predefined planning case, for example {DEFAULT_CASE_ID}.")
    parser.add_argument("--max-joint-velocity", type=float, default=0.8, help="Maximum joint velocity in rad/s for rendering.")
    parser.add_argument("--max-joint-acceleration", type=float, default=1.6, help="Maximum joint acceleration in rad/s^2 for rendering.")
    parser.add_argument("--target-dwell-seconds", type=float, default=DEFAULT_TARGET_DWELL_SECONDS)
    parser.add_argument("--shortcut", action="store_true", help="Render the shortcut-smoothed path instead of the raw RRT path.")
    args = parser.parse_args()

    if args.case_id:
        render_case_gif(
            args.model,
            args.output,
            case_by_id(args.case_id),
            args.width,
            args.height,
            args.frames,
            args.fps,
            shortcut=args.shortcut,
            max_joint_velocity=args.max_joint_velocity,
            max_joint_acceleration=args.max_joint_acceleration,
            target_dwell_seconds=args.target_dwell_seconds,
        )
    else:
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
            shortcut=args.shortcut,
            max_joint_velocity=args.max_joint_velocity,
            max_joint_acceleration=args.max_joint_acceleration,
            target_dwell_seconds=args.target_dwell_seconds,
        )


if __name__ == "__main__":
    main()
