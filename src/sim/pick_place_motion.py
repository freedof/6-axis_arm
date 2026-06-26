from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.collision import MujocoCollisionChecker
from src.planning.rrt_connect import RRTConnectConfig
from src.planning.singularity import SingularityChecker
from src.planning.trajectory import parameterize_joint_path
from src.robot.ik import solve_ik_multi_start
from src.robot.model import dobot_cr5_simplified
from src.sim.demo_xyz_joint_roundtrip import TOOL_DOWN_ROTATION, make_tool_pose
from src.sim.gripper_model import GRIPPER_OPEN_QPOS
from src.sim.gripper_pick_motion import (
    GRASP_CENTER_OFFSET,
    GRASP_CENTER_Z_LIFT,
    GRIPPER_CLOSED_QPOS,
    GRIPPER_DOF,
    LIFT_CENTER_Z_LIFT,
    PICK_APPROACH_MAX_JOINT_ACCELERATION,
    PICK_APPROACH_MAX_JOINT_VELOCITY,
    PICK_LIFT_MAX_JOINT_ACCELERATION,
    PICK_LIFT_MAX_JOINT_VELOCITY,
    PICK_TRAVEL_MAX_JOINT_ACCELERATION,
    PICK_TRAVEL_MAX_JOINT_VELOCITY,
    PLANNED_PICK_ABOVE_DWELL_SECONDS,
    PLANNED_PICK_CLOSE_SECONDS,
    PLANNED_PICK_FINAL_DWELL_SECONDS,
    PLANNED_PICK_READY_DWELL_SECONDS,
    ROBOT_DOF,
    ABOVE_CENTER_Z_LIFT,
    READY_CENTER_OFFSET,
    PlannedPickSegment,
    _plan_segment,
    _smoothstep,
    _solve_gripper_center_pose,
)
from src.sim.gripper_pick_scene import DEFAULT_MULTI_OBJECT_MODEL, TABLE_TOP_Z, write_multi_object_scene_model
from src.sim.planning_model import DEFAULT_PICK_PLANNING_MODEL, write_planning_model

PICK_PLACE_GRASP_Z_LIFT = 0.045
PLACE_RELEASE_Z_LIFT = PICK_PLACE_GRASP_Z_LIFT
PICK_PLACE_VERTICAL_CLEARANCE = 0.060
PICK_ABOVE_Z_LIFT = PICK_PLACE_GRASP_Z_LIFT + PICK_PLACE_VERTICAL_CLEARANCE
PLACE_ABOVE_Z_LIFT = PLACE_RELEASE_Z_LIFT + PICK_PLACE_VERTICAL_CLEARANCE
PLACE_OPEN_SECONDS = 0.7
PLACE_RETREAT_DWELL_SECONDS = 0.8
SEQUENCE_BRIDGE_SECONDS = 3.0
SEQUENCE_BRIDGE_MIN_CENTER_Z = 0.335
SEQUENCE_BRIDGE_PLACE_CLEARANCE = 0.260
PICK_PLACE_CANDIDATE_YAWS = (0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi)
SEQUENCE_BRIDGE_PICK_CLEARANCE = 0.245
SEQUENCE_BRIDGE_HOME_CENTER = np.array([0.35, -0.55, SEQUENCE_BRIDGE_MIN_CENTER_Z], dtype=float)


@dataclass(frozen=True)
class PickPlacePoses:
    q_ready: np.ndarray
    q_pick_above: np.ndarray
    q_pick_grasp: np.ndarray
    q_pick_lift: np.ndarray
    q_place_above: np.ndarray
    q_place: np.ndarray
    q_retreat: np.ndarray


@dataclass(frozen=True)
class PlannedPickPlaceTrajectory:
    target_surface_world: np.ndarray
    object_center: np.ndarray
    place_center: np.ndarray
    object_half_height: float
    poses: PickPlacePoses
    ready_to_pick_above: PlannedPickSegment
    pick_above_to_grasp: PlannedPickSegment
    pick_grasp_to_lift: PlannedPickSegment
    lift_to_place_above: PlannedPickSegment
    place_above_to_place: PlannedPickSegment
    place_to_retreat: PlannedPickSegment

    @property
    def total_motion_duration(self) -> float:
        return sum(segment.trajectory.duration for segment in pick_place_segments(self))

    @property
    def total_playback_duration(self) -> float:
        return (
            PLANNED_PICK_READY_DWELL_SECONDS
            + self.ready_to_pick_above.trajectory.duration
            + PLANNED_PICK_ABOVE_DWELL_SECONDS
            + self.pick_above_to_grasp.trajectory.duration
            + PLANNED_PICK_CLOSE_SECONDS
            + self.pick_grasp_to_lift.trajectory.duration
            + self.lift_to_place_above.trajectory.duration
            + PLANNED_PICK_ABOVE_DWELL_SECONDS
            + self.place_above_to_place.trajectory.duration
            + PLACE_OPEN_SECONDS
            + self.place_to_retreat.trajectory.duration
            + PLACE_RETREAT_DWELL_SECONDS
        )


@dataclass(frozen=True)
class PickPlaceSimulationResult:
    object_name: str
    initial_object_pos: np.ndarray
    final_object_pos: np.ndarray
    max_object_z: float
    place_center: np.ndarray
    final_gripper_qpos: np.ndarray
    lifted: bool
    placed: bool


@dataclass(frozen=True)
class PickPlaceSequenceItem:
    object_name: str
    planned_trajectory: PlannedPickPlaceTrajectory


@dataclass(frozen=True)
class PickPlaceSequenceResult:
    item_results: tuple[PickPlaceSimulationResult, ...]
    final_gripper_qpos: np.ndarray
    total_frames: int

    @property
    def placed(self) -> bool:
        return all(result.placed for result in self.item_results)


def plan_pick_place_trajectory(
    target_surface_world: np.ndarray,
    place_xy: np.ndarray | list[float] | tuple[float, float],
    *,
    object_half_height: float,
    placement_surface_z: float = TABLE_TOP_Z,
    source_model: Path = DEFAULT_MULTI_OBJECT_MODEL,
    shortcut: bool = True,
    grasp_yaw: float = 0.0,
    seed_q: np.ndarray | None = None,
) -> PlannedPickPlaceTrajectory:
    target = np.asarray(target_surface_world, dtype=float)
    place_xy_array = np.asarray(place_xy, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"Expected target_surface_world shape (3,), got {target.shape}")
    if place_xy_array.shape != (2,):
        raise ValueError(f"Expected place_xy shape (2,), got {place_xy_array.shape}")

    half_height = float(object_half_height)
    object_center = target - np.array([0.0, 0.0, half_height], dtype=float)
    place_center = np.array([place_xy_array[0], place_xy_array[1], float(placement_surface_z) + half_height], dtype=float)
    poses = solve_pick_place_poses(object_center, place_center, grasp_yaw=grasp_yaw, seed_q=seed_q)
    pick_grasp_center = object_center + np.array([0.0, 0.0, PICK_PLACE_GRASP_Z_LIFT], dtype=float)
    pick_above_center = object_center + np.array([0.0, 0.0, PICK_ABOVE_Z_LIFT], dtype=float)
    pick_lift_center = pick_above_center
    place_above_center = place_center + np.array([0.0, 0.0, PLACE_ABOVE_Z_LIFT], dtype=float)
    place_center_for_gripper = place_center + np.array([0.0, 0.0, PLACE_RELEASE_Z_LIFT], dtype=float)
    retreat_center = place_center + np.array([0.0, 0.0, PLACE_ABOVE_Z_LIFT], dtype=float)

    robot = dobot_cr5_simplified()
    source_model = write_multi_object_scene_model(source_model) if source_model == DEFAULT_MULTI_OBJECT_MODEL else Path(source_model)
    planning_model = write_planning_model(DEFAULT_PICK_PLANNING_MODEL, source_model=source_model)
    checker = MujocoCollisionChecker(planning_model, robot)
    singularity = SingularityChecker(robot)

    def state_valid(q: np.ndarray) -> bool:
        return checker.is_state_valid(q) and singularity.is_state_valid(q)

    config = RRTConnectConfig(
        max_iterations=3000,
        step_size=0.12,
        edge_resolution=0.04,
        goal_sample_rate=0.15,
        rng_seed=37,
    )
    return PlannedPickPlaceTrajectory(
        target_surface_world=target,
        object_center=object_center,
        place_center=place_center,
        object_half_height=half_height,
        poses=poses,
        ready_to_pick_above=_plan_segment(
            "ready_to_pick_above",
            robot,
            poses.q_ready,
            poses.q_pick_above,
            state_valid,
            config,
            shortcut=shortcut,
            max_joint_velocity=PICK_TRAVEL_MAX_JOINT_VELOCITY,
            max_joint_acceleration=PICK_TRAVEL_MAX_JOINT_ACCELERATION,
        ),
        pick_above_to_grasp=_cartesian_vertical_segment(
            "pick_above_to_grasp",
            robot,
            poses.q_pick_above,
            poses.q_pick_grasp,
            pick_above_center,
            pick_grasp_center,
            grasp_yaw,
            max_joint_velocity=PICK_APPROACH_MAX_JOINT_VELOCITY,
            max_joint_acceleration=PICK_APPROACH_MAX_JOINT_ACCELERATION,
        ),
        pick_grasp_to_lift=_cartesian_vertical_segment(
            "pick_grasp_to_lift",
            robot,
            poses.q_pick_grasp,
            poses.q_pick_lift,
            pick_grasp_center,
            pick_lift_center,
            grasp_yaw,
            max_joint_velocity=PICK_LIFT_MAX_JOINT_VELOCITY,
            max_joint_acceleration=PICK_LIFT_MAX_JOINT_ACCELERATION,
        ),
        lift_to_place_above=_validated_cartesian_or_planned_segment(
            "lift_to_place_above",
            robot,
            poses.q_pick_lift,
            poses.q_place_above,
            pick_lift_center,
            place_above_center,
            grasp_yaw,
            state_valid,
            config,
            shortcut=shortcut,
            max_joint_velocity=0.45,
            max_joint_acceleration=0.80,
            cartesian_reason="cartesian_high_transfer_validated",
        ),
        place_above_to_place=_cartesian_vertical_segment(
            "place_above_to_place",
            robot,
            poses.q_place_above,
            poses.q_place,
            place_above_center,
            place_center_for_gripper,
            grasp_yaw,
            max_joint_velocity=PICK_APPROACH_MAX_JOINT_VELOCITY,
            max_joint_acceleration=PICK_APPROACH_MAX_JOINT_ACCELERATION,
        ),
        place_to_retreat=_cartesian_vertical_segment(
            "place_to_retreat",
            robot,
            poses.q_place,
            poses.q_retreat,
            place_center_for_gripper,
            retreat_center,
            grasp_yaw,
            max_joint_velocity=PICK_LIFT_MAX_JOINT_VELOCITY,
            max_joint_acceleration=PICK_LIFT_MAX_JOINT_ACCELERATION,
        ),
    )


def solve_pick_place_poses(object_center: np.ndarray, place_center: np.ndarray, *, grasp_yaw: float = 0.0, seed_q: np.ndarray | None = None) -> PickPlacePoses:
    robot = dobot_cr5_simplified()
    object_center = np.asarray(object_center, dtype=float)
    place_center = np.asarray(place_center, dtype=float)
    pick_grasp_center = object_center + np.array([0.0, 0.0, PICK_PLACE_GRASP_Z_LIFT], dtype=float)
    pick_above_center = object_center + np.array([0.0, 0.0, PICK_ABOVE_Z_LIFT], dtype=float)
    pick_lift_center = pick_above_center
    ready_center = object_center + READY_CENTER_OFFSET
    place_above_center = place_center + np.array([0.0, 0.0, PLACE_ABOVE_Z_LIFT], dtype=float)
    place_center_for_gripper = place_center + np.array([0.0, 0.0, PLACE_RELEASE_Z_LIFT], dtype=float)
    retreat_center = place_center + np.array([0.0, 0.0, PLACE_ABOVE_Z_LIFT], dtype=float)

    q_ready_seeds = [] if seed_q is None else [np.asarray(seed_q, dtype=float)]
    q_ready = _solve_gripper_center_pose_with_yaw(robot, ready_center, q_ready_seeds, grasp_yaw)
    q_pick_above = _solve_gripper_center_pose_with_yaw(robot, pick_above_center, [q_ready], grasp_yaw)
    q_pick_grasp = _solve_gripper_center_pose_with_yaw(robot, pick_grasp_center, [q_pick_above, q_ready], grasp_yaw)
    q_pick_lift = q_pick_above.copy()
    q_place_above = _solve_gripper_center_pose_with_yaw(robot, place_above_center, [q_pick_lift, q_pick_above, q_ready], grasp_yaw)
    q_place = _solve_gripper_center_pose_with_yaw(robot, place_center_for_gripper, [q_place_above, q_pick_lift], grasp_yaw)
    q_retreat = q_place_above.copy()
    return PickPlacePoses(q_ready, q_pick_above, q_pick_grasp, q_pick_lift, q_place_above, q_place, q_retreat)


def _direct_segment(
    name: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    max_joint_velocity: float,
    max_joint_acceleration: float,
) -> PlannedPickSegment:
    path = (np.asarray(q_start, dtype=float), np.asarray(q_goal, dtype=float))
    trajectory = parameterize_joint_path(
        path,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )
    return PlannedPickSegment(
        name=name,
        raw_path=path,
        path=path,
        trajectory=trajectory,
        iterations=0,
        reason="direct_place_control",
    )


def _cartesian_vertical_segment(
    name: str,
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    start_center: np.ndarray,
    goal_center: np.ndarray,
    grasp_yaw: float,
    *,
    max_joint_velocity: float,
    max_joint_acceleration: float,
    waypoint_count: int = 7,
) -> PlannedPickSegment:
    return _cartesian_linear_segment(
        name,
        robot,
        q_start,
        q_goal,
        start_center,
        goal_center,
        grasp_yaw,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
        waypoint_count=waypoint_count,
        reason="cartesian_vertical_control",
    )


def _cartesian_linear_segment(
    name: str,
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    start_center: np.ndarray,
    goal_center: np.ndarray,
    grasp_yaw: float,
    *,
    max_joint_velocity: float,
    max_joint_acceleration: float,
    waypoint_count: int = 7,
    reason: str = "cartesian_linear_control",
) -> PlannedPickSegment:
    start = np.asarray(start_center, dtype=float)
    goal = np.asarray(goal_center, dtype=float)
    q_start_array = np.asarray(q_start, dtype=float)
    q_goal_array = np.asarray(q_goal, dtype=float)
    waypoints = [q_start_array]
    previous_q = q_start_array
    for alpha in np.linspace(0.0, 1.0, max(2, waypoint_count))[1:-1]:
        center = (1.0 - float(alpha)) * start + float(alpha) * goal
        q = _solve_gripper_center_pose_with_yaw(robot, center, [previous_q, q_goal_array], grasp_yaw)
        waypoints.append(q)
        previous_q = q
    waypoints.append(q_goal_array)
    path = _unwrap_revolute_waypoints(robot, tuple(waypoints))
    trajectory = parameterize_joint_path(
        path,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )
    return PlannedPickSegment(
        name=name,
        raw_path=path,
        path=path,
        trajectory=trajectory,
        iterations=0,
        reason=reason,
    )


def _validated_cartesian_or_planned_segment(
    name: str,
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    start_center: np.ndarray,
    goal_center: np.ndarray,
    grasp_yaw: float,
    state_valid,
    config: RRTConnectConfig,
    *,
    shortcut: bool,
    max_joint_velocity: float,
    max_joint_acceleration: float,
    waypoint_count: int = 7,
    cartesian_reason: str = "cartesian_linear_validated",
) -> PlannedPickSegment:
    cartesian = _cartesian_linear_segment(
        name,
        robot,
        q_start,
        q_goal,
        start_center,
        goal_center,
        grasp_yaw,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
        waypoint_count=waypoint_count,
        reason=cartesian_reason,
    )
    if _joint_path_valid(cartesian.path, state_valid, config.edge_resolution):
        return cartesian
    return _plan_segment(
        name,
        robot,
        q_start,
        q_goal,
        state_valid,
        config,
        shortcut=shortcut,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )


def _joint_path_valid(path: tuple[np.ndarray, ...] | list[np.ndarray], state_valid, edge_resolution: float) -> bool:
    waypoints = tuple(np.asarray(q, dtype=float) for q in path)
    if not waypoints:
        return False
    if any(not state_valid(q) for q in waypoints):
        return False
    for q_from, q_to in zip(waypoints[:-1], waypoints[1:]):
        distance = float(np.linalg.norm(q_to - q_from))
        steps = max(1, int(np.ceil(distance / edge_resolution)))
        for step in range(1, steps):
            alpha = step / steps
            q = (1.0 - alpha) * q_from + alpha * q_to
            if not state_valid(q):
                return False
    return True


def simulate_pick_place(
    model_path: Path,
    planned_trajectory: PlannedPickPlaceTrajectory,
    *,
    object_name: str,
    frames: int,
    fps: int,
) -> PickPlaceSimulationResult:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0.copy()
    data.qpos[:ROBOT_DOF] = planned_trajectory.poses.q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = planned_trajectory.poses.q_ready
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    object_body_id = _body_id(model, object_name)
    initial_object_pos = data.xpos[object_body_id].copy()
    max_object_z = float(initial_object_pos[2])
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    for frame_index in range(frames):
        for step_index in range(steps_per_frame):
            sim_time = (frame_index * steps_per_frame + step_index) * model.opt.timestep
            q_des, gripper_des = pick_place_command_at_time(planned_trajectory, sim_time)
            data.ctrl[:ROBOT_DOF] = q_des
            data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper_des
            mujoco.mj_step(model, data)
            max_object_z = max(max_object_z, float(data.xpos[object_body_id, 2]))

    final_object_pos = data.xpos[object_body_id].copy()
    final_gripper_qpos = data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF].copy()
    return _make_pick_place_result(
        object_name,
        initial_object_pos,
        final_object_pos,
        max_object_z,
        planned_trajectory,
        final_gripper_qpos,
    )


def simulate_pick_place_sequence(
    model_path: Path,
    items: tuple[PickPlaceSequenceItem, ...] | list[PickPlaceSequenceItem],
    *,
    fps: int,
    frames: int = 0,
    bridge_seconds: float = SEQUENCE_BRIDGE_SECONDS,
) -> PickPlaceSequenceResult:
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
    total_required = pick_place_sequence_required_frames(sequence, frames=frames, fps=fps, bridge_seconds=bridge_seconds)
    item_results: list[PickPlaceSimulationResult] = []
    frame_count = 0

    for index, item in enumerate(sequence):
        if index > 0:
            waypoints = sequence_bridge_waypoints(
                data.qpos[:ROBOT_DOF].copy(),
                sequence[index - 1].planned_trajectory,
                item.planned_trajectory,
            )
            frame_count += _run_bridge(
                model,
                data,
                waypoints=waypoints,
                fps=fps,
                steps_per_frame=steps_per_frame,
                duration_s=bridge_seconds,
            )

        body_id = _body_id(model, item.object_name)
        initial_object_pos = data.xpos[body_id].copy()
        max_object_z = float(initial_object_pos[2])
        task_frames = pick_place_required_frames(item.planned_trajectory, frames=0, fps=fps)
        for frame_index in range(task_frames):
            for step_index in range(steps_per_frame):
                sim_time = (frame_index * steps_per_frame + step_index) * model.opt.timestep
                q_des, gripper_des = pick_place_command_at_time(item.planned_trajectory, sim_time)
                data.ctrl[:ROBOT_DOF] = q_des
                data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = gripper_des
                mujoco.mj_step(model, data)
                max_object_z = max(max_object_z, float(data.xpos[body_id, 2]))
            frame_count += 1

        final_object_pos = data.xpos[body_id].copy()
        final_gripper_qpos = data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF].copy()
        item_results.append(
            _make_pick_place_result(
                item.object_name,
                initial_object_pos,
                final_object_pos,
                max_object_z,
                item.planned_trajectory,
                final_gripper_qpos,
            )
        )

    while frame_count < total_required:
        data.ctrl[:ROBOT_DOF] = sequence[-1].planned_trajectory.poses.q_retreat
        data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        frame_count += 1

    final_item_results: list[PickPlaceSimulationResult] = []
    for item, result in zip(sequence, item_results):
        body_id = _body_id(model, item.object_name)
        final_item_results.append(
            _make_pick_place_result(
                item.object_name,
                result.initial_object_pos,
                data.xpos[body_id].copy(),
                result.max_object_z,
                item.planned_trajectory,
                data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF].copy(),
            )
        )

    return PickPlaceSequenceResult(
        item_results=tuple(final_item_results),
        final_gripper_qpos=data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF].copy(),
        total_frames=frame_count,
    )
def pick_place_command_at_time(planned: PlannedPickPlaceTrajectory, t: float) -> tuple[np.ndarray, float]:
    if t < PLANNED_PICK_READY_DWELL_SECONDS:
        return planned.poses.q_ready, GRIPPER_OPEN_QPOS
    t -= PLANNED_PICK_READY_DWELL_SECONDS

    sequence = [
        (planned.ready_to_pick_above, GRIPPER_OPEN_QPOS),
        ("dwell_pick_above", PLANNED_PICK_ABOVE_DWELL_SECONDS, planned.poses.q_pick_above, GRIPPER_OPEN_QPOS),
        (planned.pick_above_to_grasp, GRIPPER_OPEN_QPOS),
        ("close", PLANNED_PICK_CLOSE_SECONDS, planned.poses.q_pick_grasp, None),
        (planned.pick_grasp_to_lift, GRIPPER_CLOSED_QPOS),
        (planned.lift_to_place_above, GRIPPER_CLOSED_QPOS),
        ("dwell_place_above", PLANNED_PICK_ABOVE_DWELL_SECONDS, planned.poses.q_place_above, GRIPPER_CLOSED_QPOS),
        (planned.place_above_to_place, GRIPPER_CLOSED_QPOS),
        ("open", PLACE_OPEN_SECONDS, planned.poses.q_place, None),
        (planned.place_to_retreat, GRIPPER_OPEN_QPOS),
        ("final", PLACE_RETREAT_DWELL_SECONDS, planned.poses.q_retreat, GRIPPER_OPEN_QPOS),
    ]
    for item in sequence:
        if isinstance(item[0], PlannedPickSegment):
            segment, gripper = item
            if t < segment.trajectory.duration:
                q, _, _ = segment.trajectory.sample(t)
                return q, float(gripper)
            t -= segment.trajectory.duration
            continue
        name, duration, q_hold, gripper = item
        if t < duration:
            if name == "close":
                alpha = _smoothstep(t / duration)
                return q_hold, float((1.0 - alpha) * GRIPPER_OPEN_QPOS + alpha * GRIPPER_CLOSED_QPOS)
            if name == "open":
                alpha = _smoothstep(t / duration)
                return q_hold, float((1.0 - alpha) * GRIPPER_CLOSED_QPOS + alpha * GRIPPER_OPEN_QPOS)
            return q_hold, float(gripper)
        t -= duration
    return planned.poses.q_retreat, GRIPPER_OPEN_QPOS


def pick_place_segments(planned: PlannedPickPlaceTrajectory) -> tuple[PlannedPickSegment, ...]:
    return (
        planned.ready_to_pick_above,
        planned.pick_above_to_grasp,
        planned.pick_grasp_to_lift,
        planned.lift_to_place_above,
        planned.place_above_to_place,
        planned.place_to_retreat,
    )


def pick_place_required_frames(planned: PlannedPickPlaceTrajectory, *, frames: int, fps: int) -> int:
    required = int(np.ceil(planned.total_playback_duration * fps))
    return max(int(frames), required)


def pick_place_sequence_required_frames(
    items: tuple[PickPlaceSequenceItem, ...] | list[PickPlaceSequenceItem],
    *,
    frames: int,
    fps: int,
    bridge_seconds: float = SEQUENCE_BRIDGE_SECONDS,
) -> int:
    sequence = tuple(items)
    if not sequence:
        return int(frames)
    required = sum(pick_place_required_frames(item.planned_trajectory, frames=0, fps=fps) for item in sequence)
    required += int(np.ceil(max(0, len(sequence) - 1) * bridge_seconds * fps))
    return max(int(frames), int(required))


def sequence_bridge_waypoints(
    q_start: np.ndarray,
    previous: PlannedPickPlaceTrajectory,
    next_item: PlannedPickPlaceTrajectory,
) -> tuple[np.ndarray, ...]:
    previous_high_center = previous.place_center.copy()
    previous_high_center[2] = max(
        float(previous.place_center[2] + SEQUENCE_BRIDGE_PLACE_CLEARANCE),
        SEQUENCE_BRIDGE_MIN_CENTER_Z,
    )

    robot = dobot_cr5_simplified()
    q_start_array = np.asarray(q_start, dtype=float).copy()
    try:
        q_previous_high = _solve_gripper_center_pose(robot, previous_high_center, [q_start_array])
        waypoints = (
            q_start_array,
            q_previous_high,
            next_item.poses.q_pick_above,
        )
    except RuntimeError:
        waypoints = (
            q_start_array,
            next_item.poses.q_pick_above,
        )
    return _unwrap_revolute_waypoints(robot, waypoints)


def sequence_bridge_command_at_time(waypoints: tuple[np.ndarray, ...] | list[np.ndarray], t: float, duration_s: float) -> np.ndarray:
    if duration_s <= 0.0:
        return np.asarray(waypoints[-1], dtype=float).copy()
    return _sample_waypoint_path(tuple(waypoints), _smoothstep(float(t) / float(duration_s)))

def _run_bridge(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    waypoints: tuple[np.ndarray, ...] | list[np.ndarray],
    fps: int,
    steps_per_frame: int,
    duration_s: float,
) -> int:
    bridge_frames = int(np.ceil(max(0.0, duration_s) * fps))
    if bridge_frames <= 0:
        return 0
    total_steps = bridge_frames * steps_per_frame
    for step_index in range(total_steps):
        q_des = sequence_bridge_command_at_time(waypoints, (step_index + 1) * model.opt.timestep, duration_s)
        data.ctrl[:ROBOT_DOF] = q_des
        data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
        mujoco.mj_step(model, data)
    return bridge_frames


def _solve_gripper_center_pose_with_yaw(
    robot,
    grasp_center: np.ndarray,
    seeds: list[np.ndarray],
    grasp_yaw: float,
) -> np.ndarray:
    rotation = _tool_down_rotation_with_yaw(grasp_yaw)
    tool0_target = grasp_center - rotation[:, 2] * GRASP_CENTER_OFFSET
    result = solve_ik_multi_start(
        robot,
        make_tool_pose(tool0_target, rotation),
        seeds=seeds,
        random_starts=48,
        position_tolerance=1e-5,
        rotation_tolerance=1e-4,
    )
    if not result.success:
        raise RuntimeError(
            "Gripper-center IK failed. "
            f"Best position error: {result.position_error:.6f} m, "
            f"rotation error: {result.rotation_error:.6f}"
        )
    return result.q


def _tool_down_rotation_with_yaw(grasp_yaw: float) -> np.ndarray:
    c = float(np.cos(grasp_yaw))
    s = float(np.sin(grasp_yaw))
    yaw_rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return yaw_rotation @ TOOL_DOWN_ROTATION

def _unwrap_revolute_waypoints(robot, waypoints: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    if not waypoints:
        return waypoints
    lower = robot.lower_limits
    upper = robot.upper_limits
    unwrapped = [np.asarray(waypoints[0], dtype=float).copy()]
    for waypoint in waypoints[1:]:
        previous = unwrapped[-1]
        adjusted = np.asarray(waypoint, dtype=float).copy()
        for joint_index, value in enumerate(adjusted):
            candidates = []
            for turns in range(-2, 3):
                candidate = float(value + turns * 2.0 * np.pi)
                if lower[joint_index] - 1e-9 <= candidate <= upper[joint_index] + 1e-9:
                    candidates.append(candidate)
            if candidates:
                adjusted[joint_index] = min(candidates, key=lambda candidate: abs(candidate - previous[joint_index]))
        unwrapped.append(adjusted)
    return tuple(unwrapped)

def _sample_waypoint_path(waypoints: tuple[np.ndarray, ...], alpha: float) -> np.ndarray:
    if len(waypoints) == 1:
        return np.asarray(waypoints[0], dtype=float).copy()
    distances = np.array(
        [float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))) for a, b in zip(waypoints[:-1], waypoints[1:])],
        dtype=float,
    )
    total = float(np.sum(distances))
    if total < 1e-9:
        return np.asarray(waypoints[-1], dtype=float).copy()
    target = float(np.clip(alpha, 0.0, 1.0)) * total
    accumulated = 0.0
    for index, distance in enumerate(distances):
        if target <= accumulated + distance or index == len(distances) - 1:
            local_alpha = 1.0 if distance < 1e-9 else (target - accumulated) / distance
            return (1.0 - local_alpha) * np.asarray(waypoints[index], dtype=float) + local_alpha * np.asarray(waypoints[index + 1], dtype=float)
        accumulated += distance
    return np.asarray(waypoints[-1], dtype=float).copy()

def _make_pick_place_result(
    object_name: str,
    initial_object_pos: np.ndarray,
    final_object_pos: np.ndarray,
    max_object_z: float,
    planned_trajectory: PlannedPickPlaceTrajectory,
    final_gripper_qpos: np.ndarray,
) -> PickPlaceSimulationResult:
    lifted = bool(max_object_z > initial_object_pos[2] + 0.050)
    place_distance_xy = float(np.linalg.norm(final_object_pos[:2] - planned_trajectory.place_center[:2]))
    moved_distance_xy = float(np.linalg.norm(final_object_pos[:2] - initial_object_pos[:2]))
    final_z_expected = planned_trajectory.place_center[2]
    placed = bool(
        lifted
        and moved_distance_xy > 0.035
        and place_distance_xy < 0.075
        and abs(float(final_object_pos[2]) - final_z_expected) < 0.045
    )
    return PickPlaceSimulationResult(
        object_name=object_name,
        initial_object_pos=initial_object_pos.copy(),
        final_object_pos=final_object_pos.copy(),
        max_object_z=float(max_object_z),
        place_center=planned_trajectory.place_center.copy(),
        final_gripper_qpos=final_gripper_qpos.copy(),
        lifted=lifted,
        placed=placed,
    )
def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {name}")
    return body_id
