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
from src.planning.rrt_connect import RRTConnectConfig, plan_joint_rrt_connect, shortcut_path
from src.planning.singularity import SingularityChecker
from src.planning.trajectory import JointTrajectory, parameterize_joint_path
from src.robot.ik import solve_ik_multi_start
from src.robot.model import dobot_cr5_simplified
from src.sim.demo_xyz_joint_roundtrip import READY_Q, TOOL_DOWN_ROTATION, make_tool_pose
from src.sim.gripper_model import GRIPPER_OPEN_QPOS
from src.sim.gripper_pick_scene import CUBE_CENTER, CUBE_HALF_SIZE, DEFAULT_PICK_MODEL, write_pick_scene_model
from src.sim.planning_model import DEFAULT_PICK_PLANNING_MODEL, write_planning_model


ROBOT_DOF = 6
GRIPPER_DOF = 2
GRASP_CENTER_OFFSET = 0.072
GRASP_CENTER_Z_LIFT = 0.070
READY_CENTER_OFFSET = np.array([-0.090, -0.150, 0.185], dtype=float)
ABOVE_CENTER_Z_LIFT = 0.120
LIFT_CENTER_Z_LIFT = 0.150
GRIPPER_CLOSED_QPOS = 0.0
PLANNED_PICK_READY_DWELL_SECONDS = 0.6
PLANNED_PICK_ABOVE_DWELL_SECONDS = 0.4
PLANNED_PICK_CLOSE_SECONDS = 0.8
PLANNED_PICK_FINAL_DWELL_SECONDS = 1.0
PICK_TRAVEL_MAX_JOINT_VELOCITY = 0.75
PICK_TRAVEL_MAX_JOINT_ACCELERATION = 1.40
PICK_APPROACH_MAX_JOINT_VELOCITY = 0.10
PICK_APPROACH_MAX_JOINT_ACCELERATION = 0.18
PICK_LIFT_MAX_JOINT_VELOCITY = 0.40
PICK_LIFT_MAX_JOINT_ACCELERATION = 0.70


@dataclass(frozen=True)
class PickTrajectory:
    q_ready: np.ndarray
    q_above: np.ndarray
    q_grasp: np.ndarray
    q_lift: np.ndarray


@dataclass(frozen=True)
class PlannedPickSegment:
    name: str
    raw_path: tuple[np.ndarray, ...]
    path: tuple[np.ndarray, ...]
    trajectory: JointTrajectory
    iterations: int
    reason: str


@dataclass(frozen=True)
class PlannedPickTrajectory:
    target_surface_world: np.ndarray
    cube_center: np.ndarray
    poses: PickTrajectory
    ready_to_above: PlannedPickSegment
    above_to_grasp: PlannedPickSegment
    grasp_to_lift: PlannedPickSegment

    @property
    def total_motion_duration(self) -> float:
        return (
            self.ready_to_above.trajectory.duration
            + self.above_to_grasp.trajectory.duration
            + self.grasp_to_lift.trajectory.duration
        )

    @property
    def total_playback_duration(self) -> float:
        return (
            PLANNED_PICK_READY_DWELL_SECONDS
            + self.ready_to_above.trajectory.duration
            + PLANNED_PICK_ABOVE_DWELL_SECONDS
            + self.above_to_grasp.trajectory.duration
            + PLANNED_PICK_CLOSE_SECONDS
            + self.grasp_to_lift.trajectory.duration
            + PLANNED_PICK_FINAL_DWELL_SECONDS
        )


@dataclass(frozen=True)
class PickSimulationResult:
    initial_cube_pos: np.ndarray
    final_cube_pos: np.ndarray
    max_cube_z: float
    final_gripper_qpos: np.ndarray
    lifted: bool


def cube_center_from_target_surface(
    target_surface_world: np.ndarray,
    *,
    cube_half_height: float = CUBE_HALF_SIZE[2],
) -> np.ndarray:
    target = np.asarray(target_surface_world, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"Expected target_surface_world shape (3,), got {target.shape}")
    return target - np.array([0.0, 0.0, float(cube_half_height)], dtype=float)


def solve_pick_trajectory(cube_center: np.ndarray = np.array(CUBE_CENTER, dtype=float)) -> PickTrajectory:
    robot = dobot_cr5_simplified()
    seeds = [READY_Q]
    cube_center = np.asarray(cube_center, dtype=float)
    grasp_center = cube_center + np.array([0.0, 0.0, GRASP_CENTER_Z_LIFT], dtype=float)
    above_center = cube_center + np.array([0.0, 0.0, ABOVE_CENTER_Z_LIFT], dtype=float)
    lift_center = cube_center + np.array([0.0, 0.0, LIFT_CENTER_Z_LIFT], dtype=float)
    ready_center = cube_center + READY_CENTER_OFFSET

    q_ready = _solve_gripper_center_pose(robot, ready_center, seeds)
    q_above = _solve_gripper_center_pose(robot, above_center, [q_ready, READY_Q])
    q_grasp = _solve_gripper_center_pose(robot, grasp_center, [q_above, READY_Q])
    q_lift = _solve_gripper_center_pose(robot, lift_center, [q_grasp, q_above, READY_Q])
    return PickTrajectory(q_ready, q_above, q_grasp, q_lift)


def plan_pick_trajectory_from_target_3d(
    target_surface_world: np.ndarray,
    *,
    shortcut: bool = True,
    travel_max_joint_velocity: float = PICK_TRAVEL_MAX_JOINT_VELOCITY,
    travel_max_joint_acceleration: float = PICK_TRAVEL_MAX_JOINT_ACCELERATION,
    approach_max_joint_velocity: float = PICK_APPROACH_MAX_JOINT_VELOCITY,
    approach_max_joint_acceleration: float = PICK_APPROACH_MAX_JOINT_ACCELERATION,
    lift_max_joint_velocity: float = PICK_LIFT_MAX_JOINT_VELOCITY,
    lift_max_joint_acceleration: float = PICK_LIFT_MAX_JOINT_ACCELERATION,
) -> PlannedPickTrajectory:
    target = np.asarray(target_surface_world, dtype=float)
    cube_center = cube_center_from_target_surface(target)
    poses = solve_pick_trajectory(cube_center)
    robot = dobot_cr5_simplified()
    pick_scene_model = write_pick_scene_model(DEFAULT_PICK_MODEL)
    pick_planning_model = write_planning_model(DEFAULT_PICK_PLANNING_MODEL, source_model=pick_scene_model)
    checker = MujocoCollisionChecker(pick_planning_model, robot)
    singularity = SingularityChecker(robot)

    def state_valid(q: np.ndarray) -> bool:
        return checker.is_state_valid(q) and singularity.is_state_valid(q)

    config = RRTConnectConfig(
        max_iterations=2500,
        step_size=0.12,
        edge_resolution=0.04,
        goal_sample_rate=0.15,
        rng_seed=31,
    )
    return PlannedPickTrajectory(
        target_surface_world=target,
        cube_center=cube_center,
        poses=poses,
        ready_to_above=_plan_segment(
            "ready_to_above",
            robot,
            poses.q_ready,
            poses.q_above,
            state_valid,
            config,
            shortcut=shortcut,
            max_joint_velocity=travel_max_joint_velocity,
            max_joint_acceleration=travel_max_joint_acceleration,
        ),
        above_to_grasp=_plan_segment(
            "above_to_grasp",
            robot,
            poses.q_above,
            poses.q_grasp,
            state_valid,
            config,
            shortcut=shortcut,
            max_joint_velocity=approach_max_joint_velocity,
            max_joint_acceleration=approach_max_joint_acceleration,
        ),
        grasp_to_lift=_plan_segment(
            "grasp_to_lift",
            robot,
            poses.q_grasp,
            poses.q_lift,
            state_valid,
            config,
            shortcut=shortcut,
            max_joint_velocity=lift_max_joint_velocity,
            max_joint_acceleration=lift_max_joint_acceleration,
        ),
    )


def simulate_pick(
    model_path: Path,
    *,
    frames: int = 160,
    fps: int = 20,
    trajectory: PickTrajectory | None = None,
    planned_trajectory: PlannedPickTrajectory | None = None,
) -> PickSimulationResult:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if trajectory is not None and planned_trajectory is not None:
        raise ValueError("Pass either trajectory or planned_trajectory, not both.")
    if trajectory is None and planned_trajectory is None:
        trajectory = solve_pick_trajectory()
    q_ready = planned_trajectory.poses.q_ready if planned_trajectory is not None else trajectory.q_ready

    qpos0 = model.qpos0.copy()
    data.qpos[:] = qpos0
    data.qpos[:ROBOT_DOF] = q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = q_ready
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    cube_body_id = _body_id(model, "grasp_cube")
    initial_cube_pos = data.xpos[cube_body_id].copy()
    max_cube_z = float(initial_cube_pos[2])

    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
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
            max_cube_z = max(max_cube_z, float(data.xpos[cube_body_id, 2]))

    final_cube_pos = data.xpos[cube_body_id].copy()
    final_gripper_qpos = data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF].copy()
    lifted = bool(max_cube_z > initial_cube_pos[2] + 0.055 and final_cube_pos[2] > initial_cube_pos[2] + 0.035)
    return PickSimulationResult(
        initial_cube_pos=initial_cube_pos,
        final_cube_pos=final_cube_pos,
        max_cube_z=max_cube_z,
        final_gripper_qpos=final_gripper_qpos,
        lifted=lifted,
    )


def planned_command_at_time(planned: PlannedPickTrajectory, t: float) -> tuple[np.ndarray, float]:
    if t < PLANNED_PICK_READY_DWELL_SECONDS:
        return planned.poses.q_ready, GRIPPER_OPEN_QPOS

    t -= PLANNED_PICK_READY_DWELL_SECONDS
    if t < planned.ready_to_above.trajectory.duration:
        q, _, _ = planned.ready_to_above.trajectory.sample(t)
        return q, GRIPPER_OPEN_QPOS

    t -= planned.ready_to_above.trajectory.duration
    if t < PLANNED_PICK_ABOVE_DWELL_SECONDS:
        return planned.poses.q_above, GRIPPER_OPEN_QPOS

    t -= PLANNED_PICK_ABOVE_DWELL_SECONDS
    if t < planned.above_to_grasp.trajectory.duration:
        q, _, _ = planned.above_to_grasp.trajectory.sample(t)
        return q, GRIPPER_OPEN_QPOS

    t -= planned.above_to_grasp.trajectory.duration
    if t < PLANNED_PICK_CLOSE_SECONDS:
        close_alpha = _smoothstep(t / PLANNED_PICK_CLOSE_SECONDS)
        gripper = (1.0 - close_alpha) * GRIPPER_OPEN_QPOS + close_alpha * GRIPPER_CLOSED_QPOS
        return planned.poses.q_grasp, float(gripper)

    t -= PLANNED_PICK_CLOSE_SECONDS
    if t < planned.grasp_to_lift.trajectory.duration:
        q, _, _ = planned.grasp_to_lift.trajectory.sample(t)
        return q, GRIPPER_CLOSED_QPOS
    return planned.poses.q_lift, GRIPPER_CLOSED_QPOS


def command_at_time(trajectory: PickTrajectory, t: float) -> tuple[np.ndarray, float]:
    if t < 0.6:
        return trajectory.q_ready, GRIPPER_OPEN_QPOS
    if t < 2.0:
        return _smooth_joint(trajectory.q_ready, trajectory.q_above, (t - 0.6) / 1.4), GRIPPER_OPEN_QPOS
    if t < 2.8:
        return _smooth_joint(trajectory.q_above, trajectory.q_grasp, (t - 2.0) / 0.8), GRIPPER_OPEN_QPOS
    if t < 4.0:
        close_alpha = _smoothstep((t - 2.8) / 0.8)
        gripper = (1.0 - close_alpha) * GRIPPER_OPEN_QPOS + close_alpha * GRIPPER_CLOSED_QPOS
        return trajectory.q_grasp, float(gripper)
    if t < 5.5:
        return _smooth_joint(trajectory.q_grasp, trajectory.q_lift, (t - 4.0) / 1.5), GRIPPER_CLOSED_QPOS
    return trajectory.q_lift, GRIPPER_CLOSED_QPOS


def _plan_segment(
    name: str,
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    state_valid,
    config: RRTConnectConfig,
    *,
    shortcut: bool,
    max_joint_velocity: float,
    max_joint_acceleration: float,
) -> PlannedPickSegment:
    result = plan_joint_rrt_connect(robot, q_start, q_goal, state_valid, config=config)
    if not result.success:
        raise RuntimeError(f"RRT-Connect failed for {name}: {result.reason}")
    path = result.path
    if shortcut:
        path = shortcut_path(path, state_valid, attempts=80, edge_resolution=config.edge_resolution, rng_seed=41)
    trajectory = parameterize_joint_path(
        path,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )
    return PlannedPickSegment(
        name=name,
        raw_path=result.path,
        path=path,
        trajectory=trajectory,
        iterations=result.iterations,
        reason=result.reason,
    )


def _solve_gripper_center_pose(
    robot,
    grasp_center: np.ndarray,
    seeds: list[np.ndarray],
) -> np.ndarray:
    tool0_target = grasp_center - TOOL_DOWN_ROTATION[:, 2] * GRASP_CENTER_OFFSET
    result = solve_ik_multi_start(
        robot,
        make_tool_pose(tool0_target, TOOL_DOWN_ROTATION),
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


def _smooth_joint(q_from: np.ndarray, q_to: np.ndarray, alpha: float) -> np.ndarray:
    alpha = _smoothstep(alpha)
    return (1.0 - alpha) * q_from + alpha * q_to


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {name}")
    return body_id
