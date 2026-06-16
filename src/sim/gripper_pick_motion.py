from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robot.ik import solve_ik_multi_start
from src.robot.model import dobot_cr5_simplified
from src.sim.demo_xyz_joint_roundtrip import READY_Q, TOOL_DOWN_ROTATION, make_tool_pose
from src.sim.gripper_model import GRIPPER_OPEN_QPOS
from src.sim.gripper_pick_scene import CUBE_CENTER


ROBOT_DOF = 6
GRIPPER_DOF = 2
GRASP_CENTER_OFFSET = 0.072
GRIPPER_CLOSED_QPOS = 0.0


@dataclass(frozen=True)
class PickTrajectory:
    q_ready: np.ndarray
    q_above: np.ndarray
    q_grasp: np.ndarray
    q_lift: np.ndarray


@dataclass(frozen=True)
class PickSimulationResult:
    initial_cube_pos: np.ndarray
    final_cube_pos: np.ndarray
    max_cube_z: float
    final_gripper_qpos: np.ndarray
    lifted: bool


def solve_pick_trajectory(cube_center: np.ndarray = np.array(CUBE_CENTER, dtype=float)) -> PickTrajectory:
    robot = dobot_cr5_simplified()
    seeds = [READY_Q]
    grasp_center = np.asarray(cube_center, dtype=float)
    above_center = grasp_center + np.array([0.0, 0.0, 0.120], dtype=float)
    lift_center = grasp_center + np.array([0.0, 0.0, 0.150], dtype=float)

    q_above = _solve_gripper_center_pose(robot, above_center, seeds)
    q_grasp = _solve_gripper_center_pose(robot, grasp_center, [q_above, READY_Q])
    q_lift = _solve_gripper_center_pose(robot, lift_center, [q_grasp, q_above, READY_Q])
    return PickTrajectory(READY_Q.copy(), q_above, q_grasp, q_lift)


def simulate_pick(
    model_path: Path,
    *,
    frames: int = 160,
    fps: int = 20,
) -> PickSimulationResult:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    trajectory = solve_pick_trajectory()

    qpos0 = model.qpos0.copy()
    data.qpos[:] = qpos0
    data.qpos[:ROBOT_DOF] = trajectory.q_ready
    data.qpos[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[:ROBOT_DOF] = trajectory.q_ready
    data.ctrl[ROBOT_DOF : ROBOT_DOF + GRIPPER_DOF] = GRIPPER_OPEN_QPOS
    mujoco.mj_forward(model, data)

    cube_body_id = _body_id(model, "grasp_cube")
    initial_cube_pos = data.xpos[cube_body_id].copy()
    max_cube_z = float(initial_cube_pos[2])

    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    for frame_index in range(frames):
        sim_time = frame_index / fps
        q_des, gripper_des = command_at_time(trajectory, sim_time)
        for _ in range(steps_per_frame):
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
