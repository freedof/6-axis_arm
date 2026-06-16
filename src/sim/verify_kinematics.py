from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robot.ik import solve_ik_multi_start
from src.robot.kinematics import forward_kinematics, geometric_jacobian, pose_error
from src.robot.model import dobot_cr5_simplified
from src.robot.mujoco_compare import mujoco_tool_pose


def finite_difference_position_jacobian(robot, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    jac = np.zeros((3, robot.dof), dtype=float)
    for i in range(robot.dof):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += eps
        q_minus[i] -= eps
        p_plus = forward_kinematics(robot, q_plus).position
        p_minus = forward_kinematics(robot, q_minus).position
        jac[:, i] = (p_plus - p_minus) / (2.0 * eps)
    return jac


def main() -> None:
    robot = dobot_cr5_simplified()
    samples = [
        np.zeros(6),
        np.array([0.0, -0.6, 0.85, 0.0, -0.8, 0.0]),
        np.array([0.5, -0.8, 0.7, 0.4, -0.6, 1.2]),
        np.array([-0.4, 0.35, -0.55, 1.1, 0.45, -1.4]),
    ]

    print(f"robot: {robot.name}")
    print("joints:", ", ".join(robot.joint_names))

    max_position_delta = 0.0
    max_rotation_delta = 0.0
    for idx, q in enumerate(samples, start=1):
        fk = forward_kinematics(robot, q)
        mj_pose = mujoco_tool_pose(q)
        err = pose_error(fk.pose, mj_pose)
        pos_delta = float(np.linalg.norm(err[:3]))
        rot_delta = float(np.linalg.norm(err[3:]))
        max_position_delta = max(max_position_delta, pos_delta)
        max_rotation_delta = max(max_rotation_delta, rot_delta)
        print(f"sample {idx}: position_delta={pos_delta:.8f}, rotation_delta={rot_delta:.8f}")

    jac_q = samples[2]
    jac = geometric_jacobian(robot, jac_q)
    fd_jac = finite_difference_position_jacobian(robot, jac_q)
    jac_delta = float(np.max(np.abs(jac[:3] - fd_jac)))
    print(f"jacobian_position_max_delta={jac_delta:.8e}")

    target_pose = forward_kinematics(robot, samples[2]).pose
    result = solve_ik_multi_start(
        robot,
        target_pose,
        seeds=[samples[1]],
        random_starts=8,
        position_tolerance=1e-5,
        rotation_tolerance=1e-4,
    )
    print(
        "ik_result:",
        f"success={result.success}",
        f"iterations={result.iterations}",
        f"position_error={result.position_error:.8e}",
        f"rotation_error={result.rotation_error:.8e}",
        f"q={np.round(result.q, 5)}",
    )

    if max_position_delta > 1e-6 or max_rotation_delta > 1e-5:
        raise RuntimeError("FK does not match MuJoCo within tolerance.")
    if jac_delta > 1e-5:
        raise RuntimeError("Jacobian position block does not match finite difference.")
    if not result.success:
        raise RuntimeError("IK failed to recover a reachable pose.")

    print("status: OK")


if __name__ == "__main__":
    main()
