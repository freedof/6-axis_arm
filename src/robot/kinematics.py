from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import SerialRobotModel


@dataclass(frozen=True)
class ForwardKinematicsResult:
    pose: np.ndarray
    joint_origins: np.ndarray
    joint_axes: np.ndarray

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        return self.pose[:3, :3]


def rot_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def euler_xyz_matrix(euler: np.ndarray) -> np.ndarray:
    """MuJoCo-style xyz euler for this MJCF chain: Rx(x) @ Ry(y) @ Rz(z)."""
    x, y, z = euler
    return rot_x(x) @ rot_y(y) @ rot_z(z)


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    t = np.eye(4, dtype=float)
    if rotation is not None:
        t[:3, :3] = rotation
    if translation is not None:
        t[:3, 3] = translation
    return t


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Small-angle orientation error that maps current rotation toward target."""
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


def forward_kinematics(robot: SerialRobotModel, q: np.ndarray) -> ForwardKinematicsResult:
    q = np.asarray(q, dtype=float)
    if q.shape != (robot.dof,):
        raise ValueError(f"Expected q shape {(robot.dof,)}, got {q.shape}")

    pose = np.eye(4, dtype=float)
    origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []

    for joint, qi in zip(robot.joints, q):
        pose = pose @ transform(euler_xyz_matrix(joint.euler), joint.xyz)
        origins.append(pose[:3, 3].copy())
        axes.append((pose[:3, :3] @ joint.axis).copy())
        pose = pose @ transform(rot_z(float(qi)))

    pose = pose @ transform(euler_xyz_matrix(robot.tool_euler), robot.tool_xyz)
    return ForwardKinematicsResult(
        pose=pose,
        joint_origins=np.vstack(origins),
        joint_axes=np.vstack(axes),
    )


def geometric_jacobian(robot: SerialRobotModel, q: np.ndarray) -> np.ndarray:
    fk = forward_kinematics(robot, q)
    end = fk.position
    jac = np.zeros((6, robot.dof), dtype=float)

    for index in range(robot.dof):
        axis = fk.joint_axes[index]
        origin = fk.joint_origins[index]
        jac[:3, index] = np.cross(axis, end - origin)
        jac[3:, index] = axis

    return jac


def pose_error(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    error = np.zeros(6, dtype=float)
    error[:3] = target_pose[:3, 3] - current_pose[:3, 3]
    error[3:] = rotation_error(current_pose[:3, :3], target_pose[:3, :3])
    return error
