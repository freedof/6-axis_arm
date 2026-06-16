from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    xyz: np.ndarray
    euler: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class SerialRobotModel:
    name: str
    joints: tuple[JointSpec, ...]
    tool_xyz: np.ndarray
    tool_euler: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def lower_limits(self) -> np.ndarray:
        return np.array([joint.lower for joint in self.joints], dtype=float)

    @property
    def upper_limits(self) -> np.ndarray:
        return np.array([joint.upper for joint in self.joints], dtype=float)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower_limits, self.upper_limits)


def dobot_cr5_simplified() -> SerialRobotModel:
    """Kinematic chain matching assets/dobot_cr5/mjcf/cr5_simplified.xml."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    return SerialRobotModel(
        name="dobot_cr5_simplified",
        joints=(
            JointSpec("joint1", "base_link", "Link1", np.array([0.0, 0.0, 0.147]), np.array([0.0, 0.0, 0.0]), z_axis, -3.14, 3.14),
            JointSpec("joint2", "Link1", "Link2", np.array([0.0, 0.0, 0.0]), np.array([1.5708, 1.5708, 0.0]), z_axis, -3.14, 3.14),
            JointSpec("joint3", "Link2", "Link3", np.array([-0.427, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), z_axis, -2.86, 2.86),
            JointSpec("joint4", "Link3", "Link4", np.array([-0.357, 0.0, 0.141]), np.array([0.0, 0.0, -1.5708]), z_axis, -3.14, 3.14),
            JointSpec("joint5", "Link4", "Link5", np.array([0.0, -0.116, 0.0]), np.array([1.5708, 0.0, 0.0]), z_axis, -3.14, 3.14),
            JointSpec("joint6", "Link5", "Link6", np.array([0.0, 0.105, 0.0]), np.array([-1.5708, 0.0, 0.0]), z_axis, -6.28, 6.28),
        ),
        tool_xyz=np.array([0.0, 0.0, 0.08], dtype=float),
        tool_euler=np.array([0.0, 0.0, 0.0], dtype=float),
    )
