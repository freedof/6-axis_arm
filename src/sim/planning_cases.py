from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robot.ik import solve_ik_multi_start
from src.robot.kinematics import forward_kinematics
from src.robot.model import dobot_cr5_simplified
from src.sim.demo_xyz_joint_roundtrip import (
    DEFAULT_TARGET_RADIUS,
    DEFAULT_TOOL_RADIUS,
    READY_Q,
    TOOL_DOWN_ROTATION,
    make_tool_pose,
    normalize,
    rotation_from_tool_z,
    tool_target_from_sphere_center,
)


TOOL_Z_DOWN = np.array([0.0, 0.0, -1.0], dtype=float)
TOOL_Z_PLUS_X = np.array([1.0, 0.0, 0.0], dtype=float)
TOOL_Z_MINUS_X = np.array([-1.0, 0.0, 0.0], dtype=float)
TOOL_Z_PLUS_Y = np.array([0.0, 1.0, 0.0], dtype=float)
TOOL_Z_MINUS_Y = np.array([0.0, -1.0, 0.0], dtype=float)


@dataclass(frozen=True)
class ContactTarget:
    name: str
    sphere_center: np.ndarray
    tool_z: np.ndarray
    preferred_x: np.ndarray | None = None


@dataclass(frozen=True)
class PlanningCase:
    case_id: str
    name: str
    purpose: str
    target_a: ContactTarget
    target_b: ContactTarget
    render_gif: bool = False
    ik_seed_a: np.ndarray | None = None
    ik_seed_b: np.ndarray | None = None


@dataclass(frozen=True)
class SolvedContactTarget:
    target: ContactTarget
    tool_target: np.ndarray
    rotation: np.ndarray
    q: np.ndarray
    position_error: float
    tool_z_angle_deg: float


@dataclass(frozen=True)
class SolvedPlanningCase:
    case: PlanningCase
    a: SolvedContactTarget
    b: SolvedContactTarget


def planning_cases() -> tuple[PlanningCase, ...]:
    return (
        PlanningCase(
            case_id="TC-RRT-MULTI-001",
            name="近距离竖直接触",
            purpose="验证短距离同姿态规划链路。",
            target_a=_target("A", [0.35, -0.55, 0.20], TOOL_Z_DOWN),
            target_b=_target("B", [0.31, -0.53, 0.24], TOOL_Z_DOWN),
            render_gif=True,
        ),
        PlanningCase(
            case_id="TC-RRT-MULTI-002",
            name="远距离竖直接触",
            purpose="验证较大关节运动范围下的同姿态规划。",
            target_a=_target("A", [0.42, -0.50, 0.18], TOOL_Z_DOWN),
            target_b=_target("B", [0.15, -0.63, 0.34], TOOL_Z_DOWN),
        ),
        PlanningCase(
            case_id="TC-RRT-MULTI-003",
            name="竖直高度变化",
            purpose="验证 XY 接近但高度差明显时的肘部和腕部配合。",
            target_a=_target("A", [0.31, -0.56, 0.20], TOOL_Z_DOWN),
            target_b=_target("B", [0.31, -0.56, 0.38], TOOL_Z_DOWN),
        ),
        PlanningCase(
            case_id="TC-RRT-MULTI-004",
            name="竖直到水平 +X",
            purpose="验证位置变化叠加末端姿态从竖直到水平的规划。",
            target_a=_target("A", [0.35, -0.55, 0.20], TOOL_Z_DOWN),
            target_b=_target("B", [0.20, -0.60, 0.30], TOOL_Z_PLUS_X, preferred_x=[0.0, 0.0, 1.0]),
            render_gif=True,
        ),
        PlanningCase(
            case_id="TC-RRT-MULTI-005",
            name="左右跨越水平接触",
            purpose="验证 base 关节大角度旋转和水平接触姿态。",
            target_a=_target("A", [0.26, -0.55, 0.42], TOOL_Z_PLUS_X, preferred_x=[0.0, 0.0, 1.0]),
            target_b=_target("B", [0.26, 0.42, 0.42], TOOL_Z_MINUS_X, preferred_x=[0.0, 0.0, 1.0]),
            render_gif=True,
            ik_seed_a=np.array([-0.184, -0.984, 1.573, -2.161, 0.184, -1.571], dtype=float),
            ik_seed_b=np.array([2.067, -0.801, 1.234, -2.003, 1.074, -1.571], dtype=float),
        ),
        PlanningCase(
            case_id="TC-RRT-MULTI-006",
            name="接近工作空间边界",
            purpose="验证接近可达边缘但仍可达目标下的 IK 和规划稳定性。",
            target_a=_target("A", [0.52, -0.36, 0.22], TOOL_Z_DOWN),
            target_b=_target("B", [0.46, -0.16, 0.35], TOOL_Z_PLUS_Y, preferred_x=[0.0, 0.0, 1.0]),
        ),
    )


def solve_planning_case(
    case: PlanningCase,
    target_radius: float = DEFAULT_TARGET_RADIUS,
    tool_radius: float = DEFAULT_TOOL_RADIUS,
) -> SolvedPlanningCase:
    seeds_a = [case.ik_seed_a, READY_Q] if case.ik_seed_a is not None else [READY_Q]
    solved_a = solve_contact_target(case.target_a, target_radius, tool_radius, seeds=seeds_a)
    seeds_b = [case.ik_seed_b, solved_a.q, READY_Q] if case.ik_seed_b is not None else [solved_a.q, READY_Q]
    solved_b = solve_contact_target(case.target_b, target_radius, tool_radius, seeds=seeds_b)
    return SolvedPlanningCase(
        case=case,
        a=solved_a,
        b=solved_b,
    )


def solve_contact_target(
    target: ContactTarget,
    target_radius: float = DEFAULT_TARGET_RADIUS,
    tool_radius: float = DEFAULT_TOOL_RADIUS,
    *,
    seeds: list[np.ndarray] | None = None,
) -> SolvedContactTarget:
    approach = -normalize(target.tool_z)
    tool_target = tool_target_from_sphere_center(
        target.sphere_center,
        approach,
        target_radius,
        tool_radius,
    )
    rotation = rotation_for_tool_z(target.tool_z, target.preferred_x)
    result = solve_ik_multi_start(
        dobot_cr5_simplified(),
        make_tool_pose(tool_target, rotation),
        seeds=seeds,
        position_tolerance=1e-5,
        rotation_tolerance=1e-4,
        random_starts=48,
    )
    if not result.success:
        raise RuntimeError(
            f"{target.name} IK failed. "
            f"Best position error: {result.position_error:.6f} m, "
            f"rotation error: {result.rotation_error:.6f}"
        )
    q = result.q

    fk = forward_kinematics(dobot_cr5_simplified(), q)
    position_error = float(np.linalg.norm(fk.position - tool_target))
    alignment = float(np.clip(np.dot(fk.rotation[:, 2], normalize(target.tool_z)), -1.0, 1.0))
    tool_z_angle_deg = float(np.rad2deg(np.arccos(alignment)))
    return SolvedContactTarget(
        target=target,
        tool_target=tool_target,
        rotation=rotation,
        q=q,
        position_error=position_error,
        tool_z_angle_deg=tool_z_angle_deg,
    )


def rotation_for_tool_z(tool_z: np.ndarray, preferred_x: np.ndarray | None = None) -> np.ndarray:
    tool_z = normalize(tool_z)
    if np.linalg.norm(tool_z - TOOL_Z_DOWN) <= 1e-9:
        return TOOL_DOWN_ROTATION.copy()
    preferred = None if preferred_x is None else np.asarray(preferred_x, dtype=float)
    return rotation_from_tool_z(tool_z, preferred_x=preferred)


def pose_for_contact_target(
    target: ContactTarget,
    target_radius: float = DEFAULT_TARGET_RADIUS,
    tool_radius: float = DEFAULT_TOOL_RADIUS,
) -> np.ndarray:
    approach = -normalize(target.tool_z)
    tool_target = tool_target_from_sphere_center(
        target.sphere_center,
        approach,
        target_radius,
        tool_radius,
    )
    return make_tool_pose(tool_target, rotation_for_tool_z(target.tool_z, target.preferred_x))


def _target(
    name: str,
    xyz: list[float],
    tool_z: np.ndarray,
    *,
    preferred_x: list[float] | None = None,
) -> ContactTarget:
    preferred = None if preferred_x is None else np.array(preferred_x, dtype=float)
    return ContactTarget(
        name=name,
        sphere_center=np.array(xyz, dtype=float),
        tool_z=np.asarray(tool_z, dtype=float),
        preferred_x=preferred,
    )
