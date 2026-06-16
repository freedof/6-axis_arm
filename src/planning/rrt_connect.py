from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.robot.model import SerialRobotModel


StateValidityFn = Callable[[np.ndarray], bool]


@dataclass(frozen=True)
class RRTConnectConfig:
    max_iterations: int = 2000
    step_size: float = 0.12
    edge_resolution: float = 0.04
    goal_sample_rate: float = 0.10
    rng_seed: int = 13
    try_direct: bool = True


@dataclass(frozen=True)
class RRTConnectResult:
    success: bool
    path: tuple[np.ndarray, ...]
    iterations: int
    reason: str


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parents: list[int]

    def nearest_index(self, q: np.ndarray) -> int:
        distances = [float(np.linalg.norm(node - q)) for node in self.nodes]
        return int(np.argmin(distances))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q.copy())
        self.parents.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, index: int) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        while index >= 0:
            path.append(self.nodes[index])
            index = self.parents[index]
        path.reverse()
        return path


def plan_joint_rrt_connect(
    robot: SerialRobotModel,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    is_state_valid: StateValidityFn,
    *,
    config: RRTConnectConfig | None = None,
) -> RRTConnectResult:
    cfg = RRTConnectConfig() if config is None else config
    q_start = np.asarray(q_start, dtype=float).copy()
    q_goal = np.asarray(q_goal, dtype=float).copy()
    if q_start.shape != (robot.dof,):
        raise ValueError(f"Expected q_start shape {(robot.dof,)}, got {q_start.shape}")
    if q_goal.shape != (robot.dof,):
        raise ValueError(f"Expected q_goal shape {(robot.dof,)}, got {q_goal.shape}")

    if not is_state_valid(q_start):
        return RRTConnectResult(False, (), 0, "start state is invalid")
    if not is_state_valid(q_goal):
        return RRTConnectResult(False, (), 0, "goal state is invalid")
    if cfg.try_direct and _edge_is_valid(q_start, q_goal, is_state_valid, cfg.edge_resolution):
        return RRTConnectResult(True, (q_start, q_goal), 0, "direct path is valid")

    rng = np.random.default_rng(cfg.rng_seed)
    tree_a = _Tree([q_start], [-1])
    tree_b = _Tree([q_goal], [-1])
    tree_a_from_start = True

    for iteration in range(1, cfg.max_iterations + 1):
        q_bias = q_goal if tree_a_from_start else q_start
        q_sample = _sample(robot, q_bias, rng, cfg.goal_sample_rate)
        advanced, new_index = _extend(tree_a, q_sample, is_state_valid, cfg)
        if advanced:
            reached, other_index = _connect(tree_b, tree_a.nodes[new_index], is_state_valid, cfg)
            if reached:
                path = _join_paths(tree_a, new_index, tree_b, other_index, tree_a_from_start)
                return RRTConnectResult(True, tuple(_densify_path(path, cfg.edge_resolution)), iteration, "connected")

        tree_a, tree_b = tree_b, tree_a
        tree_a_from_start = not tree_a_from_start

    return RRTConnectResult(False, (), cfg.max_iterations, "iteration limit reached")


def shortcut_path(
    path: tuple[np.ndarray, ...],
    is_state_valid: StateValidityFn,
    *,
    attempts: int = 100,
    edge_resolution: float = 0.04,
    rng_seed: int = 23,
) -> tuple[np.ndarray, ...]:
    if len(path) <= 2:
        return path

    rng = np.random.default_rng(rng_seed)
    shortened = list(path)
    for _ in range(attempts):
        if len(shortened) <= 2:
            break
        i, j = sorted(rng.choice(len(shortened), size=2, replace=False))
        if j <= i + 1:
            continue
        if _edge_is_valid(shortened[i], shortened[j], is_state_valid, edge_resolution):
            shortened = shortened[: i + 1] + shortened[j:]
    return tuple(shortened)


def path_length(path: tuple[np.ndarray, ...]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


def _sample(
    robot: SerialRobotModel,
    q_goal: np.ndarray,
    rng: np.random.Generator,
    goal_sample_rate: float,
) -> np.ndarray:
    if rng.random() < goal_sample_rate:
        return q_goal.copy()
    return rng.uniform(robot.lower_limits, robot.upper_limits)


def _extend(
    tree: _Tree,
    q_target: np.ndarray,
    is_state_valid: StateValidityFn,
    cfg: RRTConnectConfig,
) -> tuple[bool, int]:
    nearest = tree.nearest_index(q_target)
    q_new = _steer(tree.nodes[nearest], q_target, cfg.step_size)
    if not _edge_is_valid(tree.nodes[nearest], q_new, is_state_valid, cfg.edge_resolution):
        return False, nearest
    return True, tree.add(q_new, nearest)


def _connect(
    tree: _Tree,
    q_target: np.ndarray,
    is_state_valid: StateValidityFn,
    cfg: RRTConnectConfig,
) -> tuple[bool, int]:
    current_index = tree.nearest_index(q_target)
    while True:
        q_current = tree.nodes[current_index]
        if np.linalg.norm(q_target - q_current) <= cfg.step_size:
            if _edge_is_valid(q_current, q_target, is_state_valid, cfg.edge_resolution):
                current_index = tree.add(q_target, current_index)
                return True, current_index
            return False, current_index

        q_new = _steer(q_current, q_target, cfg.step_size)
        if not _edge_is_valid(q_current, q_new, is_state_valid, cfg.edge_resolution):
            return False, current_index
        current_index = tree.add(q_new, current_index)


def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    delta = q_to - q_from
    distance = float(np.linalg.norm(delta))
    if distance <= step_size:
        return q_to.copy()
    return q_from + delta * (step_size / distance)


def _edge_is_valid(
    q_from: np.ndarray,
    q_to: np.ndarray,
    is_state_valid: StateValidityFn,
    resolution: float,
) -> bool:
    distance = float(np.linalg.norm(q_to - q_from))
    steps = max(1, int(np.ceil(distance / resolution)))
    for step in range(steps + 1):
        alpha = step / steps
        q = (1.0 - alpha) * q_from + alpha * q_to
        if not is_state_valid(q):
            return False
    return True


def _join_paths(
    tree_a: _Tree,
    index_a: int,
    tree_b: _Tree,
    index_b: int,
    tree_a_from_start: bool,
) -> list[np.ndarray]:
    path_a = tree_a.path_to_root(index_a)
    path_b = tree_b.path_to_root(index_b)
    if tree_a_from_start:
        return path_a + list(reversed(path_b[:-1]))
    return path_b + list(reversed(path_a[:-1]))


def _densify_path(path: list[np.ndarray], resolution: float) -> list[np.ndarray]:
    if not path:
        return []

    dense: list[np.ndarray] = [path[0]]
    for q_from, q_to in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(q_to - q_from))
        steps = max(1, int(np.ceil(distance / resolution)))
        for step in range(1, steps + 1):
            alpha = step / steps
            dense.append((1.0 - alpha) * q_from + alpha * q_to)
    return dense
