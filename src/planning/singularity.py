from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robot.kinematics import geometric_jacobian
from src.robot.model import SerialRobotModel


DEFAULT_MAX_CONDITION_NUMBER = 1000.0
DEFAULT_MIN_SINGULAR_VALUE = 1.0e-3
DEFAULT_MIN_MANIPULABILITY = 5.0e-4


@dataclass(frozen=True)
class SingularityMetrics:
    condition_number: float
    min_singular_value: float
    manipulability: float

    @property
    def valid(self) -> bool:
        return (
            self.condition_number <= DEFAULT_MAX_CONDITION_NUMBER
            and self.min_singular_value >= DEFAULT_MIN_SINGULAR_VALUE
            and self.manipulability >= DEFAULT_MIN_MANIPULABILITY
        )


@dataclass(frozen=True)
class SingularityPathSummary:
    valid: bool
    max_condition_number: float
    min_singular_value: float
    min_manipulability: float
    samples: int


class SingularityChecker:
    def __init__(
        self,
        robot: SerialRobotModel,
        *,
        max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER,
        min_singular_value: float = DEFAULT_MIN_SINGULAR_VALUE,
        min_manipulability: float = DEFAULT_MIN_MANIPULABILITY,
    ) -> None:
        self.robot = robot
        self.max_condition_number = float(max_condition_number)
        self.min_singular_value = float(min_singular_value)
        self.min_manipulability = float(min_manipulability)

    def metrics(self, q: np.ndarray) -> SingularityMetrics:
        singular_values = np.linalg.svd(geometric_jacobian(self.robot, q), compute_uv=False)
        min_singular = float(singular_values[-1])
        condition = float("inf") if min_singular <= 0.0 else float(singular_values[0] / min_singular)
        return SingularityMetrics(
            condition_number=condition,
            min_singular_value=min_singular,
            manipulability=float(np.prod(singular_values)),
        )

    def is_state_valid(self, q: np.ndarray) -> bool:
        metrics = self.metrics(q)
        return (
            metrics.condition_number <= self.max_condition_number
            and metrics.min_singular_value >= self.min_singular_value
            and metrics.manipulability >= self.min_manipulability
        )

    def path_summary(self, path: tuple[np.ndarray, ...], *, resolution: float = 0.04) -> SingularityPathSummary:
        if not path:
            return SingularityPathSummary(False, float("inf"), 0.0, 0.0, 0)

        max_condition = 0.0
        min_singular = float("inf")
        min_manipulability = float("inf")
        samples = 0

        if len(path) == 1:
            metrics = self.metrics(path[0])
            return SingularityPathSummary(
                self._metrics_are_valid(metrics),
                metrics.condition_number,
                metrics.min_singular_value,
                metrics.manipulability,
                1,
            )

        for q_from, q_to in zip(path[:-1], path[1:]):
            distance = float(np.linalg.norm(q_to - q_from))
            steps = max(1, int(np.ceil(distance / resolution)))
            for step in range(steps + 1):
                alpha = step / steps
                q = (1.0 - alpha) * q_from + alpha * q_to
                metrics = self.metrics(q)
                max_condition = max(max_condition, metrics.condition_number)
                min_singular = min(min_singular, metrics.min_singular_value)
                min_manipulability = min(min_manipulability, metrics.manipulability)
                samples += 1

        return SingularityPathSummary(
            valid=(
                max_condition <= self.max_condition_number
                and min_singular >= self.min_singular_value
                and min_manipulability >= self.min_manipulability
            ),
            max_condition_number=max_condition,
            min_singular_value=min_singular,
            min_manipulability=min_manipulability,
            samples=samples,
        )

    def _metrics_are_valid(self, metrics: SingularityMetrics) -> bool:
        return (
            metrics.condition_number <= self.max_condition_number
            and metrics.min_singular_value >= self.min_singular_value
            and metrics.manipulability >= self.min_manipulability
        )
