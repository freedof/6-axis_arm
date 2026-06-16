from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from src.sim.planning_model import DEFAULT_PLANNING_MODEL, write_planning_model
from src.sim.planning_cases import PlanningCase, planning_cases


OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"
OBSTACLE_OUTPUT_DIR = OUTPUT_ROOT / "obstacle_scene"
TEST_GIF_DIR = OUTPUT_ROOT / "test_gifs"
DEFAULT_OBSTACLE_ID = "TC-RRT-OBSTACLE-001"
DEFAULT_OBSTACLE_MODEL = OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_001.xml"
DEFAULT_OBSTACLE_GIF = OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_001.gif"
DEFAULT_OBSTACLE_TEST_GIF = TEST_GIF_DIR / "tc_rrt_obstacle_001_test.gif"


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    pos: np.ndarray
    size: np.ndarray
    rgba: tuple[float, float, float, float] = (0.95, 0.52, 0.08, 0.72)


@dataclass(frozen=True)
class ObstaclePlanningCase:
    case_id: str
    name: str
    purpose: str
    base_case_id: str
    obstacles: tuple[BoxObstacle, ...]
    model_output: Path
    gif_output: Path
    test_gif_output: Path
    expect_direct_collision: bool = True


def obstacle_cases() -> tuple[ObstaclePlanningCase, ...]:
    return (
        ObstaclePlanningCase(
            case_id="TC-RRT-OBSTACLE-001",
            name="单箱体阻挡近距离竖直接触",
            purpose="验证一个中等箱体挡住短距离 A->B 直线路径时，RRT-Connect 能绕开障碍。",
            base_case_id="TC-RRT-MULTI-001",
            obstacles=(
                BoxObstacle(
                    name="planning_obstacle_box",
                    pos=np.array([0.24977, -0.62073, 0.17463], dtype=float),
                    size=np.array([0.07, 0.07, 0.14], dtype=float),
                ),
            ),
            model_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_001.xml",
            gif_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_001.gif",
            test_gif_output=TEST_GIF_DIR / "tc_rrt_obstacle_001_test.gif",
        ),
        ObstaclePlanningCase(
            case_id="TC-RRT-OBSTACLE-002",
            name="长距离竖直接触中的高箱体绕行",
            purpose="验证较远目标之间存在较高障碍时，规划器能选择非直线的绕行构型。",
            base_case_id="TC-RRT-MULTI-002",
            obstacles=(
                BoxObstacle(
                    name="planning_obstacle_tall_box",
                    pos=np.array([0.285, -0.575, 0.245], dtype=float),
                    size=np.array([0.08, 0.075, 0.18], dtype=float),
                    rgba=(0.90, 0.36, 0.16, 0.72),
                ),
            ),
            model_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_002.xml",
            gif_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_002.gif",
            test_gif_output=TEST_GIF_DIR / "tc_rrt_obstacle_002_test.gif",
        ),
        ObstaclePlanningCase(
            case_id="TC-RRT-OBSTACLE-003",
            name="左右跨越中的扫掠区障碍",
            purpose="验证左右跨越运动中，障碍挡住机械臂实际扫过区域时是否能绕行。",
            base_case_id="TC-RRT-MULTI-005",
            obstacles=(
                BoxObstacle(
                    name="planning_obstacle_sweep_barrier",
                    pos=np.array([0.060, -0.200, 0.200], dtype=float),
                    size=np.array([0.05, 0.05, 0.10], dtype=float),
                    rgba=(0.76, 0.28, 0.78, 0.72),
                ),
            ),
            model_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_003.xml",
            gif_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_003.gif",
            test_gif_output=TEST_GIF_DIR / "tc_rrt_obstacle_003_test.gif",
        ),
        ObstaclePlanningCase(
            case_id="TC-RRT-OBSTACLE-004",
            name="双箱体窄通道绕行",
            purpose="验证存在两个障碍形成局部窄通道时，路径仍能保持无碰撞并连续通过。",
            base_case_id="TC-RRT-MULTI-004",
            obstacles=(
                BoxObstacle(
                    name="planning_obstacle_channel_a",
                    pos=np.array([0.160, -0.580, 0.160], dtype=float),
                    size=np.array([0.04, 0.04, 0.10], dtype=float),
                    rgba=(0.20, 0.58, 0.86, 0.72),
                ),
                BoxObstacle(
                    name="planning_obstacle_channel_b",
                    pos=np.array([0.280, -0.580, 0.100], dtype=float),
                    size=np.array([0.04, 0.04, 0.10], dtype=float),
                    rgba=(0.20, 0.58, 0.86, 0.72),
                ),
            ),
            model_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_004.xml",
            gif_output=OBSTACLE_OUTPUT_DIR / "tc_rrt_obstacle_004.gif",
            test_gif_output=TEST_GIF_DIR / "tc_rrt_obstacle_004_test.gif",
        ),
    )


def obstacle_case_by_id(case_id: str) -> ObstaclePlanningCase:
    for case in obstacle_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in obstacle_cases())
    raise ValueError(f"Unknown obstacle case_id {case_id!r}. Known cases: {known}")


def default_obstacle_case() -> ObstaclePlanningCase:
    return obstacle_case_by_id(DEFAULT_OBSTACLE_ID)


def base_planning_case(obstacle_case: ObstaclePlanningCase) -> PlanningCase:
    for case in planning_cases():
        if case.case_id == obstacle_case.base_case_id:
            return case
    raise RuntimeError(f"Missing base planning case: {obstacle_case.base_case_id}")


def write_obstacle_model(
    output_path: Path | None = None,
    *,
    base_model: Path = DEFAULT_PLANNING_MODEL,
    obstacle_case: ObstaclePlanningCase | None = None,
    obstacles: tuple[BoxObstacle, ...] | None = None,
) -> Path:
    obstacle_case = default_obstacle_case() if obstacle_case is None else obstacle_case
    output_path = obstacle_case.model_output if output_path is None else output_path
    obstacles = obstacle_case.obstacles if obstacles is None else obstacles
    if Path(base_model) == DEFAULT_PLANNING_MODEL:
        write_planning_model(DEFAULT_PLANNING_MODEL)
    tree = ET.parse(base_model)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"Model has no worldbody: {base_model}")

    for index, obstacle in enumerate(obstacles):
        body = ET.SubElement(
            world,
            "body",
            {
                "name": f"planning_obstacle_{obstacle_case.case_id.lower().replace('-', '_')}_{index + 1}",
                "pos": _format_vec(obstacle.pos),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": obstacle.name,
                "type": "box",
                "size": _format_vec(obstacle.size),
                "rgba": _format_vec(np.array(obstacle.rgba, dtype=float)),
                "contype": "1",
                "conaffinity": "1",
                "group": "2",
            },
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="unicode")
    return output_path


def _format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)
