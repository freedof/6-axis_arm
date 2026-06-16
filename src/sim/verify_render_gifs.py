from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.demo_xyz_joint_roundtrip import (
    DEFAULT_MODEL,
    DEFAULT_TARGET_RADIUS,
    DEFAULT_TOOL_RADIUS,
    DEFAULT_XYZ_A,
    DEFAULT_XYZ_B,
)
from src.sim.render_mixed_orientation_gif import render_gif as render_mixed_orientation_gif
from src.sim.planning_cases import planning_cases
from src.sim.render_planned_roundtrip_gif import render_case_gif
from src.sim.render_roundtrip_gif import render_gif as render_roundtrip_gif
from src.sim.render_timing import roundtrip_motion_alpha


OUTPUT_DIR = ROOT / "outputs" / "test_gifs"
TEST_WIDTH = 960
TEST_HEIGHT = 720
TEST_FRAMES = 48
TEST_FPS = 12


def validate_roundtrip_dwell_timing() -> None:
    alpha_0, reverse_0 = roundtrip_motion_alpha(0, TEST_FRAMES, TEST_FPS)
    alpha_a, reverse_a = roundtrip_motion_alpha(TEST_FPS - 1, TEST_FRAMES, TEST_FPS)
    alpha_move, reverse_move = roundtrip_motion_alpha(TEST_FPS + 1, TEST_FRAMES, TEST_FPS)
    alpha_b, reverse_b = roundtrip_motion_alpha(TEST_FRAMES // 2, TEST_FRAMES, TEST_FPS)
    alpha_b_hold, reverse_b_hold = roundtrip_motion_alpha(TEST_FRAMES // 2 + TEST_FPS - 1, TEST_FRAMES, TEST_FPS)
    if (alpha_0, reverse_0) != (0.0, False):
        raise RuntimeError("Roundtrip timing should start by holding target A.")
    if (alpha_a, reverse_a) != (0.0, False):
        raise RuntimeError("Roundtrip timing should hold target A for about one second.")
    if not (0.0 < alpha_move < 1.0 and reverse_move is False):
        raise RuntimeError("Roundtrip timing should move from A to B after the A dwell.")
    if (alpha_b, reverse_b) != (1.0, False):
        raise RuntimeError("Roundtrip timing should hold target B at the midpoint.")
    if (alpha_b_hold, reverse_b_hold) != (1.0, False):
        raise RuntimeError("Roundtrip timing should hold target B for about one second.")


def validate_gif(
    path: Path,
    *,
    expected_frames: int,
    expected_size: tuple[int, int],
    expected_fps: int = TEST_FPS,
) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing GIF: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"GIF is empty: {path}")

    with Image.open(path) as image:
        if image.format != "GIF":
            raise RuntimeError(f"Expected GIF format for {path}, got {image.format}")
        if image.size != expected_size:
            raise RuntimeError(f"Expected size {expected_size} for {path}, got {image.size}")
        if image.n_frames > expected_frames:
            raise RuntimeError(f"Expected at most {expected_frames} stored frames for {path}, got {image.n_frames}")
        if image.n_frames < 2:
            raise RuntimeError(f"Expected animated GIF for {path}, got {image.n_frames} frame(s)")

        durations_ms: list[int] = []
        frames = []
        for frame in ImageSequence.Iterator(image):
            durations_ms.append(int(frame.info.get("duration", 0)))
            frames.append(frame.convert("RGB"))
        if len(frames) != image.n_frames:
            raise RuntimeError(f"ImageSequence returned {len(frames)} frames for {path}")
        expected_duration_ms = expected_frames * int(1000 / expected_fps)
        total_duration_ms = sum(durations_ms)
        if abs(total_duration_ms - expected_duration_ms) > max(250, int(0.10 * expected_duration_ms)):
            raise RuntimeError(
                f"Expected GIF duration near {expected_duration_ms} ms for {path}, got {total_duration_ms} ms"
            )

        first = np.asarray(frames[0], dtype=np.int16)
        midpoint = np.asarray(frames[len(frames) // 2], dtype=np.int16)
        if float(np.mean(np.abs(first - midpoint))) < 0.25:
            raise RuntimeError(f"GIF frames do not visibly change: {path}")

        if float(np.std(first)) < 1.0:
            raise RuntimeError(f"First GIF frame appears blank: {path}")

    print(f"validated: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_roundtrip_dwell_timing()

    cases = [
        (
            "tool_down",
            OUTPUT_DIR / "roundtrip_tool_down_test.gif",
            lambda output: render_roundtrip_gif(
                DEFAULT_MODEL,
                output,
                DEFAULT_XYZ_A.copy(),
                DEFAULT_XYZ_B.copy(),
                TEST_WIDTH,
                TEST_HEIGHT,
                TEST_FRAMES,
                TEST_FPS,
                DEFAULT_TARGET_RADIUS,
                DEFAULT_TOOL_RADIUS,
            ),
        ),
        (
            "mixed_orientation",
            OUTPUT_DIR / "roundtrip_mixed_orientation_test.gif",
            lambda output: render_mixed_orientation_gif(
                DEFAULT_MODEL,
                output,
                DEFAULT_XYZ_A.copy(),
                DEFAULT_XYZ_B.copy(),
                TEST_WIDTH,
                TEST_HEIGHT,
                TEST_FRAMES,
                TEST_FPS,
                DEFAULT_TARGET_RADIUS,
                DEFAULT_TOOL_RADIUS,
            ),
        ),
    ]
    for planning_case in planning_cases():
        if not planning_case.render_gif:
            continue
        safe_case_id = planning_case.case_id.lower().replace("-", "_")
        cases.append(
            (
                f"planned_{planning_case.case_id}",
                OUTPUT_DIR / f"{safe_case_id}_planned_test.gif",
                lambda output, case=planning_case: render_case_gif(
                    DEFAULT_MODEL,
                    output,
                    case,
                    TEST_WIDTH,
                    TEST_HEIGHT,
                    TEST_FRAMES,
                    TEST_FPS,
                    shortcut=False,
                ),
            )
        )

    for name, output, render in cases:
        print(f"rendering: {name}")
        render(output)
        validate_gif(output, expected_frames=TEST_FRAMES, expected_size=(TEST_WIDTH, TEST_HEIGHT))

    print("status: OK")


if __name__ == "__main__":
    main()
