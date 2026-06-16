from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.gripper_pick_motion import simulate_pick
from src.sim.gripper_pick_scene import DEFAULT_PICK_MODEL, write_pick_scene_model
from src.sim.render_gripper_pick_gif import render_gif
from src.sim.verify_render_gifs import validate_gif


OUTPUT = ROOT / "outputs" / "gripper_pick" / "simplified_gripper_pick_cube_test.gif"
TEST_WIDTH = 960
TEST_HEIGHT = 720
TEST_FRAMES = 120
TEST_FPS = 20


def main() -> None:
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    result = simulate_pick(model_path, frames=TEST_FRAMES, fps=TEST_FPS)

    print("initial_cube_pos:", np.round(result.initial_cube_pos, 5))
    print("final_cube_pos:", np.round(result.final_cube_pos, 5))
    print(f"max_cube_z={result.max_cube_z:.5f}")
    print("final_gripper_qpos:", np.round(result.final_gripper_qpos, 5))

    if not result.lifted:
        raise RuntimeError("Expected the cube to be lifted by gripper contact and friction.")

    render_gif(
        model_path,
        OUTPUT,
        width=TEST_WIDTH,
        height=TEST_HEIGHT,
        frames=TEST_FRAMES,
        fps=TEST_FPS,
    )
    validate_gif(
        OUTPUT,
        expected_frames=TEST_FRAMES,
        expected_size=(TEST_WIDTH, TEST_HEIGHT),
        expected_fps=TEST_FPS,
    )
    print("status: OK")


if __name__ == "__main__":
    main()
