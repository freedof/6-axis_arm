from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.vl_region import estimate_region_3d, locate_red_region_fixture, region3d_to_dict
from src.sim.d435i_model import DEFAULT_D435I_PICK_MODEL, write_d435i_pick_scene_model
from src.sim.gripper_pick_scene import CUBE_CENTER
from src.sim.render_d435i_preview import DEFAULT_OUTPUT_DIR, render_preview


def main() -> None:
    model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    observation = render_preview(model_path, DEFAULT_OUTPUT_DIR, width=424, height=240, pose="scan")
    overlay = DEFAULT_OUTPUT_DIR / "vl_region_overlay.png"
    region = locate_red_region_fixture(observation["rgb_path"], prompt="pick the red block", output_path=overlay)
    region3d = estimate_region_3d(
        depth_path=observation["raw_depth_path"],
        intrinsics=observation["intrinsics"],
        extrinsic_world_to_camera=observation["extrinsic_world_to_camera"],
        region=region,
    )
    result = region3d_to_dict(region3d)

    expected = np.array(CUBE_CENTER, dtype=float)
    error = float(np.linalg.norm(region3d.center_world_m - expected))
    if region3d.valid_pixel_count < 20:
        raise RuntimeError(f"Expected enough depth pixels inside VL region, got {region3d.valid_pixel_count}")
    if error > 0.08:
        raise RuntimeError(f"Expected estimated center near cube center, error={error:.6f} m, result={result}")

    print("rgb:", observation["rgb_path"])
    print("overlay:", overlay)
    print("region:", region)
    print("region_3d:", result)
    print(f"center_error_m: {error:.6f}")
    print("status: OK")


if __name__ == "__main__":
    main()
