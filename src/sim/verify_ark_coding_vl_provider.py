from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.vl_region import estimate_region_3d, locate_ark_coding_vision_region, region3d_to_dict
from src.sim.d435i_model import DEFAULT_D435I_PICK_MODEL, write_d435i_pick_scene_model
from src.sim.render_d435i_preview import DEFAULT_OUTPUT_DIR, render_preview


def main() -> None:
    if not (os.environ.get("ARK_CODING_API_KEY") or os.environ.get("ARK_API_KEY")):
        print("status: SKIPPED")
        print("reason: ARK_CODING_API_KEY/ARK_API_KEY is not set; ark_coding_vision provider was not called.")
        return

    model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    output_dir = DEFAULT_OUTPUT_DIR / "ark_coding_vl_verify"
    observation = render_preview(model_path, output_dir, width=424, height=240, pose="scan")
    overlay = output_dir / "ark_coding_vl_region_overlay.png"
    region = locate_ark_coding_vision_region(
        observation["rgb_path"],
        prompt="pick the red block",
        output_path=overlay,
    )
    region3d = estimate_region_3d(
        depth_path=observation["raw_depth_path"],
        intrinsics=observation["intrinsics"],
        extrinsic_world_to_camera=observation["extrinsic_world_to_camera"],
        region=region,
    )
    if region3d.valid_pixel_count < 20:
        raise RuntimeError(f"Ark coding VL bbox has too few valid depth pixels: {region3d.valid_pixel_count}")

    print("rgb:", observation["rgb_path"])
    print("overlay:", overlay)
    print("region:", region)
    print("region_3d:", region3d_to_dict(region3d))
    print("status: OK")


if __name__ == "__main__":
    main()
