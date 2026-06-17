from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.vl_region import estimate_region_3d, locate_ark_coding_vision_region, region3d_to_dict
from src.sim.d435i_model import DEFAULT_D435I_PICK_MODEL, write_d435i_pick_scene_model
from src.sim.render_d435i_preview import DEFAULT_OUTPUT_DIR, render_preview

CONFIG_PATH = ROOT / "config" / "vl_providers.local.json"


def main() -> None:
    if not _has_provider_key("ark_coding_vision"):
        print("status: SKIPPED")
        print("reason: config/vl_providers.local.json is missing or ark_coding_vision.api_key is not set.")
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


def _has_provider_key(provider: str) -> bool:
    if not CONFIG_PATH.exists():
        return False
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = data.get("providers", {}).get(provider, {})
    key = str(config.get("api_key", ""))
    return bool(key and not key.startswith("REPLACE_"))


if __name__ == "__main__":
    main()
