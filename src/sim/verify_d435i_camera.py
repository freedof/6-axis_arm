from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.d435i_model import DEFAULT_D435I_PICK_MODEL, write_d435i_pick_scene_model
from src.sim.render_d435i_preview import DEFAULT_OUTPUT_DIR, render_preview


def main() -> None:
    model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    for camera_name in ("d435i_depth", "d435i_rgb"):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise RuntimeError(f"Missing camera: {camera_name}")

    result = render_preview(model_path, DEFAULT_OUTPUT_DIR, width=424, height=240)
    for key in ("rgb_path", "context_rgb_path", "raw_depth_path", "noisy_depth_path", "raw_depth_vis_path", "noisy_depth_vis_path"):
        path = Path(result[key])
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Expected non-empty output for {key}: {path}")

    intrinsics = np.array(result["intrinsics"], dtype=float)
    extrinsic = np.array(result["extrinsic_world_to_camera"], dtype=float)
    if intrinsics.shape != (3, 3):
        raise RuntimeError(f"Expected 3x3 intrinsics, got {intrinsics.shape}")
    if extrinsic.shape != (4, 4):
        raise RuntimeError(f"Expected 4x4 extrinsic, got {extrinsic.shape}")

    raw_stats = result["raw_depth_stats"]
    noisy_stats = result["noisy_depth_stats"]
    if raw_stats["valid_ratio"] < 0.05:
        raise RuntimeError(f"Expected raw depth to contain visible scene pixels, got {raw_stats}")
    if noisy_stats["valid_ratio"] <= 0.0:
        raise RuntimeError(f"Expected noisy depth to contain valid pixels, got {noisy_stats}")

    print("model:", model_path)
    print("rgb:", result["rgb_path"])
    print("context_rgb:", result["context_rgb_path"])
    print("raw_depth_vis:", result["raw_depth_vis_path"])
    print("noisy_depth_vis:", result["noisy_depth_vis_path"])
    print("intrinsics:", np.round(intrinsics, 3))
    print("raw_depth_stats:", raw_stats)
    print("noisy_depth_stats:", noisy_stats)
    print("status: OK")


if __name__ == "__main__":
    main()
