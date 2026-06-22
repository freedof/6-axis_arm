from __future__ import annotations

from pathlib import Path
import sys

import mujoco

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.d435i_model import DEFAULT_D435I_MULTI_OBJECT_MODEL, write_d435i_multi_object_scene_model
from src.sim.gripper_pick_scene import DEFAULT_MULTI_OBJECT_MODEL, DEFAULT_MULTI_OBJECT_SPECS, object_specs_to_dicts, write_multi_object_scene_model
from src.sim.render_d435i_preview import render_preview


OUTPUT_DIR = ROOT / "outputs" / "multi_object_scene"


def main() -> None:
    scene_path = write_multi_object_scene_model(DEFAULT_MULTI_OBJECT_MODEL)
    d435i_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL)

    scene_model = mujoco.MjModel.from_xml_path(str(scene_path))
    d435i_model = mujoco.MjModel.from_xml_path(str(d435i_path))
    _assert_objects(scene_model)
    _assert_objects(d435i_model)
    _assert_camera(d435i_model, "d435i_depth")
    _assert_camera(d435i_model, "d435i_rgb")

    preview = render_preview(d435i_path, OUTPUT_DIR, width=424, height=240, pose="scan_high")
    raw_valid_ratio = float(preview["raw_depth_stats"]["valid_ratio"])
    if raw_valid_ratio < 0.05:
        raise RuntimeError(f"Expected multi-object D435i depth to contain visible scene pixels: {preview['raw_depth_stats']}")

    print("status: OK")
    print("scene:", scene_path)
    print("d435i_scene:", d435i_path)
    print("objects:", object_specs_to_dicts(DEFAULT_MULTI_OBJECT_SPECS))
    print("rgb:", preview["rgb_path"])
    print("raw_depth_vis:", preview["raw_depth_vis_path"])
    print("raw_depth_stats:", preview["raw_depth_stats"])


def _assert_objects(model: mujoco.MjModel) -> None:
    for spec in DEFAULT_MULTI_OBJECT_SPECS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.name)
        if body_id < 0:
            raise RuntimeError(f"Missing object body: {spec.name}")
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{spec.name}_geom")
        if geom_id < 0:
            raise RuntimeError(f"Missing object geom: {spec.name}_geom")
        if int(model.geom_contype[geom_id]) == 0 or int(model.geom_conaffinity[geom_id]) == 0:
            raise RuntimeError(f"Object geom should be collision-enabled: {spec.name}_geom")


def _assert_camera(model: mujoco.MjModel, name: str) -> None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera_id < 0:
        raise RuntimeError(f"Missing D435i camera: {name}")


if __name__ == "__main__":
    main()
