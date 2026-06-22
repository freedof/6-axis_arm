from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model
from src.sim.gripper_pick_scene import DEFAULT_MULTI_OBJECT_MODEL, DEFAULT_PICK_MODEL
from src.sim.gripper_pick_scene import TableObjectSpec, write_multi_object_scene_model, write_pick_scene_model


DEFAULT_D435I_GRIPPER_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_with_gripper_d435i.xml"
DEFAULT_D435I_PICK_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_gripper_d435i_pick_scene.xml"
DEFAULT_D435I_MULTI_OBJECT_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_gripper_d435i_multi_object_scene.xml"


def write_d435i_gripper_model(
    output_path: Path = DEFAULT_D435I_GRIPPER_MODEL,
    *,
    source_model: Path = DEFAULT_GRIPPER_MODEL,
) -> Path:
    write_gripper_model(source_model)
    return _write_with_d435i(source_model, output_path, "dobot_cr5_with_gripper_d435i")


def write_d435i_pick_scene_model(
    output_path: Path = DEFAULT_D435I_PICK_MODEL,
    *,
    source_model: Path = DEFAULT_PICK_MODEL,
) -> Path:
    write_pick_scene_model(source_model)
    return _write_with_d435i(source_model, output_path, "dobot_cr5_gripper_d435i_pick_scene")


def write_d435i_multi_object_scene_model(
    output_path: Path = DEFAULT_D435I_MULTI_OBJECT_MODEL,
    *,
    source_model: Path = DEFAULT_MULTI_OBJECT_MODEL,
    objects: tuple[TableObjectSpec, ...] | list[TableObjectSpec] | None = None,
) -> Path:
    if objects is None:
        write_multi_object_scene_model(source_model)
    else:
        write_multi_object_scene_model(source_model, objects=objects)
    return _write_with_d435i(source_model, output_path, "dobot_cr5_gripper_d435i_multi_object_scene")


def _write_with_d435i(source_model: Path, output_path: Path, model_name: str) -> Path:
    tree = ET.parse(source_model)
    root = tree.getroot()
    root.set("model", model_name)

    asset = root.find("asset")
    if asset is not None:
        _ensure_material(asset, "d435i_body_mat", "0.07 0.075 0.08 1")
        _ensure_material(asset, "d435i_face_mat", "0.015 0.018 0.022 1")
        _ensure_material(asset, "d435i_lens_mat", "0.03 0.08 0.14 1")

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"Model has no worldbody: {source_model}")
    gripper = _find_body(world, "parallel_gripper")
    if gripper is None:
        raise RuntimeError("Missing parallel_gripper body; cannot attach D435i.")

    if gripper.find("body[@name='d435i_camera_body']") is None:
        gripper.append(_d435i_body())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode")
    return output_path


def _d435i_body() -> ET.Element:
    body = ET.Element("body", {"name": "d435i_camera_body", "pos": "0 -0.060 0.024"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "d435i_body_geom",
            "type": "box",
            "pos": "0 0 0",
            "size": "0.045 0.012 0.013",
            "material": "d435i_body_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "d435i_front_face",
            "type": "box",
            "pos": "0 0 0.0135",
            "size": "0.043 0.010 0.0015",
            "material": "d435i_face_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for name, x in (
        ("d435i_left_ir_lens", -0.022),
        ("d435i_rgb_lens", 0.0),
        ("d435i_right_ir_lens", 0.022),
    ):
        ET.SubElement(
            body,
            "geom",
            {
                "name": name,
                "type": "cylinder",
                "pos": f"{x:g} 0 0.016",
                "size": "0.004 0.002",
                "material": "d435i_lens_mat",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.SubElement(body, "site", {"name": "d435i_mount", "pos": "0 0 -0.013", "size": "0.004", "rgba": "0.7 0.7 0.1 1"})
    ET.SubElement(body, "site", {"name": "d435i_depth_optical_frame", "pos": "0 0 0.018", "size": "0.004", "rgba": "0.1 0.8 1 1"})
    ET.SubElement(body, "camera", {"name": "d435i_depth", "pos": "0 0 0.018", "xyaxes": "1 0 0 0 -1 0", "fovy": "65"})
    ET.SubElement(body, "camera", {"name": "d435i_rgb", "pos": "0 0 0.018", "xyaxes": "1 0 0 0 -1 0", "fovy": "65"})
    return body


def _ensure_material(asset: ET.Element, name: str, rgba: str) -> None:
    if asset.find(f"material[@name='{name}']") is None:
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


if __name__ == "__main__":
    gripper_path = write_d435i_gripper_model()
    pick_scene_path = write_d435i_pick_scene_model()
    multi_object_path = write_d435i_multi_object_scene_model()
    print(gripper_path)
    print(pick_scene_path)
    print(multi_object_path)
