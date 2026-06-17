from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.demo_xyz_joint_roundtrip import DEFAULT_MODEL


DEFAULT_GRIPPER_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_with_gripper.xml"
GRIPPER_OPEN_QPOS = 0.025


def write_gripper_model(
    output_path: Path = DEFAULT_GRIPPER_MODEL,
    *,
    source_model: Path = DEFAULT_MODEL,
    include_roundtrip_targets: bool = False,
) -> Path:
    tree = ET.parse(source_model)
    root = tree.getroot()
    root.set("model", "dobot_cr5_with_parallel_gripper")

    asset = root.find("asset")
    if asset is not None:
        _ensure_material(asset, "gripper_palm_mat", "0.18 0.20 0.22 1")
        _ensure_material(asset, "gripper_finger_mat", "0.06 0.07 0.08 1")
        _ensure_material(asset, "gripper_pad_mat", "0.02 0.02 0.02 1")

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"Model has no worldbody: {source_model}")
    if not include_roundtrip_targets:
        _remove_roundtrip_targets(world)

    link6 = _find_body(world, "Link6")
    if link6 is None:
        raise RuntimeError("Missing Link6 body; cannot attach gripper.")

    if link6.find("body[@name='parallel_gripper']") is None:
        link6.append(_parallel_gripper_body())

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    _ensure_position_actuator(
        actuator,
        "gripper_left_position",
        "gripper_left_slide",
        "0 0.035",
    )
    _ensure_position_actuator(
        actuator,
        "gripper_right_position",
        "gripper_right_slide",
        "0 0.035",
    )

    _append_gripper_keyframe_values(root)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode")
    return output_path


def _parallel_gripper_body() -> ET.Element:
    gripper = ET.Element("body", {"name": "parallel_gripper", "pos": "0 0 0.08"})
    ET.SubElement(
        gripper,
        "geom",
        {
            "name": "gripper_palm",
            "type": "box",
            "pos": "0 0 0.018",
            "size": "0.050 0.022 0.018",
            "material": "gripper_palm_mat",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(gripper, "site", {"name": "gripper_mount", "pos": "0 0 0", "size": "0.008", "rgba": "0.95 0.65 0.10 1"})
    ET.SubElement(gripper, "site", {"name": "gripper_tcp", "pos": "0 0 0.110", "size": "0.010", "rgba": "0.10 0.95 0.40 1"})

    gripper.append(
        _finger_body(
            body_name="gripper_left_finger",
            joint_name="gripper_left_slide",
            base_pos="0.023 0 0.052",
            axis="1 0 0",
            finger_geom="gripper_left_finger_geom",
            pad_geom="gripper_left_pad",
            pad_x="-0.0075",
        )
    )
    gripper.append(
        _finger_body(
            body_name="gripper_right_finger",
            joint_name="gripper_right_slide",
            base_pos="-0.023 0 0.052",
            axis="-1 0 0",
            finger_geom="gripper_right_finger_geom",
            pad_geom="gripper_right_pad",
            pad_x="0.0075",
        )
    )
    return gripper


def _finger_body(
    *,
    body_name: str,
    joint_name: str,
    base_pos: str,
    axis: str,
    finger_geom: str,
    pad_geom: str,
    pad_x: str,
) -> ET.Element:
    finger = ET.Element("body", {"name": body_name, "pos": base_pos})
    ET.SubElement(
        finger,
        "joint",
        {
            "name": joint_name,
            "type": "slide",
            "axis": axis,
            "range": "0 0.035",
            "damping": "0.6",
            "armature": "0.001",
        },
    )
    ET.SubElement(
        finger,
        "geom",
        {
            "name": finger_geom,
            "type": "box",
            "pos": "0 0 0.035",
            "size": "0.007 0.012 0.030",
            "material": "gripper_finger_mat",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        finger,
        "geom",
        {
            "name": pad_geom,
            "type": "box",
            "pos": f"{pad_x} 0 0.035",
            "size": "0.004 0.013 0.030",
            "material": "gripper_pad_mat",
            "friction": "3.0 0.08 0.006",
            "condim": "4",
            "solref": "0.006 1",
            "solimp": "0.95 0.99 0.002",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    return finger


def _ensure_material(asset: ET.Element, name: str, rgba: str) -> None:
    if asset.find(f"material[@name='{name}']") is None:
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _remove_roundtrip_targets(world: ET.Element) -> None:
    target_body_names = {"target_marker", "target_marker_b"}
    target_node_names = {"target_sphere", "target_sphere_b", "target_site", "target_site_b"}
    for parent in world.iter():
        for child in list(parent):
            child_name = child.attrib.get("name", "")
            if child.tag == "body" and child_name in target_body_names:
                parent.remove(child)
            elif child.tag in {"geom", "site"} and child_name in target_node_names:
                parent.remove(child)


def _ensure_position_actuator(
    actuator: ET.Element,
    name: str,
    joint: str,
    ctrlrange: str,
) -> None:
    if actuator.find(f"position[@name='{name}']") is not None:
        return
    ET.SubElement(
        actuator,
        "position",
        {
            "name": name,
            "joint": joint,
            "ctrlrange": ctrlrange,
            "kp": "120",
            "kv": "8",
            "forcerange": "-30 30",
        },
    )


def _append_gripper_keyframe_values(root: ET.Element) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        return
    for key in keyframe.findall("key"):
        for attr in ("qpos", "ctrl"):
            values = key.attrib.get(attr)
            if values is None:
                continue
            parts = values.split()
            if len(parts) == 6:
                parts.extend([f"{GRIPPER_OPEN_QPOS:g}", f"{GRIPPER_OPEN_QPOS:g}"])
                key.set(attr, " ".join(parts))


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


if __name__ == "__main__":
    path = write_gripper_model()
    print(path)
