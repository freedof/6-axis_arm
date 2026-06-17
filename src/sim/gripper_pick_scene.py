from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model


DEFAULT_PICK_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_gripper_pick_scene.xml"
TABLE_TOP_Z = 0.035
CUBE_HALF_SIZE = (0.016, 0.018, 0.030)
CUBE_CENTER = (0.35, -0.55, TABLE_TOP_Z + CUBE_HALF_SIZE[2])


def write_pick_scene_model(
    output_path: Path = DEFAULT_PICK_MODEL,
    *,
    source_model: Path = DEFAULT_GRIPPER_MODEL,
) -> Path:
    write_gripper_model(source_model)

    tree = ET.parse(source_model)
    root = tree.getroot()
    root.set("model", "dobot_cr5_gripper_pick_scene")

    asset = root.find("asset")
    if asset is not None:
        _ensure_material(asset, "pick_table_mat", "0.55 0.57 0.54 1")
        _ensure_material(asset, "pick_cube_mat", "0.88 0.16 0.10 1")

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"Model has no worldbody: {source_model}")

    _tune_pick_actuators(root)
    _remove_roundtrip_targets(world)

    if world.find("geom[@name='pick_table']") is None:
        ET.SubElement(
            world,
            "geom",
            {
                "name": "pick_table",
                "type": "box",
                "pos": f"0.35 -0.55 {TABLE_TOP_Z / 2.0:g}",
                "size": "0.34 0.28 0.0175",
                "material": "pick_table_mat",
                "friction": "1.0 0.02 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    if world.find("body[@name='grasp_cube']") is None:
        cube = ET.SubElement(
            world,
            "body",
            {
                "name": "grasp_cube",
                "pos": f"{CUBE_CENTER[0]:g} {CUBE_CENTER[1]:g} {CUBE_CENTER[2]:g}",
            },
        )
        ET.SubElement(cube, "freejoint", {"name": "grasp_cube_free"})
        ET.SubElement(
            cube,
            "geom",
            {
                "name": "grasp_cube_geom",
                "type": "box",
                "size": f"{CUBE_HALF_SIZE[0]:g} {CUBE_HALF_SIZE[1]:g} {CUBE_HALF_SIZE[2]:g}",
                "material": "pick_cube_mat",
                "rgba": "0.88 0.16 0.10 1",
                "mass": "0.020",
                "friction": "3.0 0.08 0.006",
                "condim": "4",
                "solref": "0.006 1",
                "solimp": "0.95 0.99 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode")
    return output_path


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


def _tune_pick_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        return
    for position in actuator.findall("position"):
        name = position.attrib.get("name", "")
        if name.startswith("joint"):
            position.set("kp", "420")
            position.set("kv", "60")
            position.set("forcerange", "-320 320")
        elif name.startswith("gripper"):
            position.set("kp", "220")
            position.set("kv", "12")
            position.set("forcerange", "-80 80")


if __name__ == "__main__":
    path = write_pick_scene_model()
    print(path)
