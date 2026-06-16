from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.demo_xyz_joint_roundtrip import DEFAULT_MODEL


DEFAULT_PLANNING_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_planning.xml"


COLLISION_GEOMS: dict[str, list[dict[str, str]]] = {
    "base_link": [
        {
            "name": "collision_base",
            "type": "cylinder",
            "size": "0.095 0.075",
            "pos": "0 0 0.075",
        },
    ],
    "Link1": [
        {
            "name": "collision_link1",
            "type": "capsule",
            "fromto": "0 0 0.025 0 0 0.150",
            "size": "0.045",
        },
    ],
    "Link2": [
        {
            "name": "collision_link2",
            "type": "capsule",
            "fromto": "0 0 0 -0.427 0 0",
            "size": "0.040",
        },
    ],
    "Link3": [
        {
            "name": "collision_link3",
            "type": "capsule",
            "fromto": "0 0 0 -0.357 0 0.141",
            "size": "0.035",
        },
    ],
    "Link4": [
        {
            "name": "collision_link4",
            "type": "capsule",
            "fromto": "0 0 0 0 -0.116 0",
            "size": "0.030",
        },
    ],
    "Link5": [
        {
            "name": "collision_link5",
            "type": "capsule",
            "fromto": "0 0 0 0 0.105 0",
            "size": "0.026",
        },
    ],
    "Link6": [
        {
            "name": "collision_tool",
            "type": "capsule",
            "fromto": "0 0 0 0 0 0.08",
            "size": "0.020",
        },
    ],
}


def write_planning_model(
    output_path: Path = DEFAULT_PLANNING_MODEL,
    *,
    source_model: Path = DEFAULT_MODEL,
) -> Path:
    tree = ET.parse(source_model)
    root = tree.getroot()
    root.set("model", "dobot_cr5_planning")

    asset = root.find("asset")
    if asset is not None and asset.find("material[@name='collision_mat']") is None:
        ET.SubElement(asset, "material", {"name": "collision_mat", "rgba": "0.10 0.95 0.35 0.18"})

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"Model has no worldbody: {source_model}")

    _disable_visual_geoms(root)
    _replace_floor_with_collision_floor(world)
    _add_robot_collision_geoms(world)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="unicode")
    return output_path


def _disable_visual_geoms(root: ET.Element) -> None:
    for geom in root.iter("geom"):
        name = geom.attrib.get("name", "")
        if _is_planning_collision_geom(name):
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("group", "2")
            continue
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")


def _replace_floor_with_collision_floor(world: ET.Element) -> None:
    floor = world.find("geom[@name='floor']")
    if floor is not None:
        floor.set("contype", "0")
        floor.set("conaffinity", "0")
        floor.set("group", "1")
    if world.find("geom[@name='floor_collision']") is None:
        ET.SubElement(
            world,
            "geom",
            {
                "name": "floor_collision",
                "type": "plane",
                "pos": "0 0 0",
                "size": "1.8 1.8 0.02",
                "rgba": "0.2 0.7 0.2 0.20",
                "contype": "1",
                "conaffinity": "1",
                "group": "2",
            },
        )


def _add_robot_collision_geoms(world: ET.Element) -> None:
    bodies_by_name = {
        body.attrib["name"]: body
        for body in world.iter("body")
        if "name" in body.attrib
    }
    for body_name, geoms in COLLISION_GEOMS.items():
        body = bodies_by_name.get(body_name)
        if body is None:
            raise RuntimeError(f"Missing body for collision geometry: {body_name}")
        for spec in geoms:
            if body.find(f"geom[@name='{spec['name']}']") is not None:
                continue
            attrs = {
                **spec,
                "material": "collision_mat",
                "contype": "1",
                "conaffinity": "1",
                "group": "2",
            }
            ET.SubElement(body, "geom", attrs)


def _is_planning_collision_geom(name: str) -> bool:
    return (
        name.startswith("collision_")
        or name.endswith("_collision")
        or name.startswith("planning_obstacle")
        or name in {"target_sphere", "target_sphere_b"}
    )


if __name__ == "__main__":
    path = write_planning_model()
    print(path)
