from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model


DEFAULT_PICK_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_gripper_pick_scene.xml"
DEFAULT_MULTI_OBJECT_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_gripper_multi_object_scene.xml"
TABLE_TOP_Z = 0.035
TABLE_CENTER = (0.35, -0.55)
TABLE_HALF_SIZE = (0.34, 0.28, 0.0175)
CUBE_HALF_SIZE = (0.016, 0.018, 0.030)
CUBE_CENTER = (0.35, -0.55, TABLE_TOP_Z + CUBE_HALF_SIZE[2])
DEFAULT_OBJECT_MASS = 0.020
DEFAULT_OBJECT_FRICTION = (3.0, 0.08, 0.006)
TRAY_CENTER = (0.55, -0.47)
TRAY_HALF_SIZE = (0.115, 0.075, 0.006)
TRAY_WALL_THICKNESS = 0.006
TRAY_WALL_HEIGHT = 0.026
TRAY_FLOOR_TOP_Z = TABLE_TOP_Z + TRAY_HALF_SIZE[2] * 2.0
TRAY_PLACE_SLOTS = (
    (0.49, -0.43),
    (0.61, -0.51),
    (0.49, -0.51),
    (0.61, -0.43),
)


@dataclass(frozen=True)
class TableObjectSpec:
    name: str
    shape: str
    color: str
    rgba: tuple[float, float, float, float]
    position_xy: tuple[float, float]
    size: tuple[float, ...]
    mass: float = DEFAULT_OBJECT_MASS
    friction: tuple[float, float, float] = DEFAULT_OBJECT_FRICTION

    @property
    def center_z(self) -> float:
        if self.shape == "box":
            return TABLE_TOP_Z + self.size[2]
        if self.shape == "cylinder":
            return TABLE_TOP_Z + self.size[1]
        raise ValueError(f"Unsupported object shape: {self.shape}")


DEFAULT_MULTI_OBJECT_SPECS = (
    TableObjectSpec("red_cube", "box", "red", (0.88, 0.16, 0.10, 1.0), (0.35, -0.55), (0.016, 0.018, 0.030)),
    TableObjectSpec("blue_cube", "box", "blue", (0.12, 0.32, 0.86, 1.0), (0.26, -0.51), (0.017, 0.017, 0.026)),
    TableObjectSpec("green_cylinder", "cylinder", "green", (0.12, 0.64, 0.28, 1.0), (0.39, -0.46), (0.018, 0.028)),
    TableObjectSpec("yellow_cylinder", "cylinder", "yellow", (0.95, 0.74, 0.12, 1.0), (0.29, -0.64), (0.016, 0.024)),
    TableObjectSpec("purple_cube", "box", "purple", (0.54, 0.22, 0.80, 1.0), (0.45, -0.63), (0.015, 0.019, 0.025)),
)


def write_pick_scene_model(
    output_path: Path = DEFAULT_PICK_MODEL,
    *,
    source_model: Path = DEFAULT_GRIPPER_MODEL,
) -> Path:
    return write_multi_object_scene_model(
        output_path,
        source_model=source_model,
        objects=(
            TableObjectSpec(
                name="grasp_cube",
                shape="box",
                color="red",
                rgba=(0.88, 0.16, 0.10, 1.0),
                position_xy=(CUBE_CENTER[0], CUBE_CENTER[1]),
                size=CUBE_HALF_SIZE,
            ),
        ),
        model_name="dobot_cr5_gripper_pick_scene",
        include_tray=False,
    )


def write_multi_object_scene_model(
    output_path: Path = DEFAULT_MULTI_OBJECT_MODEL,
    *,
    source_model: Path = DEFAULT_GRIPPER_MODEL,
    objects: tuple[TableObjectSpec, ...] | list[TableObjectSpec] = DEFAULT_MULTI_OBJECT_SPECS,
    model_name: str = "dobot_cr5_gripper_multi_object_scene",
    include_tray: bool = True,
) -> Path:
    write_gripper_model(source_model)

    specs = tuple(objects)
    _validate_object_specs(specs)

    tree = ET.parse(source_model)
    root = tree.getroot()
    root.set("model", model_name)

    asset = root.find("asset")
    if asset is not None:
        _ensure_material(asset, "pick_table_mat", "0.55 0.57 0.54 1")
        if include_tray:
            _ensure_material(asset, "tray_mat", "0.12 0.13 0.14 1")
        for spec in specs:
            _ensure_material(asset, f"{spec.name}_mat", _fmt_vec(spec.rgba))

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError(f"Model has no worldbody: {source_model}")

    _tune_pick_actuators(root)
    _remove_roundtrip_targets(world)
    _add_pick_table(world)
    if include_tray:
        _add_tray(world)
    for spec in specs:
        _add_table_object(world, spec)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode")
    return output_path



def tray_metadata() -> dict:
    return {
        "name": "tabletop_tray",
        "center_xy_m": [round(TRAY_CENTER[0], 6), round(TRAY_CENTER[1], 6)],
        "floor_top_z_m": round(TRAY_FLOOR_TOP_Z, 6),
        "half_size_m": [round(value, 6) for value in TRAY_HALF_SIZE],
        "wall_height_m": round(TRAY_WALL_HEIGHT, 6),
        "place_slots_xy_m": [[round(x, 6), round(y, 6)] for x, y in TRAY_PLACE_SLOTS],
    }

def object_specs_from_config(objects: list[dict]) -> tuple[TableObjectSpec, ...]:
    specs = []
    for item in objects:
        shape = str(item.get("shape", "box")).lower()
        if shape not in {"box", "cylinder"}:
            raise ValueError(f"Unsupported object shape: {shape}")
        rgba = tuple(float(value) for value in item["rgba"])
        position_xy = tuple(float(value) for value in item["position_xy"])
        if shape == "box":
            size = tuple(float(value) for value in item["half_size_m"])
        else:
            size = (float(item["radius_m"]), float(item["half_height_m"]))
        specs.append(
            TableObjectSpec(
                name=str(item["name"]),
                shape=shape,
                color=str(item.get("color", item["name"])),
                rgba=rgba,  # type: ignore[arg-type]
                position_xy=position_xy,  # type: ignore[arg-type]
                size=size,
                mass=float(item.get("mass_kg", DEFAULT_OBJECT_MASS)),
                friction=tuple(float(value) for value in item.get("friction", DEFAULT_OBJECT_FRICTION)),  # type: ignore[arg-type]
            )
        )
    return tuple(specs)


def object_specs_to_dicts(objects: tuple[TableObjectSpec, ...] | list[TableObjectSpec]) -> list[dict]:
    result = []
    for spec in objects:
        data = {
            "name": spec.name,
            "shape": spec.shape,
            "color": spec.color,
            "rgba": [round(value, 4) for value in spec.rgba],
            "position_xy_m": [round(value, 6) for value in spec.position_xy],
            "initial_center_m": [round(spec.position_xy[0], 6), round(spec.position_xy[1], 6), round(spec.center_z, 6)],
            "mass_kg": round(spec.mass, 6),
            "friction": [round(value, 6) for value in spec.friction],
            "free_joint": True,
        }
        if spec.shape == "box":
            data["half_size_m"] = [round(value, 6) for value in spec.size]
        else:
            data["radius_m"] = round(spec.size[0], 6)
            data["half_height_m"] = round(spec.size[1], 6)
        result.append(data)
    return result


def _add_pick_table(world: ET.Element) -> None:
    ET.SubElement(
        world,
        "geom",
        {
            "name": "pick_table",
            "type": "box",
            "pos": f"{TABLE_CENTER[0]:g} {TABLE_CENTER[1]:g} {TABLE_TOP_Z / 2.0:g}",
            "size": _fmt_vec(TABLE_HALF_SIZE),
            "material": "pick_table_mat",
            "friction": "1.0 0.02 0.002",
            "contype": "1",
            "conaffinity": "1",
        },
    )



def _add_tray(world: ET.Element) -> None:
    cx, cy = TRAY_CENTER
    base_z = TABLE_TOP_Z + TRAY_HALF_SIZE[2]
    wall_z = TABLE_TOP_Z + TRAY_HALF_SIZE[2] * 2.0 + TRAY_WALL_HEIGHT / 2.0
    ET.SubElement(
        world,
        "geom",
        {
            "name": "tabletop_tray_floor",
            "type": "box",
            "pos": f"{cx:g} {cy:g} {base_z:g}",
            "size": _fmt_vec(TRAY_HALF_SIZE),
            "material": "tray_mat",
            "friction": "1.2 0.02 0.002",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    wall_specs = (
        ("front", (cx + TRAY_HALF_SIZE[0], cy, wall_z), (TRAY_WALL_THICKNESS, TRAY_HALF_SIZE[1] + TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT / 2.0)),
        ("back", (cx - TRAY_HALF_SIZE[0], cy, wall_z), (TRAY_WALL_THICKNESS, TRAY_HALF_SIZE[1] + TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT / 2.0)),
        ("left", (cx, cy - TRAY_HALF_SIZE[1], wall_z), (TRAY_HALF_SIZE[0], TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT / 2.0)),
        ("right", (cx, cy + TRAY_HALF_SIZE[1], wall_z), (TRAY_HALF_SIZE[0], TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT / 2.0)),
    )
    for suffix, pos, size in wall_specs:
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"tabletop_tray_wall_{suffix}",
                "type": "box",
                "pos": _fmt_vec(pos),
                "size": _fmt_vec(size),
                "material": "tray_mat",
                "friction": "1.2 0.02 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )

def _add_table_object(world: ET.Element, spec: TableObjectSpec) -> None:
    body = ET.SubElement(
        world,
        "body",
        {
            "name": spec.name,
            "pos": f"{spec.position_xy[0]:g} {spec.position_xy[1]:g} {spec.center_z:g}",
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"{spec.name}_free"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{spec.name}_geom",
            "type": spec.shape,
            "size": _fmt_vec(spec.size),
            "material": f"{spec.name}_mat",
            "rgba": _fmt_vec(spec.rgba),
            "mass": f"{spec.mass:g}",
            "friction": _fmt_vec(spec.friction),
            "condim": "4",
            "solref": "0.006 1",
            "solimp": "0.95 0.99 0.002",
            "contype": "1",
            "conaffinity": "1",
        },
    )


def _validate_object_specs(objects: tuple[TableObjectSpec, ...]) -> None:
    if not objects:
        raise ValueError("A table object scene requires at least one object.")
    names = set()
    for spec in objects:
        if spec.name in names:
            raise ValueError(f"Duplicate object name: {spec.name}")
        names.add(spec.name)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec.name):
            raise ValueError(f"Object name must be a valid MJCF identifier: {spec.name}")
        if spec.shape not in {"box", "cylinder"}:
            raise ValueError(f"Unsupported object shape: {spec.shape}")
        if len(spec.rgba) != 4:
            raise ValueError(f"RGBA must contain four values: {spec.name}")
        if len(spec.position_xy) != 2:
            raise ValueError(f"position_xy must contain two values: {spec.name}")
        if spec.shape == "box" and len(spec.size) != 3:
            raise ValueError(f"Box size must be half-size xyz: {spec.name}")
        if spec.shape == "cylinder" and len(spec.size) != 2:
            raise ValueError(f"Cylinder size must be radius and half-height: {spec.name}")
        if spec.mass <= 0.0:
            raise ValueError(f"Object mass must be positive: {spec.name}")
        if any(value <= 0.0 for value in spec.size):
            raise ValueError(f"Object size values must be positive: {spec.name}")
        _validate_inside_table(spec)


def _validate_inside_table(spec: TableObjectSpec) -> None:
    x, y = spec.position_xy
    margin_x = spec.size[0]
    margin_y = spec.size[1] if spec.shape == "box" else spec.size[0]
    if not (TABLE_CENTER[0] - TABLE_HALF_SIZE[0] + margin_x <= x <= TABLE_CENTER[0] + TABLE_HALF_SIZE[0] - margin_x):
        raise ValueError(f"Object x position is outside the table bounds: {spec.name}")
    if not (TABLE_CENTER[1] - TABLE_HALF_SIZE[1] + margin_y <= y <= TABLE_CENTER[1] + TABLE_HALF_SIZE[1] - margin_y):
        raise ValueError(f"Object y position is outside the table bounds: {spec.name}")


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


def _fmt_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:g}" for value in values)


if __name__ == "__main__":
    path = write_pick_scene_model()
    multi_path = write_multi_object_scene_model()
    print(path)
    print(multi_path)
    print(tray_metadata())
