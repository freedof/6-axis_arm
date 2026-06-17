from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.demo_xyz_joint_roundtrip import DEFAULT_MODEL as ROUNDTRIP_MODEL
from src.sim.d435i_model import (
    DEFAULT_D435I_GRIPPER_MODEL,
    DEFAULT_D435I_PICK_MODEL,
    write_d435i_gripper_model,
    write_d435i_pick_scene_model,
)
from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model
from src.sim.gripper_pick_scene import DEFAULT_PICK_MODEL, write_pick_scene_model
from src.sim.planning_model import DEFAULT_PICK_PLANNING_MODEL, DEFAULT_PLANNING_MODEL, write_planning_model


ROUNDTRIP_TARGETS = {"target_marker", "target_marker_b", "target_sphere", "target_sphere_b"}


def main() -> None:
    gripper_model = write_gripper_model(DEFAULT_GRIPPER_MODEL)
    pick_model = write_pick_scene_model(DEFAULT_PICK_MODEL)
    d435i_gripper_model = write_d435i_gripper_model(DEFAULT_D435I_GRIPPER_MODEL)
    d435i_pick_model = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    roundtrip_planning_model = write_planning_model(DEFAULT_PLANNING_MODEL)
    pick_planning_model = write_planning_model(DEFAULT_PICK_PLANNING_MODEL, source_model=pick_model)

    roundtrip_family = {
        "roundtrip": ROUNDTRIP_MODEL,
        "roundtrip_planning": roundtrip_planning_model,
    }
    pick_family = {
        "gripper": gripper_model,
        "pick": pick_model,
        "d435i_gripper": d435i_gripper_model,
        "d435i_pick": d435i_pick_model,
        "pick_planning": pick_planning_model,
    }

    for scene_id, path in roundtrip_family.items():
        missing = sorted(ROUNDTRIP_TARGETS - _names(path))
        if missing:
            raise RuntimeError(f"{scene_id} should keep roundtrip targets, missing {missing}: {path}")

    for scene_id, path in pick_family.items():
        present = sorted(ROUNDTRIP_TARGETS & _names(path))
        if present:
            raise RuntimeError(f"{scene_id} should not contain roundtrip targets, found {present}: {path}")

    for scene_id, path in {
        "pick": pick_model,
        "d435i_pick": d435i_pick_model,
    }.items():
        names = _names(path)
        if "grasp_cube" not in names or "grasp_cube_geom" not in names:
            raise RuntimeError(f"{scene_id} should contain the grasp cube: {path}")

    print("roundtrip_family:", {key: str(path) for key, path in roundtrip_family.items()})
    print("pick_family:", {key: str(path) for key, path in pick_family.items()})
    print("status: OK")


def _names(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    return {element.attrib["name"] for element in root.iter() if "name" in element.attrib}


if __name__ == "__main__":
    main()
