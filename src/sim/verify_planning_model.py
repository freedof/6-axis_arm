from __future__ import annotations

import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.planning_model import DEFAULT_PLANNING_MODEL, write_planning_model


EXPECTED_COLLISION_GEOMS = (
    "floor_collision",
    "target_sphere",
    "target_sphere_b",
    "collision_base",
    "collision_link1",
    "collision_link2",
    "collision_link3",
    "collision_link4",
    "collision_link5",
    "collision_tool",
)


EXPECTED_VISUAL_GEOMS = (
    "floor",
    "base_column",
    "base_top",
    "link1_joint",
    "link1_body",
    "link2_joint",
    "link2_arm",
    "link3_joint",
    "link3_arm",
    "link4_joint",
    "link4_body",
    "link5_joint",
    "link5_body",
    "link6_joint",
    "tool_stub",
)


def main() -> None:
    model_path = write_planning_model(DEFAULT_PLANNING_MODEL)
    model = mujoco.MjModel.from_xml_path(str(model_path))

    for name in EXPECTED_COLLISION_GEOMS:
        geom_id = _geom_id(model, name)
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        print(f"{name}: contype={contype}, conaffinity={conaffinity}")
        if contype == 0 or conaffinity == 0:
            raise RuntimeError(f"Expected collision geom to be enabled: {name}")

    for name in EXPECTED_VISUAL_GEOMS:
        geom_id = _geom_id(model, name)
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        print(f"{name}: contype={contype}, conaffinity={conaffinity}")
        if contype != 0 or conaffinity != 0:
            raise RuntimeError(f"Expected visual geom to be collision-disabled: {name}")

    print(f"model: {model_path}")
    print("status: OK")


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"Missing geom: {name}")
    return geom_id


if __name__ == "__main__":
    main()
