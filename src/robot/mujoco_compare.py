from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_simplified.xml"


def mujoco_tool_pose(q: np.ndarray, model_path: Path = DEFAULT_MJCF, site_name: str = "tool0") -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(q, dtype=float)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Missing MuJoCo site: {site_name}")

    pose = np.eye(4, dtype=float)
    pose[:3, 3] = data.site_xpos[site_id]
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return pose
