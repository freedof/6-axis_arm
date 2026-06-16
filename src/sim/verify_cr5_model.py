from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_simplified.xml"


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def site_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise RuntimeError(f"Missing MuJoCo site: {name}")
    return data.site_xpos[site_id].copy()


def verify_model(model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"model: {model_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")
    print("joints:", ", ".join(joint_names(model)))

    if model.nq != 6 or model.nu != 6:
        raise RuntimeError("CR5 simplified model should expose 6 qpos and 6 actuators.")

    samples = [
        np.zeros(6),
        np.array([0.0, -0.6, 0.85, 0.0, -0.8, 0.0]),
        np.array([0.5, -0.8, 0.7, 0.4, -0.6, 1.2]),
    ]

    for idx, qpos in enumerate(samples, start=1):
        data.qpos[:] = qpos
        data.ctrl[:] = qpos
        mujoco.mj_forward(model, data)
        tool = site_position(model, data, "tool0")
        print(f"sample {idx}: q={np.round(qpos, 4)} tool0={np.round(tool, 4)}")

    data.qpos[:] = samples[0]
    data.ctrl[:] = samples[0]
    for step in range(500):
        t = step * model.opt.timestep
        data.ctrl[:] = [
            0.30 * math.sin(1.0 * t),
            -0.45 + 0.20 * math.sin(0.7 * t),
            0.70 + 0.18 * math.sin(0.9 * t),
            0.25 * math.sin(1.3 * t),
            -0.50 + 0.15 * math.sin(1.1 * t),
            0.60 * math.sin(1.7 * t),
        ]
        mujoco.mj_step(model, data)

    print("simulation_steps=500")
    print("final_qpos:", np.round(data.qpos.copy(), 4))
    print("final_tool0:", np.round(site_position(model, data, "tool0"), 4))
    print("status: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless load and motion smoke test for the simplified Dobot CR5 MuJoCo model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    verify_model(args.model)


if __name__ == "__main__":
    main()
