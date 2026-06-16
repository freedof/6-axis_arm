from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_simplified.xml"
ROBOT_DOF = 6
GRIPPER_OPENING = 0.025


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a MuJoCo viewer for the simplified Dobot CR5 model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hold", action="store_true", help="Hold the ready pose instead of sweeping joints.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    ready = [0.0, -0.6, 0.85, 0.0, -0.8, 0.0]
    _set_initial_state(model, data, ready)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            if args.hold:
                data.ctrl[:ROBOT_DOF] = ready
                _hold_gripper_open(model, data)
            else:
                data.ctrl[:ROBOT_DOF] = [
                    0.35 * math.sin(0.6 * t),
                    -0.60 + 0.25 * math.sin(0.45 * t),
                    0.85 + 0.20 * math.sin(0.50 * t),
                    0.35 * math.sin(0.70 * t),
                    -0.80 + 0.20 * math.sin(0.65 * t),
                    0.80 * math.sin(0.90 * t),
                ]
                _animate_gripper(model, data, t)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def _set_initial_state(model: mujoco.MjModel, data: mujoco.MjData, ready: list[float]) -> None:
    data.qpos[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[:ROBOT_DOF] = ready
    data.ctrl[:ROBOT_DOF] = ready
    if model.nq > ROBOT_DOF:
        data.qpos[ROBOT_DOF:] = GRIPPER_OPENING
    _hold_gripper_open(model, data)


def _hold_gripper_open(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nu > ROBOT_DOF:
        data.ctrl[ROBOT_DOF:] = GRIPPER_OPENING


def _animate_gripper(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
    if model.nu > ROBOT_DOF:
        opening = 0.0175 * (1.0 + math.sin(0.8 * t))
        data.ctrl[ROBOT_DOF:] = opening


if __name__ == "__main__":
    main()
