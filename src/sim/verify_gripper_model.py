from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model


READY_QPOS = np.array([0.0, -0.6, 0.85, 0.0, -0.8, 0.0])
OPENING = 0.030


def main() -> None:
    model_path = write_gripper_model(DEFAULT_GRIPPER_MODEL)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"model: {model_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")
    print("joints:", ", ".join(_joint_names(model)))
    print("actuators:", ", ".join(_actuator_names(model)))

    if model.nq != 8 or model.nv != 8 or model.nu != 8:
        raise RuntimeError("Gripper model should expose 8 qpos, 8 nv, and 8 actuators.")

    _require_joint(model, "gripper_left_slide")
    _require_joint(model, "gripper_right_slide")
    _require_site(model, "tool0")
    _require_site(model, "gripper_mount")
    _require_site(model, "gripper_tcp")
    _require_body(model, "gripper_left_finger")
    _require_body(model, "gripper_right_finger")

    left_adr = _joint_qpos_address(model, "gripper_left_slide")
    right_adr = _joint_qpos_address(model, "gripper_right_slide")
    if left_adr != 6 or right_adr != 7:
        raise RuntimeError(f"Unexpected gripper qpos addresses: left={left_adr}, right={right_adr}")

    data.qpos[:6] = READY_QPOS
    data.qpos[left_adr] = 0.0
    data.qpos[right_adr] = 0.0
    data.ctrl[:6] = READY_QPOS
    data.ctrl[6:] = [0.0, 0.0]
    mujoco.mj_forward(model, data)
    closed_gap = _finger_distance(model, data)
    tcp_offset = np.linalg.norm(_site_position(model, data, "gripper_tcp") - _site_position(model, data, "tool0"))

    data.qpos[left_adr] = OPENING
    data.qpos[right_adr] = OPENING
    mujoco.mj_forward(model, data)
    open_gap = _finger_distance(model, data)

    if open_gap <= closed_gap + 0.04:
        raise RuntimeError(f"Expected fingers to open wider: closed={closed_gap}, open={open_gap}")
    if not (0.09 <= tcp_offset <= 0.13):
        raise RuntimeError(f"Unexpected gripper TCP offset from tool0: {tcp_offset}")

    data.qpos[:6] = READY_QPOS
    data.qpos[left_adr] = 0.0
    data.qpos[right_adr] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:6] = READY_QPOS
    data.ctrl[6:] = [OPENING, OPENING]
    for _ in range(350):
        mujoco.mj_step(model, data)

    final_opening = data.qpos[[left_adr, right_adr]].copy()
    if np.min(final_opening) < 0.020:
        raise RuntimeError(f"Gripper position actuators did not open far enough: {final_opening}")

    print(f"closed_finger_distance_m={closed_gap:.6f}")
    print(f"open_finger_distance_m={open_gap:.6f}")
    print(f"tool0_to_gripper_tcp_m={tcp_offset:.6f}")
    print("final_gripper_qpos:", np.round(final_opening, 5))
    print("status: OK")


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


def _actuator_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]


def _require_joint(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"Missing joint: {name}")
    return joint_id


def _require_site(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise RuntimeError(f"Missing site: {name}")
    return site_id


def _require_body(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {name}")
    return body_id


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[_require_joint(model, name)])


def _site_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.site_xpos[_require_site(model, name)].copy()


def _finger_distance(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    left = data.xpos[_require_body(model, "gripper_left_finger")]
    right = data.xpos[_require_body(model, "gripper_right_finger")]
    return float(np.linalg.norm(left - right))


if __name__ == "__main__":
    main()
