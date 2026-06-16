from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.robot.model import SerialRobotModel


DEFAULT_IGNORED_GEOMS: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContactInfo:
    geom1: str
    geom2: str
    body1: str
    body2: str
    distance: float


@dataclass(frozen=True)
class CollisionResult:
    within_limits: bool
    contacts: tuple[ContactInfo, ...]

    @property
    def collision_free(self) -> bool:
        return not self.contacts

    @property
    def valid(self) -> bool:
        return self.within_limits and self.collision_free


class MujocoCollisionChecker:
    """MuJoCo-backed joint-state collision checker.

    The checker keeps kinematics and planning code independent from MuJoCo while
    still using MuJoCo's narrow-phase collision engine for state validity.
    """

    def __init__(
        self,
        model_path: Path,
        robot: SerialRobotModel,
        *,
        ignored_geom_names: tuple[str, ...] = DEFAULT_IGNORED_GEOMS,
        minimum_distance: float = 0.0,
        contact_tolerance: float = 1e-6,
        ignore_robot_self_collisions: bool = True,
        enable_all_non_ignored_geoms: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.robot = robot
        self.minimum_distance = float(minimum_distance)
        self.contact_tolerance = float(contact_tolerance)
        self.robot_body_names = frozenset(
            body
            for joint in robot.joints
            for body in (joint.parent, joint.child)
        )
        self.ignore_robot_self_collisions = bool(ignore_robot_self_collisions)
        self.enable_all_non_ignored_geoms = bool(enable_all_non_ignored_geoms)
        xml = self._collision_enabled_xml(ignored_geom_names)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        if self.model.nq < robot.dof:
            raise ValueError(f"Model nq={self.model.nq} is smaller than robot dof={robot.dof}.")

        self.ignored_geom_ids = frozenset(
            geom_id
            for name in ignored_geom_names
            if (geom_id := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        )
        self._configure_collision_masks()

    def _collision_enabled_xml(self, ignored_geom_names: tuple[str, ...]) -> str:
        tree = ET.parse(self.model_path)
        root = tree.getroot()
        ignored = set(ignored_geom_names)
        for geom in root.iter("geom"):
            name = geom.attrib.get("name", "")
            if name in ignored:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            elif self.enable_all_non_ignored_geoms or _is_planning_collision_geom(name):
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                if self.minimum_distance > 0.0:
                    geom.set("margin", str(self.minimum_distance))
            elif self.minimum_distance > 0.0 and _geom_is_enabled(geom):
                geom.set("margin", str(self.minimum_distance))
        return ET.tostring(root, encoding="unicode")

    def _configure_collision_masks(self) -> None:
        for geom_id in range(self.model.ngeom):
            if geom_id in self.ignored_geom_ids:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
            elif self.enable_all_non_ignored_geoms or _is_planning_collision_geom(self._geom_name(geom_id)):
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 1
                if self.minimum_distance > 0.0:
                    self.model.geom_margin[geom_id] = max(
                        float(self.model.geom_margin[geom_id]),
                        self.minimum_distance,
                    )

    def check(self, q: np.ndarray) -> CollisionResult:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.robot.dof,):
            raise ValueError(f"Expected q shape {(self.robot.dof,)}, got {q.shape}")

        within_limits = bool(
            np.all(q >= self.robot.lower_limits - self.contact_tolerance)
            and np.all(q <= self.robot.upper_limits + self.contact_tolerance)
        )
        if not within_limits:
            return CollisionResult(False, ())

        self.data.qpos[: self.robot.dof] = q
        if self.model.nu >= self.robot.dof:
            self.data.ctrl[: self.robot.dof] = q
        mujoco.mj_forward(self.model, self.data)

        contacts: list[ContactInfo] = []
        threshold = self.minimum_distance - self.contact_tolerance
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 in self.ignored_geom_ids or contact.geom2 in self.ignored_geom_ids:
                continue
            if float(contact.dist) >= threshold:
                continue
            body1 = self._body_name_for_geom(contact.geom1)
            body2 = self._body_name_for_geom(contact.geom2)
            if self.ignore_robot_self_collisions and body1 in self.robot_body_names and body2 in self.robot_body_names:
                continue
            contacts.append(
                ContactInfo(
                    geom1=self._geom_name(contact.geom1),
                    geom2=self._geom_name(contact.geom2),
                    body1=body1,
                    body2=body2,
                    distance=float(contact.dist),
                )
            )

        return CollisionResult(True, tuple(contacts))

    def is_state_valid(self, q: np.ndarray) -> bool:
        return self.check(q).valid

    def is_edge_valid(self, q_from: np.ndarray, q_to: np.ndarray, *, resolution: float = 0.04) -> bool:
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        distance = float(np.linalg.norm(q_to - q_from))
        steps = max(1, int(np.ceil(distance / resolution)))
        for step in range(steps + 1):
            alpha = step / steps
            q = (1.0 - alpha) * q_from + alpha * q_to
            if not self.is_state_valid(q):
                return False
        return True

    def _geom_name(self, geom_id: int) -> str:
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return name if name is not None else f"geom#{geom_id}"

    def _body_name_for_geom(self, geom_id: int) -> str:
        body_id = int(self.model.geom_bodyid[geom_id])
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name if name is not None else f"body#{body_id}"


def _is_planning_collision_geom(name: str) -> bool:
    return (
        name.startswith("collision_")
        or name.endswith("_collision")
        or name.startswith("planning_obstacle")
        or name in {"target_sphere", "target_sphere_b"}
    )


def _geom_is_enabled(geom: ET.Element) -> bool:
    return geom.attrib.get("contype", "0") != "0" or geom.attrib.get("conaffinity", "0") != "0"
