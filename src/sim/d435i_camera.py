from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class D435iParams:
    baseline_mm: float = 50.0
    hfov_deg: float = 90.0
    vfov_deg: float = 65.0
    depth_resolution: tuple[int, int] = (848, 480)
    depth_scale: float = 0.001
    min_z_m: float = 0.17
    max_z_m: float = 10.0
    subpixel_rms: float = 0.08
    focal_length_pix: float = 424.0

    accel_rate_hz: int = 250
    gyro_rate_hz: int = 400
    imu_pos_offset_m: tuple[float, float, float] = (-0.0055, 0.005, 0.012)

    def depth_rms_m(self, distance_m: np.ndarray) -> np.ndarray:
        baseline_m = self.baseline_mm / 1000.0
        return (distance_m**2 * self.subpixel_rms) / (self.focal_length_pix * baseline_m)


class D435iCamera:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        camera_name: str = "d435i_depth",
        width: int | None = None,
        height: int | None = None,
        seed: int = 7,
        params: D435iParams | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.params = D435iParams() if params is None else params
        self.width = int(width or self.params.depth_resolution[0])
        self.height = int(height or self.params.depth_resolution[1])
        self.rng = np.random.default_rng(seed)

        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise RuntimeError(f"Missing MuJoCo camera: {camera_name}")

        if model.vis.global_.offwidth < self.width:
            model.vis.global_.offwidth = self.width
        if model.vis.global_.offheight < self.height:
            model.vis.global_.offheight = self.height

        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def close(self) -> None:
        self.renderer.close()

    def get_intrinsics(self) -> np.ndarray:
        fovy = float(self.model.cam_fovy[self.camera_id])
        focal = 0.5 * self.height / np.tan(np.deg2rad(fovy) * 0.5)
        cx = self.width / 2.0
        cy = self.height / 2.0
        return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=float)

    def get_pose_matrix(self) -> np.ndarray:
        cam_pos = self.data.cam_xpos[self.camera_id].copy()
        cam_mat = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()

        rotation_cv = np.eye(3, dtype=float)
        rotation_cv[0, :] = cam_mat[:, 0]
        rotation_cv[1, :] = -cam_mat[:, 1]
        rotation_cv[2, :] = -cam_mat[:, 2]
        extrinsic = np.eye(4, dtype=float)
        extrinsic[:3, :3] = rotation_cv
        extrinsic[:3, 3] = -rotation_cv @ cam_pos
        return extrinsic

    def render(self, *, apply_noise: bool = True, show_sites: bool = False) -> tuple[np.ndarray, np.ndarray]:
        scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_option)
        if not show_sites:
            scene_option.sitegroup[:] = 0

        self.renderer.update_scene(self.data, camera=self.camera_name, scene_option=scene_option)
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy().astype(np.float32)
        self.renderer.disable_depth_rendering()

        depth[depth > self.params.max_z_m] = 0.0
        depth[depth < 0.01] = 0.0
        if apply_noise:
            depth = self.apply_depth_noise(depth)
        return rgb, depth

    def apply_depth_noise(self, depth: np.ndarray) -> np.ndarray:
        noisy = depth.copy().astype(np.float32)
        valid = (depth > self.params.min_z_m) & (depth < self.params.max_z_m)
        if np.any(valid):
            rms = self.params.depth_rms_m(depth[valid])
            noisy[valid] += self.rng.normal(0.0, rms).astype(np.float32)

        grad_y, grad_x = np.gradient(depth)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        noisy[edge_strength > 0.05] = 0.0

        far_probability = np.clip(depth / self.params.max_z_m, 0.0, 1.0) ** 2 * 0.5
        far_dropout = self.rng.random(depth.shape) < far_probability
        noisy[far_dropout & valid] = 0.0
        noisy[noisy < self.params.min_z_m] = 0.0
        noisy[noisy > self.params.max_z_m] = 0.0
        return noisy.astype(np.float32)


def depth_stats(depth: np.ndarray, params: D435iParams | None = None) -> dict[str, float | int]:
    p = D435iParams() if params is None else params
    valid = (depth >= p.min_z_m) & (depth <= p.max_z_m)
    valid_depth = depth[valid]
    stats: dict[str, float | int] = {
        "total_pixels": int(depth.size),
        "valid_pixels": int(valid.sum()),
        "valid_ratio": float(valid.sum() / max(1, depth.size)),
    }
    if valid_depth.size:
        stats.update(
            {
                "min_depth_m": float(valid_depth.min()),
                "mean_depth_m": float(valid_depth.mean()),
                "max_depth_m": float(valid_depth.max()),
            }
        )
    return stats
