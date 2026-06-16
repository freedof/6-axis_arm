from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Region3D:
    center_world_m: np.ndarray
    center_camera_m: np.ndarray
    center_pixel: tuple[float, float]
    depth_m: float
    valid_pixel_count: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


def locate_red_region_fixture(
    rgb_path: str | Path,
    *,
    prompt: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Deterministic stand-in for a VL model during local integration tests."""
    rgb_path = Path(rgb_path)
    image = np.asarray(Image.open(rgb_path).convert("RGB"))
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)

    mask = (red > 100) & (red > green * 1.35) & (red > blue * 1.35)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise RuntimeError(f"Fixture VL provider could not find a red target in {rgb_path}")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    confidence = float(np.clip(mask.mean() * 18.0, 0.05, 0.99))

    overlay_path = None
    if output_path is not None:
        overlay_path = Path(output_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 230, 30), width=3)
        draw.text((x1, max(0, y1 - 14)), "VL region", fill=(255, 230, 30))
        overlay.save(overlay_path)

    return {
        "type": "bbox",
        "label": "red_object",
        "prompt": prompt,
        "provider": "color_fixture",
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": round(confidence, 4),
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
    }


def estimate_region_3d(
    *,
    depth_path: str | Path,
    intrinsics: list[list[float]] | np.ndarray,
    extrinsic_world_to_camera: list[list[float]] | np.ndarray,
    region: dict[str, Any],
    min_depth_m: float = 0.17,
    max_depth_m: float = 10.0,
    foreground_quantile: float = 0.55,
    foreground_margin_m: float = 0.015,
) -> Region3D:
    depth = np.load(Path(depth_path)).astype(np.float32)
    k = np.asarray(intrinsics, dtype=float)
    world_to_camera = np.asarray(extrinsic_world_to_camera, dtype=float)
    camera_to_world = np.linalg.inv(world_to_camera)

    bbox = _bbox_from_region(region, width=depth.shape[1], height=depth.shape[0])
    x1, y1, x2, y2 = bbox
    crop = depth[y1:y2, x1:x2]
    valid = (crop >= min_depth_m) & (crop <= max_depth_m)
    values = crop[valid]
    if values.size == 0:
        raise RuntimeError(f"No valid depth pixels inside region bbox={bbox}")

    cutoff = float(np.quantile(values, foreground_quantile) + foreground_margin_m)
    foreground = valid & (crop <= cutoff)
    fy, fx = np.nonzero(foreground)
    if fx.size == 0:
        fy, fx = np.nonzero(valid)

    pixels_x = fx.astype(float) + x1
    pixels_y = fy.astype(float) + y1
    depths = crop[fy, fx].astype(float)
    points_camera = _backproject(pixels_x, pixels_y, depths, k)
    points_world = _transform_points(camera_to_world, points_camera)

    center_camera = np.median(points_camera, axis=0)
    center_world = np.median(points_world, axis=0)
    center_pixel = (float(np.median(pixels_x)), float(np.median(pixels_y)))
    depth_m = float(np.median(depths))

    return Region3D(
        center_world_m=center_world,
        center_camera_m=center_camera,
        center_pixel=center_pixel,
        depth_m=depth_m,
        valid_pixel_count=int(depths.size),
        bbox_xyxy=bbox,
        confidence=float(region.get("confidence", 0.0)),
    )


def region3d_to_dict(region: Region3D) -> dict[str, Any]:
    return {
        "center_world_m": _round_vector(region.center_world_m),
        "center_camera_m": _round_vector(region.center_camera_m),
        "center_pixel": [round(float(region.center_pixel[0]), 3), round(float(region.center_pixel[1]), 3)],
        "depth_m": round(float(region.depth_m), 6),
        "valid_pixel_count": region.valid_pixel_count,
        "bbox_xyxy": list(region.bbox_xyxy),
        "confidence": round(float(region.confidence), 4),
        "interpretation": "VL region lifted to 3D with D435i depth; suitable as a grasp-target estimate, not final grasp pose.",
    }


def _bbox_from_region(region: dict[str, Any], *, width: int, height: int) -> tuple[int, int, int, int]:
    if region.get("type", "bbox") == "point":
        x = int(round(float(region["x"])))
        y = int(round(float(region["y"])))
        radius = int(region.get("radius_px", 12))
        return _clip_bbox((x - radius, y - radius, x + radius + 1, y + radius + 1), width, height)
    if "bbox_xyxy" not in region:
        raise ValueError("Region must include bbox_xyxy, or type=point with x/y.")
    return _clip_bbox(tuple(int(round(float(v))) for v in region["bbox_xyxy"]), width, height)


def _clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _backproject(xs: np.ndarray, ys: np.ndarray, depths: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (xs - cx) * depths / fx
    y = (ys - cy) * depths / fy
    return np.column_stack((x, y, depths))


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    return (transform @ homogeneous.T).T[:, :3]


def _round_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]

