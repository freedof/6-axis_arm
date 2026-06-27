from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVIDER_CONFIG_PATH = ROOT / "config" / "vl_providers.local.json"


@dataclass(frozen=True)
class Region3D:
    center_world_m: np.ndarray
    center_camera_m: np.ndarray
    center_pixel: tuple[float, float]
    depth_m: float
    valid_pixel_count: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


VL_REGION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["bbox"]},
        "label": {"type": "string"},
        "bbox_xyxy": {
            "type": "array",
            "items": {"type": "number"},
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["type", "label", "bbox_xyxy", "confidence", "reasoning"],
    "additionalProperties": False,
}


def locate_red_region_fixture(
    rgb_path: str | Path,
    *,
    prompt: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Deterministic stand-in for a VL model during local integration tests.

    The original fixture only looked for the red cube. It now infers a requested
    color from the prompt so multi-object tabletop tests can validate the full
    engineering chain without depending on a remote VL model.
    """
    rgb_path = Path(rgb_path)
    image = np.asarray(Image.open(rgb_path).convert("RGB"))
    color = _fixture_color_from_prompt(prompt)
    mask = _fixture_color_mask(image, color)
    bbox = _largest_mask_component_bbox(mask)
    if bbox is None:
        raise RuntimeError(f"Fixture VL provider could not find a {color} target in {rgb_path}")

    x1, y1, x2, y2 = bbox
    confidence = float(np.clip(mask.mean() * 18.0, 0.05, 0.99))

    overlay_path = None
    if output_path is not None:
        overlay_path = Path(output_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 230, 30), width=3)
        draw.text((x1, max(0, y1 - 14)), f"VL {color}", fill=(255, 230, 30))
        overlay.save(overlay_path)

    return {
        "type": "bbox",
        "label": f"{color}_object",
        "prompt": prompt,
        "provider": "color_fixture",
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": round(confidence, 4),
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
    }


def _fixture_color_from_prompt(prompt: str) -> str:
    text = prompt.lower()
    aliases = {
        "red": ("red", "红", "红色"),
        "blue": ("blue", "蓝", "蓝色"),
        "green": ("green", "绿", "绿色"),
        "yellow": ("yellow", "黄", "黄色"),
        "purple": ("purple", "紫", "紫色"),
    }
    for color, words in aliases.items():
        if any(word in text for word in words):
            return color
    return "red"



def _largest_mask_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: tuple[int, int, int, int, int] | None = None
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        count = 0
        min_x = max_x = start_x
        min_y = max_y = start_y
        while stack:
            y, x = stack.pop()
            count += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if count < 4:
            continue
        if best is None or count > best[0]:
            best = (count, min_x, min_y, max_x + 1, max_y + 1)
    if best is None:
        return None
    _, x1, y1, x2, y2 = best
    return int(x1), int(y1), int(x2), int(y2)

def _fixture_color_mask(image: np.ndarray, color: str) -> np.ndarray:
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    if color == "red":
        return (red > 100) & (red > green * 1.35) & (red > blue * 1.35)
    if color == "blue":
        return (blue > 90) & (blue > red * 1.25) & (blue > green * 1.10)
    if color == "green":
        return (green > 85) & (green > red * 1.25) & (green > blue * 1.15)
    if color == "yellow":
        return (red > 120) & (green > 95) & (blue < 120) & (np.abs(red - green) < 95)
    if color == "purple":
        return (red > 80) & (blue > 100) & (green < 125) & (blue > green * 1.10)
    raise ValueError(f"Unsupported fixture color: {color}")

def locate_manual_region(
    region: dict[str, Any],
    *,
    prompt: str,
    rgb_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a caller-provided region in the same schema used by VL providers."""
    normalized = _normalize_region(region, prompt=prompt, provider="manual_region")
    if rgb_path is not None and output_path is not None:
        overlay_path = _draw_region_overlay(rgb_path, normalized, output_path, label="manual region")
        normalized["overlay_path"] = str(overlay_path)
    return normalized


def locate_codex_vision_region(
    region: dict[str, Any],
    *,
    prompt: str,
    rgb_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a Codex-inspected region for interactive VL validation.

    The repository cannot call the current Codex conversation as a runtime API.
    Instead, Codex inspects the rendered RGB image in the chat, supplies the
    bbox/point, and this provider normalizes it into the same schema as the
    autonomous VL providers.
    """
    normalized = _normalize_region(region, prompt=prompt, provider="codex_vision")
    normalized["provider_note"] = "Codex-in-the-loop visual region supplied from the current Codex session."
    if rgb_path is not None and output_path is not None:
        overlay_path = _draw_region_overlay(rgb_path, normalized, output_path, label="Codex VL")
        normalized["overlay_path"] = str(overlay_path)
    return normalized


def locate_openai_vision_region(
    rgb_path: str | Path,
    *,
    prompt: str,
    output_path: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    config_path: str | Path | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Call an OpenAI vision model and return a normalized bbox region."""
    rgb_path = Path(rgb_path)
    config, config_file = _provider_config("openai_vision", config_path)
    key = api_key or str(config.get("api_key", ""))
    if not key:
        raise RuntimeError(f"api_key is required for provider='openai_vision' in {config_file}.")

    selected_model = model or str(config.get("model", "gpt-5.5"))
    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    image_url = _image_data_url(rgb_path)
    instructions = _robot_vl_localization_instructions()
    user_text = (
        f"Image size: width={width}, height={height}. "
        f"Target request: {prompt}. "
        "Return only the structured region."
    )
    payload = {
        "model": selected_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instructions + "\n" + user_text},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "vl_region",
                "strict": True,
                "schema": VL_REGION_SCHEMA,
            }
        },
    }
    response = _post_openai_response(payload, api_key=key, timeout_s=timeout_s)
    raw_region = json.loads(_extract_response_text(response))
    normalized = _normalize_region(raw_region, prompt=prompt, provider="openai_vision", image_size=(width, height))
    normalized["model"] = selected_model
    normalized["config_path"] = str(config_file)
    if output_path is not None:
        overlay_path = _draw_region_overlay(rgb_path, normalized, output_path, label="OpenAI VL")
        normalized["overlay_path"] = str(overlay_path)
    return normalized


def locate_ark_coding_vision_region(
    rgb_path: str | Path,
    *,
    prompt: str,
    output_path: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    config_path: str | Path | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Call an OpenAI-compatible Ark coding endpoint and return a bbox region."""
    rgb_path = Path(rgb_path)
    config, config_file = _provider_config("ark_coding_vision", config_path)
    key = api_key or str(config.get("api_key", ""))
    if not key:
        raise RuntimeError(f"api_key is required for provider='ark_coding_vision' in {config_file}.")

    selected_model = model or str(config.get("model", "glm-5.2"))
    endpoint = _chat_completions_endpoint(
        base_url or str(config.get("base_url", "https://ark.cn-beijing.volces.com/api/coding/v3"))
    )
    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    image_url = _image_data_url(rgb_path)
    instructions = _robot_vl_localization_instructions(json_only=True)
    user_text = (
        f"Image size: width={width}, height={height}. "
        f"Target request: {prompt}. "
        "Return strict JSON only."
    )
    raw_region = _locate_ark_region_with_retry(
        endpoint=endpoint,
        api_key=key,
        model=selected_model,
        image_url=image_url,
        instructions=instructions,
        user_text=user_text,
        timeout_s=timeout_s,
        provider_name="Ark coding vision",
    )
    normalized = _normalize_region(raw_region, prompt=prompt, provider="ark_coding_vision", image_size=(width, height))
    normalized["model"] = selected_model
    normalized["base_url"] = endpoint.rsplit("/chat/completions", 1)[0]
    normalized["config_path"] = str(config_file)
    normalized = _review_chat_completion_region(
        endpoint=endpoint,
        api_key=key,
        model=selected_model,
        image_url=image_url,
        prompt=prompt,
        image_size=(width, height),
        region=normalized,
        timeout_s=timeout_s,
        provider="ark_coding_vision",
        provider_name="Ark coding vision",
    )
    if output_path is not None:
        overlay_path = _draw_region_overlay(rgb_path, normalized, output_path, label="Ark VL")
        normalized["overlay_path"] = str(overlay_path)
    return normalized


def locate_openrouter_vision_region(
    rgb_path: str | Path,
    *,
    prompt: str,
    output_path: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    config_path: str | Path | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Call an OpenRouter OpenAI-compatible endpoint and return a bbox region."""
    rgb_path = Path(rgb_path)
    config, config_file = _provider_config("openrouter_vision", config_path)
    key = api_key or str(config.get("api_key", ""))
    if not key:
        raise RuntimeError(f"api_key is required for provider='openrouter_vision' in {config_file}.")

    selected_model = model or str(config.get("model", "google/gemini-3.5-flash"))
    endpoint = _chat_completions_endpoint(base_url or str(config.get("base_url", "https://openrouter.ai/api/v1")))
    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    image_url = _image_data_url(rgb_path)
    instructions = _robot_vl_localization_instructions(json_only=True)
    user_text = (
        f"Image size: width={width}, height={height}. "
        f"Target request: {prompt}. "
        "Return strict JSON only."
    )
    raw_region = _locate_ark_region_with_retry(
        endpoint=endpoint,
        api_key=key,
        model=selected_model,
        image_url=image_url,
        instructions=instructions,
        user_text=user_text,
        timeout_s=timeout_s,
        provider_name="OpenRouter vision",
    )
    normalized = _normalize_region(raw_region, prompt=prompt, provider="openrouter_vision", image_size=(width, height))
    normalized["model"] = selected_model
    normalized["base_url"] = endpoint.rsplit("/chat/completions", 1)[0]
    normalized["config_path"] = str(config_file)
    normalized = _review_chat_completion_region(
        endpoint=endpoint,
        api_key=key,
        model=selected_model,
        image_url=image_url,
        prompt=prompt,
        image_size=(width, height),
        region=normalized,
        timeout_s=timeout_s,
        provider="openrouter_vision",
        provider_name="OpenRouter vision",
    )
    if output_path is not None:
        overlay_path = _draw_region_overlay(rgb_path, normalized, output_path, label="OpenRouter VL")
        normalized["overlay_path"] = str(overlay_path)
    return normalized


def locate_openrouter_vision_regions(
    rgb_path: str | Path,
    *,
    targets: list[dict[str, Any]],
    output_path: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    config_path: str | Path | None = None,
    timeout_s: float = 60.0,
) -> dict[str, dict[str, Any]]:
    """Call OpenRouter once and return one bbox region per requested target."""
    rgb_path = Path(rgb_path)
    config, config_file = _provider_config("openrouter_vision", config_path)
    key = api_key or str(config.get("api_key", ""))
    if not key:
        raise RuntimeError(f"api_key is required for provider='openrouter_vision' in {config_file}.")

    selected_model = model or str(config.get("model", "google/gemini-3.5-flash"))
    endpoint = _chat_completions_endpoint(base_url or str(config.get("base_url", "https://openrouter.ai/api/v1")))
    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    image_url = _image_data_url(rgb_path)
    target_lines = []
    for target in targets:
        target_lines.append(
            "- "
            f"name={target['name']}, color={target.get('color', 'unknown')}, "
            f"shape={target.get('shape', 'object')}"
        )
    instructions = (
        "You are a visual grounding model for a robot manipulation system. "
        "The image is captured by a wrist-mounted RGB-D camera looking at a tabletop scene. "
        "Locate every requested tabletop target in one pass. "
        "Return strict JSON only with key objects. "
        "objects must be an array; each item must have keys: name, type, label, bbox_xyxy, confidence, reasoning. "
        "Use type='bbox'. Use pixel coordinates in the original image with bbox_xyxy=[x1,y1,x2,y2]. "
        "Box the full visible outer body of each physical object, including gray-lit top faces attached to colored side faces. "
        "Do not return tiny color patches, edges, shadows, table, tray, gripper, camera mount, or empty background. "
        "If a target is not visible, omit it from objects rather than guessing."
    )
    user_text = (
        f"Image size: width={width}, height={height}. "
        "Requested targets:\n"
        + "\n".join(target_lines)
    )
    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions + "\n" + user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    response = _post_json(endpoint, payload, api_key=key, timeout_s=timeout_s, provider_name="OpenRouter multi-target vision")
    raw = json.loads(_extract_chat_completion_text(response))
    objects = _multi_target_objects(raw)
    by_name: dict[str, dict[str, Any]] = {}
    target_names = {str(target["name"]) for target in targets}
    for item in objects:
        name = str(item.get("name") or item.get("label") or "")
        if name not in target_names:
            continue
        region = dict(item)
        region.setdefault("label", name)
        normalized = _normalize_region(region, prompt=f"Locate {name}", provider="openrouter_vision", image_size=(width, height))
        normalized["name"] = name
        normalized["model"] = selected_model
        normalized["base_url"] = endpoint.rsplit("/chat/completions", 1)[0]
        normalized["config_path"] = str(config_file)
        by_name[name] = normalized
    if output_path is not None:
        overlay_path = _draw_regions_overlay(rgb_path, by_name, output_path)
        for region in by_name.values():
            region["overlay_path"] = str(overlay_path)
    return by_name


def locate_openrouter_vision_category_regions(
    rgb_path: str | Path,
    *,
    category: str,
    target_query: str | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    config_path: str | Path | None = None,
    timeout_s: float = 60.0,
    max_objects: int = 8,
) -> dict[str, dict[str, Any]]:
    """Call OpenRouter once and return all visible tabletop objects matching a query."""
    rgb_path = Path(rgb_path)
    config, config_file = _provider_config("openrouter_vision", config_path)
    key = api_key or str(config.get("api_key", ""))
    if not key:
        raise RuntimeError(f"api_key is required for provider='openrouter_vision' in {config_file}.")

    selected_model = model or str(config.get("model", "google/gemini-3.5-flash"))
    endpoint = _chat_completions_endpoint(base_url or str(config.get("base_url", "https://openrouter.ai/api/v1")))
    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    image_url = _image_data_url(rgb_path)
    query = str(target_query or category)
    instructions = (
        "You are a visual grounding model for a robot manipulation system. "
        "The image is captured by a wrist-mounted RGB-D camera looking at a tabletop scene. "
        f"Find every visible tabletop object matching this target query: {query}. "
        "The target query may include color, shape, count, spatial relation, inclusion, or exclusion constraints. "
        "Return strict JSON only with key objects. "
        "objects must be an array; each item must have keys: label, type, bbox_xyxy, confidence, reasoning. "
        "Use type='bbox'. Use pixel coordinates in the original image with bbox_xyxy=[x1,y1,x2,y2]. "
        "Box the full visible outer body of each physical object, including gray-lit top faces attached to colored side faces. "
        "Do not include cylinders, tray, gripper, camera mount, table, shadows, empty background, or non-tabletop objects. "
        "Do not invent hidden objects. If none are visible, return objects=[]."
    )
    user_text = f"Image size: width={width}, height={height}. Maximum objects: {int(max_objects)}."
    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions + "\n" + user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    response = _post_json(endpoint, payload, api_key=key, timeout_s=timeout_s, provider_name="OpenRouter category vision")
    raw = json.loads(_extract_chat_completion_text(response))
    objects = _multi_target_objects(raw)[: max(0, int(max_objects))]

    normalized_regions: list[dict[str, Any]] = []
    for index, item in enumerate(objects):
        region = dict(item)
        region.setdefault("label", f"candidate_{index + 1}")
        normalized = _normalize_region(region, prompt=f"Locate visible targets matching: {query}", provider="openrouter_vision", image_size=(width, height))
        normalized["raw_label"] = str(item.get("label") or item.get("name") or normalized["label"])
        normalized["model"] = selected_model
        normalized["base_url"] = endpoint.rsplit("/chat/completions", 1)[0]
        normalized["config_path"] = str(config_file)
        normalized_regions.append(normalized)

    normalized_regions.sort(key=lambda item: (_bbox_center(item["bbox_xyxy"])[1], _bbox_center(item["bbox_xyxy"])[0]))
    by_name: dict[str, dict[str, Any]] = {}
    for index, region in enumerate(normalized_regions, start=1):
        name = f"candidate_{index}"
        region["name"] = name
        region["label"] = name
        by_name[name] = region
    if output_path is not None:
        overlay_path = _draw_regions_overlay(rgb_path, by_name, output_path)
        for region in by_name.values():
            region["overlay_path"] = str(overlay_path)
    return by_name


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
    bbox_expansion: float = 1.25,
    min_world_z_m: float | None = None,
) -> Region3D:
    depth = np.load(Path(depth_path)).astype(np.float32)
    k = np.asarray(intrinsics, dtype=float)
    world_to_camera = np.asarray(extrinsic_world_to_camera, dtype=float)
    camera_to_world = np.linalg.inv(world_to_camera)

    source_bbox = _bbox_from_region(region, width=depth.shape[1], height=depth.shape[0])
    bbox = _expand_bbox(source_bbox, width=depth.shape[1], height=depth.shape[0], scale=bbox_expansion)
    x1, y1, x2, y2 = bbox
    crop = depth[y1:y2, x1:x2]
    valid = (crop >= min_depth_m) & (crop <= max_depth_m)
    valid_y, valid_x = np.nonzero(valid)
    if valid_x.size == 0:
        raise RuntimeError(f"No valid depth pixels inside region bbox={bbox}")

    valid_pixels_x = valid_x.astype(float) + x1
    valid_pixels_y = valid_y.astype(float) + y1
    valid_depths = crop[valid_y, valid_x].astype(float)
    valid_points_camera = _backproject(valid_pixels_x, valid_pixels_y, valid_depths, k)
    valid_points_world = _transform_points(camera_to_world, valid_points_camera)

    object_mask = np.zeros_like(valid, dtype=bool)
    if min_world_z_m is None:
        object_mask[valid_y, valid_x] = True
    else:
        above_table = valid_points_world[:, 2] >= float(min_world_z_m)
        if not np.any(above_table):
            raise RuntimeError(f"No valid depth pixels above min_world_z_m={min_world_z_m:.6f} inside region bbox={bbox}")
        object_mask[valid_y[above_table], valid_x[above_table]] = True

    values = crop[object_mask]
    cutoff = float(np.quantile(values, foreground_quantile) + foreground_margin_m)
    foreground = object_mask & (crop <= cutoff)
    if not np.any(foreground):
        foreground = object_mask
    foreground = _largest_connected_mask(foreground)
    fy, fx = np.nonzero(foreground)

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


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    if scale <= 1.0:
        return _clip_bbox(bbox, width, height)
    x1, y1, x2, y2 = bbox
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    half_w = 0.5 * (x2 - x1) * float(scale)
    half_h = 0.5 * (y2 - y1) * float(scale)
    return _clip_bbox(
        (
            int(np.floor(center_x - half_w)),
            int(np.floor(center_y - half_h)),
            int(np.ceil(center_x + half_w)),
            int(np.ceil(center_y + half_h)),
        ),
        width,
        height,
    )


def _clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _bbox_center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _largest_connected_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return mask
    visited = np.zeros_like(mask, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > len(best_pixels):
            best_pixels = pixels
    component = np.zeros_like(mask, dtype=bool)
    for y, x in best_pixels:
        component[y, x] = True
    return component


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


def _normalize_region(
    region: dict[str, Any],
    *,
    prompt: str,
    provider: str,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if region.get("type", "bbox") == "point":
        x = float(region["x"])
        y = float(region["y"])
        radius = float(region.get("radius_px", 12))
        region = {
            "type": "bbox",
            "label": str(region.get("label", "target_object")),
            "bbox_xyxy": [x - radius, y - radius, x + radius + 1, y + radius + 1],
            "confidence": float(region.get("confidence", 1.0)),
            "reasoning": str(region.get("reasoning", "manual point converted to bbox")),
        }

    if region.get("type", "bbox") != "bbox":
        raise ValueError("Only bbox and point regions are currently supported.")
    if "bbox_xyxy" not in region:
        raise ValueError("Region must include bbox_xyxy.")

    bbox_values = [float(value) for value in region["bbox_xyxy"]]
    if len(bbox_values) != 4:
        raise ValueError(f"bbox_xyxy must contain four values, got {bbox_values}")
    if image_size is None:
        bbox = [int(round(value)) for value in bbox_values]
    else:
        bbox = list(_clip_bbox(tuple(int(round(value)) for value in bbox_values), image_size[0], image_size[1]))

    confidence = float(region.get("confidence", 1.0))
    return {
        "type": "bbox",
        "label": str(region.get("label", "target_object")),
        "prompt": prompt,
        "provider": provider,
        "bbox_xyxy": bbox,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "reasoning": str(region.get("reasoning", "")),
        "overlay_path": region.get("overlay_path"),
    }


def _draw_region_overlay(
    rgb_path: str | Path,
    region: dict[str, Any],
    output_path: str | Path,
    *,
    label: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay = Image.open(rgb_path).convert("RGB")
    x1, y1, x2, y2 = _bbox_from_region(region, width=overlay.width, height=overlay.height)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 230, 30), width=3)
    draw.text((x1, max(0, y1 - 14)), label, fill=(255, 230, 30))
    overlay.save(output)
    return output


def _draw_regions_overlay(
    rgb_path: str | Path,
    regions_by_name: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    colors = ((255, 80, 60), (60, 150, 255), (190, 90, 255), (255, 220, 40))
    for index, (name, region) in enumerate(regions_by_name.items()):
        x1, y1, x2, y2 = _bbox_from_region(region, width=overlay.width, height=overlay.height)
        color = colors[index % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1 - 14)), name, fill=color)
    overlay.save(output)
    return output


def _multi_target_objects(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        objects = raw.get("objects")
        if isinstance(objects, list):
            return [item for item in objects if isinstance(item, dict)]
        result = []
        for name, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", name)
                result.append(item)
        return result
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _provider_config(provider: str, config_path: str | Path | None) -> tuple[dict[str, Any], Path]:
    path = Path(config_path) if config_path is not None else DEFAULT_PROVIDER_CONFIG_PATH
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise RuntimeError(
            f"VL provider config file not found: {path}. "
            "Copy config/vl_providers.example.json to config/vl_providers.local.json and fill in your API key."
        )

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    providers = data.get("providers")
    if not isinstance(providers, dict):
        raise RuntimeError(f"VL provider config must contain an object field named 'providers': {path}")

    config = providers.get(provider)
    if not isinstance(config, dict):
        raise RuntimeError(f"VL provider '{provider}' is not configured in {path}")
    return config, path


def _robot_vl_localization_instructions(*, json_only: bool = False) -> str:
    output_rule = (
        "Return only a JSON object with keys: type, label, bbox_xyxy, confidence, reasoning. "
        if json_only
        else "Return one structured region object. "
    )
    return (
        "You are a visual grounding model for a robot manipulation system. "
        "The image is captured by a wrist-mounted Intel RealSense D435i-style RGB-D camera on a robot gripper. "
        "The camera is looking from above or obliquely from above at a tabletop scene. "
        "Your job is to locate the single physical tabletop object that best matches the user's target request. "
        f"{output_rule}"
        "Use type='bbox'. Use pixel coordinates in the original image with bbox_xyxy=[x1,y1,x2,y2]. "
        "The target may be small, often only a few tens of pixels wide. "
        "Return a tight box around only the visible body of the target object. "
        "If the object is partially occluded, box only the visible target surface. "
        "Do not include the robot gripper, gray fingers, camera mount, table, floor, background, shadows, highlights, or empty image borders. "
        "Do not return a 1-pixel corner or image-edge box unless the requested object itself visibly touches that edge. "
        "Prefer a smaller precise bbox over a large bbox that includes table/background. "
        "If multiple similar objects are visible, choose the one matching the user's color, spatial relation, or task wording, and explain that choice in reasoning."
    )


def _review_chat_completion_region(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    image_url: str,
    prompt: str,
    image_size: tuple[int, int],
    region: dict[str, Any],
    timeout_s: float,
    provider: str,
    provider_name: str,
) -> dict[str, Any]:
    """Ask the VL provider to verify or correct its own bbox."""
    width, height = image_size
    initial_bbox = list(region["bbox_xyxy"])
    instructions = (
        "You are reviewing a candidate bbox for robot visual grounding. "
        "Inspect the image and decide whether the candidate bbox tightly covers the visible body of the requested target object. "
        "The target may be small. The bbox must not cover mostly table, floor, background, gripper, shadows, highlights, or empty borders. "
        "If the candidate is wrong but the requested target is visible, provide a corrected tight bbox. "
        "If the requested target is not visible, return valid_bbox=false and corrected_bbox_xyxy=[]. "
        "Return strict JSON only with keys: valid_bbox, corrected_bbox_xyxy, confidence, reasoning."
    )
    user_text = (
        f"Image size: width={width}, height={height}. "
        f"Target request: {prompt}. "
        f"Candidate bbox_xyxy: {initial_bbox}. "
        "Review the candidate bbox."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions + "\n" + user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    response = _post_json(endpoint, payload, api_key=api_key, timeout_s=timeout_s, provider_name=f"{provider_name} self-check")
    review = json.loads(_extract_chat_completion_text(response))
    valid = bool(review.get("valid_bbox", False))
    corrected = review.get("corrected_bbox_xyxy", [])
    if isinstance(corrected, list) and len(corrected) == 4:
        reviewed = {
            "type": "bbox",
            "label": region.get("label", "target_object"),
            "bbox_xyxy": corrected,
            "confidence": float(review.get("confidence", region.get("confidence", 0.5))),
            "reasoning": str(review.get("reasoning", "")),
        }
        normalized = _normalize_region(reviewed, prompt=prompt, provider=provider, image_size=image_size)
        normalized["model"] = region.get("model", model)
        normalized["base_url"] = region.get("base_url")
        normalized["config_path"] = region.get("config_path")
        normalized["self_check"] = {
            "initial_bbox_xyxy": initial_bbox,
            "valid_bbox": valid,
            "corrected": corrected != initial_bbox,
            "confidence": normalized["confidence"],
            "reasoning": str(review.get("reasoning", "")),
        }
        return normalized
    review_confidence = float(review.get("confidence", region.get("confidence", 0.5)))
    region["self_check"] = {
        "initial_bbox_xyxy": initial_bbox,
        "valid_bbox": valid,
        "corrected": False,
        "confidence": review_confidence,
        "reasoning": str(review.get("reasoning", "")),
    }
    if valid:
        region["confidence"] = round(float(np.clip(review_confidence, 0.0, 1.0)), 4)
    return region


def _locate_ark_region_with_retry(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    image_url: str,
    instructions: str,
    user_text: str,
    timeout_s: float,
    provider_name: str,
    attempts: int = 2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        retry_text = user_text
        if attempt > 1:
            retry_text += (
                " Retry carefully: the target is a small physical tabletop object, not the table/background. "
                "Return your best tight bbox around the visible target body."
            )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions + "\n" + retry_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0,
        }
        try:
            response = _post_json(endpoint, payload, api_key=api_key, timeout_s=timeout_s, provider_name=provider_name)
            return json.loads(_extract_chat_completion_text(response))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{provider_name} failed after {attempts} attempts: {last_error}") from last_error


def _post_openai_response(payload: dict[str, Any], *, api_key: str, timeout_s: float) -> dict[str, Any]:
    return _post_json(
        "https://api.openai.com/v1/responses",
        payload,
        api_key=api_key,
        timeout_s=timeout_s,
        provider_name="OpenAI vision",
    )


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_s: float,
    provider_name: str,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider_name} request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        try:
            return _post_json_with_curl(url, payload, api_key=api_key, timeout_s=timeout_s, provider_name=provider_name)
        except RuntimeError as curl_exc:
            raise RuntimeError(f"{provider_name} request failed: {exc}; curl fallback failed: {curl_exc}") from exc


def _post_json_with_curl(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_s: float,
    provider_name: str,
) -> dict[str, Any]:
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    if curl_path is None:
        raise RuntimeError("curl executable was not found")
    with tempfile.TemporaryDirectory(prefix="vl_curl_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        payload_path = tmp_path / "payload.json"
        config_path = tmp_path / "curl.conf"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        config_path.write_text(
            "\n".join(
                [
                    "silent",
                    "show-error",
                    "fail-with-body",
                    f"max-time = {max(1, int(round(timeout_s)))}",
                    f"url = {json.dumps(url)}",
                    "request = POST",
                    f"header = {json.dumps(f'Authorization: Bearer {api_key}')}",
                    "header = \"Content-Type: application/json\"",
                ]
            ),
            encoding="utf-8",
        )
        attempts = (
            [curl_path, "--http1.1", "--config", str(config_path), "--data-binary", f"@{payload_path}"],
            [curl_path, "--ssl-no-revoke", "--http1.1", "--config", str(config_path), "--data-binary", f"@{payload_path}"],
        )
        errors = []
        result = None
        for command in attempts:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s + 5,
                check=False,
            )
            if result.returncode == 0:
                break
            errors.append((result.stderr or result.stdout or "").strip())
    if result is None or result.returncode != 0:
        message = " | ".join(error for error in errors if error)
        raise RuntimeError(f"{provider_name} curl request failed: {message}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{provider_name} curl response was not JSON: {result.stdout[:500]}") from exc


def _extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise RuntimeError(f"Could not extract text output from OpenAI response: {response}")


def _extract_chat_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Could not extract choices from chat completion response: {response}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return _extract_json_object_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return _extract_json_object_text("\n".join(parts))
    raise RuntimeError(f"Could not extract message content from chat completion response: {response}")


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"Expected a JSON object in model output, got: {text}")
    return stripped[start : end + 1]


def _chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _round_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]



