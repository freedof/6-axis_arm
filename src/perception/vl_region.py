from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
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
    instructions = (
        "You are locating a manipulation target in a robot gripper camera image. "
        "Return one tight bounding box around the single object that best matches the user's request. "
        "Use pixel coordinates in the original image with bbox_xyxy = [x1, y1, x2, y2]. "
        "If the object is partially occluded, bound the visible target surface. "
        "Do not include robot fingers, table, shadows, or debug overlays unless the user explicitly asks for them."
    )
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
    instructions = (
        "You are locating a manipulation target in a robot gripper camera image. "
        "Return only a JSON object with keys: type, label, bbox_xyxy, confidence, reasoning. "
        "Use type='bbox'. Use pixel coordinates in the original image with bbox_xyxy=[x1,y1,x2,y2]. "
        "Return one tight bounding box around the single object that best matches the user's request. "
        "Do not include robot fingers, table, shadows, or debug overlays unless the user explicitly asks for them."
    )
    user_text = (
        f"Image size: width={width}, height={height}. "
        f"Target request: {prompt}. "
        "Return strict JSON only."
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
    response = _post_json(endpoint, payload, api_key=key, timeout_s=timeout_s, provider_name="Ark coding vision")
    raw_region = json.loads(_extract_chat_completion_text(response))
    normalized = _normalize_region(raw_region, prompt=prompt, provider="ark_coding_vision", image_size=(width, height))
    normalized["model"] = selected_model
    normalized["base_url"] = endpoint.rsplit("/chat/completions", 1)[0]
    normalized["config_path"] = str(config_file)
    if output_path is not None:
        overlay_path = _draw_region_overlay(rgb_path, normalized, output_path, label="Ark VL")
        normalized["overlay_path"] = str(overlay_path)
    return normalized


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
        raise RuntimeError(f"{provider_name} request failed: {exc}") from exc


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
