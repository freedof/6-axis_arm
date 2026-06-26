from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red", "红", "红色"),
    "blue": ("blue", "蓝", "蓝色"),
    "green": ("green", "绿", "绿色"),
    "yellow": ("yellow", "黄", "黄色"),
    "purple": ("purple", "紫", "紫色"),
}

SHAPE_ALIASES: dict[str, tuple[str, ...]] = {
    "box": ("box", "cube", "方块", "立方体", "小块", "块"),
    "cylinder": ("cylinder", "圆柱", "圆柱体"),
}

ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "pick": ("pick", "grasp", "抓取", "夹取", "拿起", "抓起", "夹起", "抓起来", "夹起来"),
    "place": ("place", "put", "放到", "放在", "放置", "放下"),
    "move": ("move", "移动到", "移到", "搬到", "拿到", "放到", "夹到"),
}

REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "tray": ("tray", "托盘", "盘子", "托盘中", "托盘里"),
    "left": ("left", "左侧", "左边", "左方"),
    "right": ("right", "右侧", "右边", "右方"),
    "front": ("front", "前方", "前面", "靠前"),
    "back": ("back", "后方", "后面", "靠后"),
    "center": ("center", "middle", "中心", "中间", "中央"),
}

TABLE_CENTER_XY_M = (0.35, -0.55)
TABLE_REGION_OFFSET_M = 0.11
TRAY_CENTER_XY_M = (0.64, -0.37)
QUANTIFIER_ALL_ALIASES = ("所有", "全部", "全部的", "all", "every")


@dataclass(frozen=True)
class ParsedToken:
    canonical: str
    text: str


def parse_language_goal(
    instruction: str,
    *,
    objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse a natural-language tabletop manipulation instruction."""
    if not instruction or not instruction.strip():
        raise ValueError("instruction must be a non-empty string.")

    normalized = _normalize_text(instruction)
    color = _find_first(normalized, COLOR_ALIASES)
    shape = _find_first(normalized, SHAPE_ALIASES)
    region = _find_first(normalized, REGION_ALIASES)
    quantifier = "all" if any(alias in normalized for alias in QUANTIFIER_ALL_ALIASES) else "one"
    action = _resolve_action(normalized, has_destination=region is not None)
    matched_objects = _match_objects(objects or [], color=color, shape=shape, instruction=normalized)
    target = _target_payload(color=color, shape=shape, matched_objects=matched_objects, quantifier=quantifier)
    destination = _destination_payload(region)
    ambiguities = _ambiguities(target, matched_objects)
    warnings = _warnings(target, matched_objects)

    status = "ok"
    if warnings:
        status = "no_match"
    elif ambiguities:
        status = "ambiguous"

    return {
        "status": status,
        "instruction": instruction,
        "action": action,
        "target": target,
        "destination": destination,
        "matched_objects": matched_objects,
        "ambiguities": ambiguities,
        "warnings": warnings,
        "vl_prompt": _build_vl_prompt(target),
    }


def _normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    return compact.replace("臺", "台")


def _find_first(text: str, aliases: dict[str, tuple[str, ...]]) -> ParsedToken | None:
    best: tuple[int, str, str] | None = None
    for canonical, words in aliases.items():
        for word in words:
            index = text.find(word.lower())
            if index < 0:
                continue
            if best is None or index < best[0] or (index == best[0] and len(word) > len(best[2])):
                best = (index, canonical, word)
    if best is None:
        return None
    return ParsedToken(canonical=best[1], text=best[2])


def _resolve_action(text: str, *, has_destination: bool) -> str:
    found = {canonical for canonical, words in ACTION_ALIASES.items() if any(word.lower() in text for word in words)}
    if has_destination and found:
        return "pick_and_place"
    if "pick" in found:
        return "pick"
    if "move" in found:
        return "pick_and_place" if has_destination else "move"
    if "place" in found:
        return "place"
    return "pick_and_place" if has_destination else "pick"


def _match_objects(
    objects: list[dict[str, Any]],
    *,
    color: ParsedToken | None,
    shape: ParsedToken | None,
    instruction: str,
) -> list[dict[str, Any]]:
    matches = []
    for item in objects:
        object_name = str(item.get("name", "")).lower()
        if object_name and object_name in instruction:
            matches.append(_candidate_payload(item))
            continue
        if color is not None and str(item.get("color", "")).lower() != color.canonical:
            continue
        if shape is not None and str(item.get("shape", "")).lower() != shape.canonical:
            continue
        if color is None and shape is None:
            continue
        matches.append(_candidate_payload(item))
    return matches


def _candidate_payload(item: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "name": item.get("name"),
        "shape": item.get("shape"),
        "color": item.get("color"),
        "initial_center_m": item.get("initial_center_m"),
        "position_xy_m": item.get("position_xy_m"),
    }
    return {key: value for key, value in keep.items() if value is not None}


def _target_payload(
    *,
    color: ParsedToken | None,
    shape: ParsedToken | None,
    matched_objects: list[dict[str, Any]],
    quantifier: str,
) -> dict[str, Any]:
    object_name = matched_objects[0]["name"] if len(matched_objects) == 1 else None
    return {
        "color": color.canonical if color else None,
        "color_text": color.text if color else None,
        "shape": shape.canonical if shape else None,
        "shape_text": shape.text if shape else None,
        "object_name": object_name,
        "object_names": [item["name"] for item in matched_objects],
        "quantifier": quantifier,
        "constraints": {
            "color": color.canonical if color else None,
            "shape": shape.canonical if shape else None,
        },
    }


def _destination_payload(region: ParsedToken | None) -> dict[str, Any] | None:
    if region is None:
        return None
    x, y = TABLE_CENTER_XY_M
    offset = TABLE_REGION_OFFSET_M
    xy_by_region = {
        "tray": list(TRAY_CENTER_XY_M),
        "left": [x, y - offset],
        "right": [x, y + offset],
        "front": [x + offset, y],
        "back": [x - offset, y],
        "center": [x, y],
    }
    return {
        "type": "tray" if region.canonical == "tray" else "table_region",
        "region": region.canonical,
        "region_text": region.text,
        "world_xy_m": [round(value, 6) for value in xy_by_region[region.canonical]],
    }


def _ambiguities(target: dict[str, Any], matched_objects: list[dict[str, Any]]) -> list[str]:
    messages = []
    if target["color"] is None and target["shape"] is None:
        messages.append("未解析到颜色或形状，目标约束不足。")
    if len(matched_objects) > 1 and target.get("quantifier") != "all":
        names = ", ".join(str(item["name"]) for item in matched_objects)
        messages.append(f"目标约束匹配到多个物体: {names}。")
    return messages


def _warnings(target: dict[str, Any], matched_objects: list[dict[str, Any]]) -> list[str]:
    if (target["color"] is not None or target["shape"] is not None) and not matched_objects:
        return ["目标约束没有匹配到当前场景中的已知物体。"]
    return []


def _build_vl_prompt(target: dict[str, Any]) -> str:
    color = target["color"] or "specified"
    shape = target["shape"] or "object"
    shape_text = "box/cube" if shape == "box" else shape
    if target.get("quantifier") == "all":
        return (
            f"Locate one visible {color} {shape_text} target object on the tabletop. "
            "Return one tight bbox around only one matching object; do not include the table or gripper."
        )
    return (
        f"Locate the {color} {shape_text} on the tabletop. "
        "Return one tight bbox around the visible target object only; do not include the table or gripper."
    )
