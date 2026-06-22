from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp_robot import skills


def main() -> None:
    cases = [
        {
            "instruction": "夹取蓝色方块",
            "status": "ok",
            "action": "pick",
            "object_name": "blue_cube",
            "color": "blue",
            "shape": "box",
            "destination": None,
        },
        {
            "instruction": "把绿色圆柱体放到桌面左侧",
            "status": "ok",
            "action": "pick_and_place",
            "object_name": "green_cylinder",
            "color": "green",
            "shape": "cylinder",
            "destination_region": "left",
        },
        {
            "instruction": "抓起红色小块",
            "status": "ok",
            "action": "pick",
            "object_name": "red_cube",
            "color": "red",
            "shape": "box",
            "destination": None,
        },
        {
            "instruction": "将黄色圆柱体移动到右侧",
            "status": "ok",
            "action": "pick_and_place",
            "object_name": "yellow_cylinder",
            "color": "yellow",
            "shape": "cylinder",
            "destination_region": "right",
        },
        {
            "instruction": "夹取方块",
            "status": "ambiguous",
            "action": "pick",
            "object_name": None,
            "color": None,
            "shape": "box",
            "matched_count": 3,
            "destination": None,
        },
    ]

    for case in cases:
        result = skills.parse_language_goal(case["instruction"])
        _assert_case(result, case)
        print(f"case: {case['instruction']}")
        print("  status:", result["status"])
        print("  action:", result["action"])
        print("  target:", result["target"])
        print("  destination:", result["destination"])
        print("  matched:", [item["name"] for item in result["matched_objects"]])

    mcp_like = skills.parse_language_goal(
        "pick the purple cube and place it at the center",
    )
    if mcp_like["target"]["object_name"] != "purple_cube":
        raise RuntimeError(f"English instruction should match purple_cube: {mcp_like}")
    if mcp_like["destination"]["region"] != "center":
        raise RuntimeError(f"English instruction should parse center destination: {mcp_like}")

    print("english_case:", mcp_like["target"], mcp_like["destination"])
    print("status: OK")


def _assert_case(result: dict, case: dict) -> None:
    if result["status"] != case["status"]:
        raise RuntimeError(f"Unexpected status for {case['instruction']}: {result}")
    if result["action"] != case["action"]:
        raise RuntimeError(f"Unexpected action for {case['instruction']}: {result}")
    if result["target"]["object_name"] != case["object_name"]:
        raise RuntimeError(f"Unexpected object for {case['instruction']}: {result}")
    if result["target"]["color"] != case["color"]:
        raise RuntimeError(f"Unexpected color for {case['instruction']}: {result}")
    if result["target"]["shape"] != case["shape"]:
        raise RuntimeError(f"Unexpected shape for {case['instruction']}: {result}")

    if "matched_count" in case and len(result["matched_objects"]) != case["matched_count"]:
        raise RuntimeError(f"Unexpected matched count for {case['instruction']}: {result}")
    if case.get("destination") is None and "destination_region" not in case and result["destination"] is not None:
        raise RuntimeError(f"Unexpected destination for {case['instruction']}: {result}")
    if "destination_region" in case:
        if result["destination"] is None or result["destination"]["region"] != case["destination_region"]:
            raise RuntimeError(f"Unexpected destination region for {case['instruction']}: {result}")
        if len(result["destination"]["world_xy_m"]) != 2:
            raise RuntimeError(f"Destination should expose world_xy_m for {case['instruction']}: {result}")
    if not result["vl_prompt"]:
        raise RuntimeError(f"Expected a VL prompt for {case['instruction']}: {result}")


if __name__ == "__main__":
    main()
