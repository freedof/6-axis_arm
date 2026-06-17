from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "src" / "mcp_robot" / "server.py"


def main() -> None:
    process = subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        initialize = _request(process, 1, "initialize", {"clientInfo": {"name": "verify_mcp_robot_server"}})
        if initialize["result"]["serverInfo"]["name"] != "dobot-cr5-mujoco-mcp":
            raise RuntimeError("Unexpected MCP server name.")

        _notify(process, "notifications/initialized")

        listed = _request(process, 2, "tools/list", {})
        tool_names = {tool["name"] for tool in listed["result"]["tools"]}
        required = {
            "get_robot_capabilities",
            "list_available_scenes",
            "get_scene_state",
            "pick_cube",
            "render_d435i_preview",
            "vl_locate_object_3d",
            "plan_pick_from_target_3d",
            "vl_pick_cube",
            "multi_view_vl_locate_object_3d",
            "multi_view_vl_pick_cube",
        }
        missing = sorted(required - tool_names)
        if missing:
            raise RuntimeError(f"Missing expected MCP tools: {missing}")

        capabilities = _call_tool(process, 3, "get_robot_capabilities", {})
        if "pick_cube" not in capabilities["available_skills"]:
            raise RuntimeError("Capability response should include pick_cube.")
        if "render_d435i_preview" not in capabilities["available_skills"]:
            raise RuntimeError("Capability response should include render_d435i_preview.")
        if "vl_locate_object_3d" not in capabilities["available_skills"]:
            raise RuntimeError("Capability response should include vl_locate_object_3d.")
        if "vl_pick_cube" not in capabilities["available_skills"]:
            raise RuntimeError("Capability response should include vl_pick_cube.")
        if "multi_view_vl_pick_cube" not in capabilities["available_skills"]:
            raise RuntimeError("Capability response should include multi_view_vl_pick_cube.")

        scene_state = _call_tool(process, 4, "get_scene_state", {"scene_id": "gripper_pick_cube"})
        if scene_state["objects"][0]["name"] != "grasp_cube":
            raise RuntimeError("Pick scene state should expose grasp_cube.")

        d435i_state = _call_tool(process, 5, "get_scene_state", {"scene_id": "gripper_pick_cube_d435i"})
        if d435i_state["sensors"][0]["name"] != "d435i_depth":
            raise RuntimeError("D435i scene state should expose d435i_depth.")

        d435i_preview = _call_tool(
            process,
            6,
            "render_d435i_preview",
            {
                "width": 160,
                "height": 120,
                "output_dir": "outputs/d435i_preview/mcp_verify",
                "pose": "above",
            },
        )
        if d435i_preview["raw_depth_stats"]["valid_ratio"] < 0.05:
            raise RuntimeError(f"D435i preview depth has too few valid pixels: {d435i_preview}")

        vl_result = _call_tool(
            process,
            7,
            "vl_locate_object_3d",
            {
                "prompt": "pick the red block",
                "output_dir": "outputs/vl_region/mcp_verify",
                "pose": "scan",
                "width": 424,
                "height": 240,
            },
        )
        if vl_result["target_3d"]["valid_pixel_count"] < 20:
            raise RuntimeError(f"VL 3D estimate should use valid depth pixels: {vl_result}")

        pick_result = _call_tool(
            process,
            8,
            "pick_cube",
            {
                "render_gif": True,
                "frames": 120,
                "fps": 20,
                "width": 320,
                "height": 240,
                "output_path": "outputs/gripper_pick/mcp_pick_cube_verify.gif",
            },
        )
        if pick_result["status"] != "automatic_precheck_passed":
            raise RuntimeError(f"pick_cube did not pass automatic pre-check: {pick_result}")
        if not pick_result["metrics"]["lifted"]:
            raise RuntimeError("pick_cube metrics should report lifted=true.")

        planned_pick = _call_tool(
            process,
            9,
            "vl_pick_cube",
            {
                "prompt": "pick the red block",
                "provider": "color_fixture",
                "output_dir": "outputs/end_to_end/mcp_vl_pick_verify",
                "render_gif": True,
                "frames": 0,
                "fps": 20,
                "width": 320,
                "height": 240,
                "output_path": "outputs/end_to_end/mcp_vl_pick_verify.gif",
            },
        )
        if planned_pick["status"] != "automatic_precheck_passed":
            raise RuntimeError(f"vl_pick_cube did not pass automatic pre-check: {planned_pick}")
        if not planned_pick["metrics"]["lifted"]:
            raise RuntimeError("vl_pick_cube metrics should report lifted=true.")

        multi_view_pick = _call_tool(
            process,
            10,
            "multi_view_vl_pick_cube",
            {
                "prompt": "pick the red block",
                "provider": "color_fixture",
                "output_dir": "outputs/end_to_end/mcp_multi_view_vl_pick_verify",
                "poses": ["scan", "scan_left", "scan_right"],
                "max_parallel_vl": 3,
                "render_gif": False,
                "frames": 0,
                "fps": 20,
                "width": 320,
                "height": 240,
            },
        )
        if multi_view_pick["status"] != "automatic_precheck_passed":
            raise RuntimeError(f"multi_view_vl_pick_cube did not pass automatic pre-check: {multi_view_pick}")
        if not multi_view_pick["metrics"]["lifted"]:
            raise RuntimeError("multi_view_vl_pick_cube metrics should report lifted=true.")

        print("tools:", ", ".join(sorted(tool_names)))
        print("d435i_preview_rgb:", d435i_preview["files"]["rgb"])
        print("vl_target_world:", vl_result["target_3d"]["center_world_m"])
        print("pick_cube_status:", pick_result["status"])
        print("pick_cube_gif:", pick_result.get("gif"))
        print("vl_pick_cube_status:", planned_pick["status"])
        print("vl_pick_cube_gif:", planned_pick.get("gif"))
        print("multi_view_vl_pick_cube_status:", multi_view_pick["status"])
        print("multi_view_used_views:", multi_view_pick["perception"]["fusion"]["used_views"])
        print("status: OK")
    finally:
        process.kill()
        process.wait(timeout=5)


def _request(
    process: subprocess.Popen[str],
    request_id: int,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    _write(process, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
    response = _read(process)
    if response.get("id") != request_id:
        raise RuntimeError(f"Expected response id {request_id}, got {response}")
    if "error" in response:
        raise RuntimeError(f"MCP error: {response['error']}")
    return response


def _notify(process: subprocess.Popen[str], method: str) -> None:
    _write(process, {"jsonrpc": "2.0", "method": method, "params": {}})


def _call_tool(
    process: subprocess.Popen[str],
    request_id: int,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    response = _request(process, request_id, "tools/call", {"name": name, "arguments": arguments})
    result = response["result"]
    if result.get("isError"):
        raise RuntimeError(f"Tool returned isError=true: {result}")
    return result["structuredContent"]


def _write(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("MCP process stdin is closed.")
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()


def _read(process: subprocess.Popen[str]) -> dict[str, Any]:
    if process.stdout is None:
        raise RuntimeError("MCP process stdout is closed.")
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"MCP process closed without response. stderr={stderr}")
    return json.loads(line)


if __name__ == "__main__":
    main()
