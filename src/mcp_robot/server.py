from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import skills
except ImportError:
    from src.mcp_robot import skills


PROTOCOL_VERSION = "2024-11-05"


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


def main() -> None:
    server = RobotMcpServer()
    server.run()


class RobotMcpServer:
    def __init__(self) -> None:
        self.tools: dict[str, ToolHandler] = {
            "get_robot_capabilities": lambda args: skills.get_robot_capabilities(),
            "list_available_scenes": lambda args: skills.list_available_scenes(),
            "get_scene_state": self._get_scene_state,
            "generate_gripper_model": lambda args: skills.generate_gripper_model(),
            "generate_pick_scene": lambda args: skills.generate_pick_scene(),
            "generate_d435i_scene": lambda args: skills.generate_d435i_scene(),
            "render_d435i_preview": self._render_d435i_preview,
            "vl_locate_object_region": self._vl_locate_object_region,
            "estimate_region_3d": self._estimate_region_3d,
            "vl_locate_object_3d": self._vl_locate_object_3d,
            "simulate_pick_cube": self._simulate_pick_cube,
            "render_pick_cube_gif": self._render_pick_cube_gif,
            "pick_cube": self._pick_cube,
        }

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle(request)
            except Exception as exc:  # Keep the server alive after malformed input.
                response = self._error_response(None, -32700, str(exc))

            if response is not None:
                self._write_response(response)

    def _handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "dobot-cr5-mujoco-mcp",
                            "version": "0.1.0",
                        },
                    },
                }
            if method == "notifications/initialized":
                return None
            if method == "tools/list":
                return self._ok(request_id, {"tools": self._tool_specs()})
            if method == "tools/call":
                return self._call_tool(request_id, params)

            if request_id is None:
                return None
            return self._error_response(request_id, -32601, f"Method not found: {method}")
        except Exception as exc:
            return self._error_response(request_id, -32000, str(exc))

    def _call_tool(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in self.tools:
            return self._error_response(request_id, -32602, f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            return self._error_response(request_id, -32602, "Tool arguments must be an object.")

        with contextlib.redirect_stdout(sys.stderr):
            result = self.tools[name](arguments)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return self._ok(
            request_id,
            {
                "content": [{"type": "text", "text": text}],
                "structuredContent": result,
                "isError": False,
            },
        )

    def _get_scene_state(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.get_scene_state(args.get("scene_id", "gripper_pick_cube"))

    def _simulate_pick_cube(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.simulate_pick_cube(
            frames=int(args.get("frames", 120)),
            fps=int(args.get("fps", 20)),
        )

    def _render_pick_cube_gif(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.render_pick_cube_gif(
            args.get("output_path"),
            frames=int(args.get("frames", 160)),
            fps=int(args.get("fps", 20)),
            width=int(args.get("width", 960)),
            height=int(args.get("height", 720)),
            show_sites=bool(args.get("show_sites", False)),
        )

    def _render_d435i_preview(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.render_d435i_preview(
            args.get("output_dir"),
            width=int(args.get("width", 424)),
            height=int(args.get("height", 240)),
            seed=int(args.get("seed", 7)),
            pose=str(args.get("pose", "above")),
        )

    def _vl_locate_object_region(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.vl_locate_object_region(
            str(args.get("prompt", "target object")),
            args.get("output_dir"),
            provider=str(args.get("provider", "color_fixture")),
            manual_region=args.get("manual_region"),
            model=args.get("model"),
            config_path=args.get("config_path"),
            width=int(args.get("width", 424)),
            height=int(args.get("height", 240)),
            seed=int(args.get("seed", 7)),
            pose=str(args.get("pose", "scan")),
        )

    def _estimate_region_3d(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.estimate_region_3d(
            region=args["region"],
            depth_path=args["depth_path"],
            intrinsics=args["intrinsics"],
            extrinsic_world_to_camera=args["extrinsic_world_to_camera"],
        )

    def _vl_locate_object_3d(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.vl_locate_object_3d(
            str(args.get("prompt", "target object")),
            args.get("output_dir"),
            provider=str(args.get("provider", "color_fixture")),
            manual_region=args.get("manual_region"),
            model=args.get("model"),
            config_path=args.get("config_path"),
            width=int(args.get("width", 424)),
            height=int(args.get("height", 240)),
            seed=int(args.get("seed", 7)),
            pose=str(args.get("pose", "scan")),
        )

    def _pick_cube(self, args: dict[str, Any]) -> dict[str, Any]:
        return skills.pick_cube(
            render_gif=bool(args.get("render_gif", True)),
            output_path=args.get("output_path"),
            frames=int(args.get("frames", 160)),
            fps=int(args.get("fps", 20)),
            width=int(args.get("width", 960)),
            height=int(args.get("height", 720)),
            show_sites=bool(args.get("show_sites", False)),
        )

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_robot_capabilities",
                "description": "Return current CR5 simulation, gripper, planning, and validation capabilities.",
                "inputSchema": _object_schema({}),
            },
            {
                "name": "list_available_scenes",
                "description": "List known MuJoCo scenes and whether their generated model files exist.",
                "inputSchema": _object_schema({}),
            },
            {
                "name": "get_scene_state",
                "description": "Return structured state metadata for a scene.",
                "inputSchema": _object_schema(
                    {
                        "scene_id": {
                            "type": "string",
                            "default": "gripper_pick_cube",
                            "enum": list(skills.SCENES.keys()),
                        }
                    }
                ),
            },
            {
                "name": "generate_gripper_model",
                "description": "Generate the CR5 model with the simplified parallel gripper.",
                "inputSchema": _object_schema({}),
            },
            {
                "name": "generate_pick_scene",
                "description": "Generate the table and dynamic cube pick scene.",
                "inputSchema": _object_schema({}),
            },
            {
                "name": "generate_d435i_scene",
                "description": "Generate the simplified-gripper cube-pick scene with a gripper-mounted D435i camera.",
                "inputSchema": _object_schema({}),
            },
            {
                "name": "render_d435i_preview",
                "description": "Render RGB/depth previews from the gripper-mounted D435i camera.",
                "inputSchema": _object_schema(
                    {
                        "output_dir": {"type": "string"},
                        "width": {"type": "integer", "default": 424, "minimum": 1},
                        "height": {"type": "integer", "default": 240, "minimum": 1},
                        "seed": {"type": "integer", "default": 7},
                        "pose": {
                            "type": "string",
                            "default": "above",
                            "enum": ["ready", "above", "grasp", "lift", "scan"],
                        },
                    }
                ),
            },
            {
                "name": "vl_locate_object_region",
                "description": "Locate an object in the D435i RGB image and return a VL-style 2D region. The local provider is a deterministic fixture for integration testing.",
                "inputSchema": _vl_observation_schema(),
            },
            {
                "name": "estimate_region_3d",
                "description": "Lift a VL 2D region to a 3D target estimate using D435i depth and camera calibration.",
                "inputSchema": _object_schema(
                    {
                        "region": {"type": "object"},
                        "depth_path": {"type": "string"},
                        "intrinsics": {"type": "array"},
                        "extrinsic_world_to_camera": {"type": "array"},
                    },
                    required=["region", "depth_path", "intrinsics", "extrinsic_world_to_camera"],
                ),
            },
            {
                "name": "vl_locate_object_3d",
                "description": "Run VL-style object localization and lift the selected region to a 3D target estimate.",
                "inputSchema": _vl_observation_schema(),
            },
            {
                "name": "simulate_pick_cube",
                "description": "Run the simplified-gripper cube pick simulation and return lift metrics.",
                "inputSchema": _object_schema(
                    {
                        "frames": {"type": "integer", "default": 120, "minimum": 1},
                        "fps": {"type": "integer", "default": 20, "minimum": 1},
                    }
                ),
            },
            {
                "name": "render_pick_cube_gif",
                "description": "Render and validate a GIF for the simplified-gripper cube pick scene.",
                "inputSchema": _render_schema(),
            },
            {
                "name": "pick_cube",
                "description": "Execute the cube-pick skill, optionally render a GIF, and return validation metrics.",
                "inputSchema": _object_schema(
                    {
                        **_render_schema()["properties"],
                        "render_gif": {"type": "boolean", "default": True},
                    }
                ),
            },
        ]

    def _ok(self, request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _write_response(self, response: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        **({"required": required} if required else {}),
        "additionalProperties": False,
    }


def _vl_observation_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "prompt": {"type": "string", "default": "pick the target object"},
            "provider": {
                "type": "string",
                "default": "color_fixture",
                "enum": ["color_fixture", "manual_region", "openai_vision", "ark_coding_vision"],
            },
            "manual_region": {"type": "object"},
            "model": {"type": "string"},
            "config_path": {"type": "string"},
            "output_dir": {"type": "string"},
            "width": {"type": "integer", "default": 424, "minimum": 1},
            "height": {"type": "integer", "default": 240, "minimum": 1},
            "seed": {"type": "integer", "default": 7},
            "pose": {
                "type": "string",
                "default": "scan",
                "enum": ["ready", "above", "grasp", "lift", "scan"],
            },
        }
    )


def _render_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "output_path": {"type": "string"},
            "frames": {"type": "integer", "default": 160, "minimum": 1},
            "fps": {"type": "integer", "default": 20, "minimum": 1},
            "width": {"type": "integer", "default": 960, "minimum": 1},
            "height": {"type": "integer", "default": 720, "minimum": 1},
            "show_sites": {"type": "boolean", "default": False},
        }
    )


if __name__ == "__main__":
    main()
