from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.vl_region import estimate_region_3d as estimate_vl_region_3d
from src.perception.vl_region import locate_ark_coding_vision_region
from src.perception.vl_region import locate_codex_vision_region
from src.perception.vl_region import locate_manual_region, locate_openai_vision_region, locate_openrouter_vision_region, locate_openrouter_vision_regions, locate_red_region_fixture
from src.perception.vl_region import region3d_to_dict
from src.perception.language_goal import parse_language_goal as parse_language_goal_instruction
from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model
from src.sim.gripper_pick_motion import plan_pick_trajectory_from_target_3d, simulate_pick
from src.sim.pick_place_motion import PICK_PLACE_CANDIDATE_YAWS, PickPlaceSequenceItem, SEQUENCE_BRIDGE_SECONDS, pick_place_required_frames, pick_place_segments, pick_place_sequence_required_frames, plan_pick_place_trajectory, sequence_bridge_waypoints, simulate_pick_place, simulate_pick_place_sequence
from src.sim.gripper_pick_scene import CUBE_HALF_SIZE, DEFAULT_MULTI_OBJECT_MODEL, DEFAULT_MULTI_OBJECT_SPECS, DEFAULT_PICK_MODEL, TABLE_TOP_Z, TRAY_FLOOR_TOP_Z, TRAY_PLACE_SLOTS, object_specs_from_config, object_specs_to_dicts, tray_metadata, write_multi_object_scene_model, write_pick_scene_model
from src.sim.d435i_model import DEFAULT_D435I_GRIPPER_MODEL, DEFAULT_D435I_MULTI_OBJECT_MODEL, DEFAULT_D435I_PICK_MODEL, write_d435i_multi_object_scene_model, write_d435i_pick_scene_model
from src.sim.render_d435i_preview import DEFAULT_OUTPUT_DIR as DEFAULT_D435I_OUTPUT_DIR
from src.sim.render_d435i_preview import POSE_CHOICES
from src.sim.render_d435i_preview import render_preview as render_d435i_camera_preview
from src.sim.render_gripper_pick_gif import DEFAULT_OUTPUT as DEFAULT_PICK_GIF
from src.sim.render_gripper_pick_gif import render_gif as render_pick_gif
from src.sim.render_pick_place_gif import play_pick_place_sequence_viewer, play_pick_place_viewer, render_pick_place_gif, render_pick_place_sequence_gif
from src.sim.verify_render_gifs import validate_gif


SCENES: dict[str, dict[str, Any]] = {
    "cr5_simplified": {
        "name": "CR5 simplified roundtrip target scene",
        "model_path": ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_simplified.xml",
        "description": "6-axis CR5 baseline scene with red/blue roundtrip target markers.",
        "dynamic_objects": ["target_marker", "target_marker_b"],
    },
    "cr5_with_gripper": {
        "name": "CR5 with simplified parallel gripper",
        "model_path": DEFAULT_GRIPPER_MODEL,
        "description": "Generated 8-actuator CR5 model with a simple parallel gripper and no roundtrip target markers.",
        "dynamic_objects": [],
    },
    "gripper_pick_cube": {
        "name": "Simplified gripper cube-pick scene",
        "model_path": DEFAULT_PICK_MODEL,
        "description": "CR5 gripper scene with a table and one free-joint red cube.",
        "dynamic_objects": ["grasp_cube"],
    },
    "gripper_pick_cube_d435i": {
        "name": "Simplified gripper cube-pick scene with D435i",
        "model_path": DEFAULT_D435I_PICK_MODEL,
        "description": "CR5 gripper pick scene with a gripper-root D435i RGB-D camera.",
        "dynamic_objects": ["grasp_cube"],
    },
    "gripper_multi_object": {
        "name": "Simplified gripper multi-object tabletop scene",
        "model_path": DEFAULT_MULTI_OBJECT_MODEL,
        "description": "CR5 gripper scene with multiple colored boxes and cylinders on the table.",
        "dynamic_objects": [spec.name for spec in DEFAULT_MULTI_OBJECT_SPECS],
    },
    "gripper_multi_object_d435i": {
        "name": "Simplified gripper multi-object scene with D435i",
        "model_path": DEFAULT_D435I_MULTI_OBJECT_MODEL,
        "description": "CR5 gripper multi-object tabletop scene with a gripper-root D435i RGB-D camera.",
        "dynamic_objects": [spec.name for spec in DEFAULT_MULTI_OBJECT_SPECS],
    },
}


MULTI_VIEW_DEFAULT_POSES = ("scan_high", "scan_front_high", "scan_left_high", "scan_right_high")
VL_PROVIDERS = ("color_fixture", "manual_region", "codex_vision", "openai_vision", "ark_coding_vision", "openrouter_vision")
DEPTH_VARIANTS = ("raw", "noisy")


def get_robot_capabilities() -> dict[str, Any]:
    return {
        "robot": "Dobot CR5 simplified",
        "simulator": "MuJoCo",
        "models": {
            "baseline": _relative(SCENES["cr5_simplified"]["model_path"]),
            "gripper": _relative(DEFAULT_GRIPPER_MODEL),
            "pick_scene": _relative(DEFAULT_PICK_MODEL),
            "pick_planning": _relative(ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_gripper_pick_planning.xml"),
            "gripper_d435i": _relative(DEFAULT_D435I_GRIPPER_MODEL),
            "pick_scene_d435i": _relative(DEFAULT_D435I_PICK_MODEL),
            "multi_object_scene": _relative(DEFAULT_MULTI_OBJECT_MODEL),
            "multi_object_scene_d435i": _relative(DEFAULT_D435I_MULTI_OBJECT_MODEL),
        },
        "actuation": {
            "baseline_dof": 6,
            "gripper_model_dof": 8,
            "gripper_type": "simplified two-finger parallel gripper",
        },
        "available_skills": [
            "generate_gripper_model",
            "generate_pick_scene",
            "generate_multi_object_scene",
            "parse_language_goal",
            "simulate_pick_cube",
            "render_pick_cube_gif",
            "pick_cube",
            "generate_d435i_scene",
            "render_d435i_preview",
            "vl_locate_object_region",
            "vl_locate_object_3d",
            "estimate_region_3d",
            "plan_pick_from_target_3d",
            "vl_pick_cube",
            "multi_view_vl_locate_object_3d",
            "multi_view_vl_pick_cube",
            "multi_object_vl_locate",
            "language_multi_view_pick_and_place",
            "get_live_robot_state",
        ],
        "validation_policy": "automatic pre-check passed; waiting for user GIF confirmation",
    }


def get_live_robot_state(output_dir: str | Path | None = None) -> dict[str, Any]:
    session_dir = ROOT / "outputs" / "live_session" if output_dir is None else Path(output_dir)
    if not session_dir.is_absolute():
        session_dir = ROOT / session_dir
    status = _read_json_file(session_dir / "status.json")
    robot_state = _read_json_file(session_dir / "robot_state.json")
    waypoint_trace = _read_json_file(session_dir / "waypoint_trace.json")
    if not isinstance(waypoint_trace, list):
        waypoint_trace = []
    pid = int(status.get("pid") or 0) if isinstance(status, dict) else 0
    return {
        "status": status if isinstance(status, dict) else {},
        "live_process_alive": _pid_is_alive(pid),
        "robot_state": robot_state if isinstance(robot_state, dict) else {},
        "latest_waypoint": waypoint_trace[-1] if waypoint_trace else None,
        "waypoint_count": len(waypoint_trace),
        "waypoint_labels": [str(item.get("label")) for item in waypoint_trace if isinstance(item, dict)],
        "trace_path": _relative(session_dir / "waypoint_trace.json"),
        "state_path": _relative(session_dir / "robot_state.json"),
    }


def list_available_scenes() -> dict[str, Any]:
    scenes = []
    for scene_id, scene in SCENES.items():
        scenes.append(
            {
                "scene_id": scene_id,
                "name": scene["name"],
                "description": scene["description"],
                "model_path": _relative(scene["model_path"]),
                "dynamic_objects": scene["dynamic_objects"],
                "exists": Path(scene["model_path"]).exists(),
            }
        )
    return {"scenes": scenes}


def get_scene_state(scene_id: str = "gripper_pick_cube") -> dict[str, Any]:
    if scene_id not in SCENES:
        raise ValueError(f"Unknown scene_id: {scene_id}")
    if scene_id == "cr5_with_gripper":
        model_path = write_gripper_model(DEFAULT_GRIPPER_MODEL)
    elif scene_id == "gripper_pick_cube":
        model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    elif scene_id == "gripper_pick_cube_d435i":
        model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    elif scene_id == "gripper_multi_object":
        model_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL)
    elif scene_id == "gripper_multi_object_d435i":
        model_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL)
    else:
        model_path = Path(SCENES[scene_id]["model_path"])

    scene = SCENES[scene_id]
    state: dict[str, Any] = {
        "scene_id": scene_id,
        "name": scene["name"],
        "model_path": _relative(model_path),
        "dynamic_objects": scene["dynamic_objects"],
    }
    if scene_id in ("gripper_pick_cube", "gripper_pick_cube_d435i"):
        state["objects"] = [
            {
                "name": "grasp_cube",
                "type": "box",
                "color": "red",
                "free_joint": True,
                "mass_kg": 0.020,
                "half_size_m": [0.016, 0.018, 0.030],
                "initial_center_m": [0.35, -0.55, 0.065],
                "friction": [3.0, 0.08, 0.006],
            }
        ]
        state["environment"] = {
            "table": {
                "name": "pick_table",
                "top_z_m": 0.035,
                "friction": [1.0, 0.02, 0.002],
            }
        }
    if scene_id in ("gripper_multi_object", "gripper_multi_object_d435i"):
        state["objects"] = object_specs_to_dicts(DEFAULT_MULTI_OBJECT_SPECS)
        state["environment"] = {
            "table": {
                "name": "pick_table",
                "top_z_m": TABLE_TOP_Z,
                "friction": [1.0, 0.02, 0.002],
            },
            "tray": tray_metadata(),
        }
    if scene_id in ("gripper_pick_cube_d435i", "gripper_multi_object_d435i"):
        state["sensors"] = [
            {
                "name": "d435i_depth",
                "type": "depth_camera",
                "mount": "parallel_gripper",
                "default_preview_pose": "above",
                "outputs": ["rgb", "raw_depth", "noisy_depth", "intrinsics", "world_to_camera_extrinsic"],
            },
            {
                "name": "d435i_rgb",
                "type": "rgb_camera",
                "mount": "parallel_gripper",
            },
        ]
    return state


def generate_gripper_model() -> dict[str, Any]:
    model_path = write_gripper_model(DEFAULT_GRIPPER_MODEL)
    return {
        "status": "ok",
        "model_path": _relative(model_path),
        "scene_id": "cr5_with_gripper",
    }


def generate_pick_scene() -> dict[str, Any]:
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    return {
        "status": "ok",
        "model_path": _relative(model_path),
        "scene_id": "gripper_pick_cube",
    }


def generate_multi_object_scene(objects: list[dict[str, Any]] | None = None, *, include_d435i: bool = True) -> dict[str, Any]:
    specs = DEFAULT_MULTI_OBJECT_SPECS if objects is None else object_specs_from_config(objects)
    model_path = write_multi_object_scene_model(DEFAULT_MULTI_OBJECT_MODEL, objects=specs)
    response: dict[str, Any] = {
        "status": "ok",
        "scene_id": "gripper_multi_object",
        "model_path": _relative(model_path),
        "objects": object_specs_to_dicts(specs),
    }
    if include_d435i:
        d435i_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL, objects=specs)
        response["d435i_scene_id"] = "gripper_multi_object_d435i"
        response["d435i_model_path"] = _relative(d435i_path)
        response["camera_names"] = ["d435i_depth", "d435i_rgb"]
    return response


def parse_language_goal(instruction: str, objects: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    scene_objects = object_specs_to_dicts(DEFAULT_MULTI_OBJECT_SPECS) if objects is None else objects
    parsed = parse_language_goal_instruction(instruction, objects=scene_objects)
    parsed["scene_id"] = "gripper_multi_object_d435i"
    parsed["available_objects"] = scene_objects
    return parsed


def _resolve_language_goal(
    instruction: str,
    language_goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if language_goal is None:
        return parse_language_goal(instruction)
    if not isinstance(language_goal, dict):
        raise ValueError("language_goal must be an object when provided.")

    parsed = dict(language_goal)
    parsed.setdefault("status", "ok")
    parsed.setdefault("instruction", instruction)
    parsed.setdefault("scene_id", "gripper_multi_object_d435i")
    parsed.setdefault("available_objects", object_specs_to_dicts(DEFAULT_MULTI_OBJECT_SPECS))
    parsed.setdefault("ambiguities", [])
    parsed.setdefault("warnings", [])

    target = parsed.get("target")
    if not isinstance(target, dict):
        raise ValueError("language_goal.target must be an object.")
    target = dict(target)
    object_names = list(target.get("object_names") or [])
    object_name = target.get("object_name")
    if object_name and object_name not in object_names:
        object_names.insert(0, str(object_name))
    if not object_name and len(object_names) == 1:
        object_name = object_names[0]
    if object_name:
        target["object_name"] = str(object_name)
    target["object_names"] = [str(name) for name in object_names]
    target.setdefault("quantifier", "all" if len(object_names) > 1 else "one")

    if object_name:
        target_object = _object_by_name(str(object_name))
        target.setdefault("color", target_object.get("color"))
        target.setdefault("shape", target_object.get("shape"))
    target.setdefault(
        "constraints",
        {
            "color": target.get("color"),
            "shape": target.get("shape"),
        },
    )
    parsed["target"] = target
    parsed.setdefault("matched_objects", [_candidate_from_object(_object_by_name(name)) for name in target["object_names"]])
    if not parsed.get("vl_prompt"):
        parsed["vl_prompt"] = _build_goal_vl_prompt(parsed)
    return parsed


def _candidate_from_object(item: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "shape", "color", "initial_center_m", "position_xy_m")
    return {key: item.get(key) for key in keys if item.get(key) is not None}


def _build_goal_vl_prompt(parsed: dict[str, Any]) -> str:
    target = parsed.get("target", {})
    name = target.get("object_name")
    color = target.get("color") or "specified"
    shape = target.get("shape") or "object"
    shape_text = "box/cube" if shape == "box" else str(shape)
    bbox_guidance = (
        "Return the full visible outer bbox of the physical object, including any gray-lit top face "
        "attached to the colored side face. Do not return a tiny color patch, edge, shadow, table, tray, or gripper."
    )
    if name:
        return (
            f"Locate the {color} {shape_text} target named {name} on the tabletop. "
            f"{bbox_guidance}"
        )
    return (
        f"Locate one visible {color} {shape_text} target object on the tabletop. "
        f"{bbox_guidance}"
    )


def generate_d435i_scene() -> dict[str, Any]:
    model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    return {
        "status": "ok",
        "model_path": _relative(model_path),
        "scene_id": scene_id,
        "camera_names": ["d435i_depth", "d435i_rgb"],
    }


def render_d435i_preview(
    output_dir: str | Path | None = None,
    *,
    width: int = 424,
    height: int = 240,
    seed: int = 7,
    pose: str = "above",
    scene_id: str = "gripper_pick_cube_d435i",
) -> dict[str, Any]:
    if scene_id == "gripper_pick_cube_d435i":
        model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    elif scene_id == "gripper_multi_object_d435i":
        model_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL)
    else:
        raise ValueError(f"render_d435i_preview requires a D435i scene, got: {scene_id}")
    output = DEFAULT_D435I_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    result = render_d435i_camera_preview(model_path, output, width=width, height=height, seed=seed, pose=pose)
    return {
        "status": "ok",
        "scene_id": scene_id,
        "model_path": _relative(model_path),
        "pose": result["pose"],
        "camera": "d435i_depth",
        "files": {
            "rgb": _relative(Path(result["rgb_path"])),
            "context_rgb": _relative(Path(result["context_rgb_path"])),
            "raw_depth": _relative(Path(result["raw_depth_path"])),
            "noisy_depth": _relative(Path(result["noisy_depth_path"])),
            "raw_depth_vis": _relative(Path(result["raw_depth_vis_path"])),
            "noisy_depth_vis": _relative(Path(result["noisy_depth_vis_path"])),
        },
        "intrinsics": result["intrinsics"],
        "extrinsic_world_to_camera": result["extrinsic_world_to_camera"],
        "raw_depth_stats": result["raw_depth_stats"],
        "noisy_depth_stats": result["noisy_depth_stats"],
    }

def vl_locate_object_region(
    prompt: str,
    output_dir: str | Path | None = None,
    *,
    provider: str = "color_fixture",
    manual_region: dict[str, Any] | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    width: int = 424,
    height: int = 240,
    seed: int = 7,
    pose: str = "scan",
    scene_id: str = "gripper_pick_cube_d435i",
) -> dict[str, Any]:
    if provider not in VL_PROVIDERS:
        raise ValueError(f"Unknown VL provider: {provider}")
    output = _resolve_output_dir(output_dir, "vl_region")
    observation = render_d435i_preview(output, width=width, height=height, seed=seed, pose=pose, scene_id=scene_id)
    rgb_path = ROOT / observation["files"]["rgb"]
    overlay_path = output / "vl_region_overlay.png"
    if provider == "color_fixture":
        region = locate_red_region_fixture(rgb_path, prompt=prompt, output_path=overlay_path)
        provider_note = "color_fixture is a deterministic stand-in for validating the VL-to-depth interface."
    elif provider == "manual_region":
        if manual_region is None:
            raise ValueError("manual_region is required when provider='manual_region'.")
        region = locate_manual_region(manual_region, prompt=prompt, rgb_path=rgb_path, output_path=overlay_path)
        provider_note = "manual_region uses caller-provided coordinates for debugging and repeatable acceptance checks."
    elif provider == "codex_vision":
        if manual_region is None:
            raise ValueError("manual_region is required when provider='codex_vision'. Codex must inspect the RGB image and provide a bbox/point.")
        region = locate_codex_vision_region(manual_region, prompt=prompt, rgb_path=rgb_path, output_path=overlay_path)
        provider_note = "codex_vision uses a Codex-inspected bbox/point from the current interactive session."
    elif provider == "openai_vision":
        region = locate_openai_vision_region(
            rgb_path,
            prompt=prompt,
            output_path=overlay_path,
            model=model,
            config_path=config_path,
        )
        provider_note = "openai_vision calls the OpenAI Responses API using config/vl_providers.local.json."
    elif provider == "ark_coding_vision":
        region = locate_ark_coding_vision_region(
            rgb_path,
            prompt=prompt,
            output_path=overlay_path,
            model=model,
            config_path=config_path,
        )
        provider_note = (
            "ark_coding_vision calls the Ark coding OpenAI-compatible chat-completions endpoint; "
            "it uses config/vl_providers.local.json."
        )
    else:
        region = locate_openrouter_vision_region(
            rgb_path,
            prompt=prompt,
            output_path=overlay_path,
            model=model,
            config_path=config_path,
        )
        provider_note = (
            "openrouter_vision calls the OpenRouter OpenAI-compatible chat-completions endpoint; "
            "it uses config/vl_providers.local.json."
        )
    region["overlay_path"] = _relative(Path(region["overlay_path"])) if region.get("overlay_path") else None
    return {
        "status": "ok",
        "provider": provider,
        "provider_note": provider_note,
        "prompt": prompt,
        "observation": observation,
        "region": region,
    }


def _locate_region_from_observation(
    *,
    prompt: str,
    observation: dict[str, Any],
    output_dir: Path,
    provider: str,
    manual_region: dict[str, Any] | None,
    model: str | None,
    config_path: str | Path | None,
    overlay_name: str = "vl_region_overlay.png",
) -> dict[str, Any]:
    rgb_path = ROOT / observation["files"]["rgb"]
    overlay_path = output_dir / overlay_name
    if provider == "color_fixture":
        region = locate_red_region_fixture(rgb_path, prompt=prompt, output_path=overlay_path)
    elif provider == "manual_region":
        if manual_region is None:
            raise ValueError("manual_region is required when provider='manual_region'.")
        region = locate_manual_region(manual_region, prompt=prompt, rgb_path=rgb_path, output_path=overlay_path)
    elif provider == "codex_vision":
        if manual_region is None:
            raise ValueError("manual_region is required when provider='codex_vision'. Codex must inspect the RGB image and provide a bbox/point.")
        region = locate_codex_vision_region(manual_region, prompt=prompt, rgb_path=rgb_path, output_path=overlay_path)
    elif provider == "openai_vision":
        region = locate_openai_vision_region(
            rgb_path,
            prompt=prompt,
            output_path=overlay_path,
            model=model,
            config_path=config_path,
        )
    elif provider == "ark_coding_vision":
        region = locate_ark_coding_vision_region(
            rgb_path,
            prompt=prompt,
            output_path=overlay_path,
            model=model,
            config_path=config_path,
        )
    elif provider == "openrouter_vision":
        region = locate_openrouter_vision_region(
            rgb_path,
            prompt=prompt,
            output_path=overlay_path,
            model=model,
            config_path=config_path,
        )
    else:
        raise ValueError(f"Unknown VL provider: {provider}")
    region["overlay_path"] = _relative(Path(region["overlay_path"])) if region.get("overlay_path") else None
    return region


def estimate_region_3d(
    *,
    region: dict[str, Any],
    depth_path: str | Path,
    intrinsics: list[list[float]],
    extrinsic_world_to_camera: list[list[float]],
    bbox_expansion: float = 1.25,
    min_world_z_m: float | None = None,
) -> dict[str, Any]:
    depth = Path(depth_path)
    if not depth.is_absolute():
        depth = ROOT / depth
    result = estimate_vl_region_3d(
        depth_path=depth,
        intrinsics=intrinsics,
        extrinsic_world_to_camera=extrinsic_world_to_camera,
        region=region,
        bbox_expansion=bbox_expansion,
        min_world_z_m=min_world_z_m,
    )
    return {
        "status": "ok",
        "region": region,
        "target_3d": region3d_to_dict(result),
    }


def vl_locate_object_3d(
    prompt: str,
    output_dir: str | Path | None = None,
    *,
    provider: str = "color_fixture",
    manual_region: dict[str, Any] | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    width: int = 424,
    height: int = 240,
    seed: int = 7,
    pose: str = "scan",
    depth_variant: str = "raw",
    scene_id: str = "gripper_pick_cube_d435i",
) -> dict[str, Any]:
    depth_file_key = _depth_file_key(depth_variant)
    located = vl_locate_object_region(
        prompt,
        output_dir,
        provider=provider,
        manual_region=manual_region,
        model=model,
        config_path=config_path,
        width=width,
        height=height,
        seed=seed,
        pose=pose,
        scene_id=scene_id,
    )
    observation = located["observation"]
    estimate = estimate_region_3d(
        region=located["region"],
        depth_path=observation["files"][depth_file_key],
        intrinsics=observation["intrinsics"],
        extrinsic_world_to_camera=observation["extrinsic_world_to_camera"],
    )
    return {
        "status": "ok",
        "scene_id": scene_id,
        "prompt": prompt,
        "provider": provider,
        "depth_variant": depth_variant,
        "depth_file": observation["files"][depth_file_key],
        "observation": observation,
        "region": located["region"],
        "target_3d": estimate["target_3d"],
        "next_step": "Use plan_pick_from_target_3d or vl_pick_cube to convert target_3d into grasp poses and an RRT-Connect pick path.",
    }


def multi_view_vl_locate_object_3d(
    prompt: str,
    output_dir: str | Path | None = None,
    *,
    provider: str = "color_fixture",
    manual_regions: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    camera_width: int = 424,
    camera_height: int = 240,
    seed: int = 7,
    poses: list[str] | tuple[str, ...] = MULTI_VIEW_DEFAULT_POSES,
    max_parallel_vl: int = 4,
    min_valid_pixels: int = 40,
    min_surface_z_m: float | None = None,
    max_surface_z_m: float = 0.140,
    max_cluster_radius_m: float = 0.040,
    min_accepted_views: int = 1,
    depth_variant: str = "raw",
    scene_id: str = "gripper_pick_cube_d435i",
) -> dict[str, Any]:
    if provider not in VL_PROVIDERS:
        raise ValueError(f"Unknown VL provider: {provider}")
    depth_file_key = _depth_file_key(depth_variant)
    selected_poses = tuple(str(pose) for pose in poses)
    for pose in selected_poses:
        if pose not in POSE_CHOICES:
            raise ValueError(f"Unknown D435i pose for multi-view: {pose}")

    output = _resolve_output_dir(output_dir, "multi_view_vl_region")
    observations: dict[str, dict[str, Any]] = {}
    for index, pose in enumerate(selected_poses):
        view_dir = output / f"{index:02d}_{pose}"
        observations[pose] = render_d435i_preview(
            view_dir,
            width=camera_width,
            height=camera_height,
            seed=seed + index,
            pose=pose,
            scene_id=scene_id,
        )

    candidates: list[dict[str, Any]] = []
    workers = max(1, min(int(max_parallel_vl), len(selected_poses)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _multi_view_candidate,
                prompt=prompt,
                pose=pose,
                observation=observations[pose],
                output_dir=output / f"{index:02d}_{pose}",
                provider=provider,
                manual_region=(manual_regions or {}).get(pose) if manual_regions else None,
                model=model,
                config_path=config_path,
                depth_file_key=depth_file_key,
                depth_variant=depth_variant,
            ): pose
            for index, pose in enumerate(selected_poses)
        }
        for future in as_completed(futures):
            pose = futures[future]
            try:
                candidates.append(future.result())
            except Exception as exc:
                candidates.append(
                    {
                        "pose": pose,
                        "accepted": False,
                        "reject_reason": f"vl_or_depth_failed: {exc}",
                        "observation": observations[pose],
                    }
                )

    candidates.sort(key=lambda item: selected_poses.index(item["pose"]))
    min_z = TABLE_TOP_Z + CUBE_HALF_SIZE[2] + 0.010 if min_surface_z_m is None else float(min_surface_z_m)
    candidates = _score_multi_view_candidates(
        candidates,
        min_valid_pixels=min_valid_pixels,
        min_surface_z_m=min_z,
        max_surface_z_m=max_surface_z_m,
        max_cluster_radius_m=max_cluster_radius_m,
    )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    required_accepted = max(int(min_accepted_views), 2 if provider in ("openai_vision", "ark_coding_vision", "openrouter_vision") else 1)
    if not accepted:
        return {
            "status": "failed",
            "scene_id": scene_id,
            "prompt": prompt,
            "provider": provider,
            "depth_variant": depth_variant,
            "poses": list(selected_poses),
            "candidates": candidates,
            "fusion": {
                "accepted_count": 0,
                "rejected_count": len(candidates),
                "reason": "no multi-view candidates passed geometry and consistency checks",
            },
        }
    if len(accepted) < required_accepted:
        return {
            "status": "failed",
            "scene_id": scene_id,
            "prompt": prompt,
            "provider": provider,
            "depth_variant": depth_variant,
            "poses": list(selected_poses),
            "candidates": candidates,
            "fusion": {
                "accepted_count": len(accepted),
                "rejected_count": len(candidates) - len(accepted),
                "used_views": [candidate["pose"] for candidate in accepted],
                "rejected_views": [candidate["pose"] for candidate in candidates if not candidate["accepted"]],
                "required_accepted_views": required_accepted,
                "reason": "not enough accepted views for reliable external-VL grasp planning",
            },
        }

    points = np.asarray([candidate["target_3d"]["center_world_m"] for candidate in accepted], dtype=float)
    fused = np.median(points, axis=0)
    mean_distance = float(np.mean(np.linalg.norm(points - fused, axis=1))) if len(points) else 0.0
    if len(accepted) == 1:
        confidence = min(float(accepted[0].get("region", {}).get("confidence", 0.5)), 0.55)
    else:
        confidence = float(np.clip(1.0 - mean_distance / max(max_cluster_radius_m, 1e-6), 0.05, 1.0))
    return {
        "status": "ok",
        "scene_id": scene_id,
        "prompt": prompt,
        "provider": provider,
        "depth_variant": depth_variant,
        "poses": list(selected_poses),
        "candidates": candidates,
        "fusion": {
            "fused_target_surface_world_m": _round_vector(fused),
            "used_views": [candidate["pose"] for candidate in accepted],
            "rejected_views": [candidate["pose"] for candidate in candidates if not candidate["accepted"]],
            "accepted_count": len(accepted),
            "rejected_count": len(candidates) - len(accepted),
            "mean_distance_to_fused_m": round(mean_distance, 6),
            "confidence": round(confidence, 4),
            "rules": {
                "min_valid_pixels": int(min_valid_pixels),
                "min_surface_z_m": round(float(min_z), 6),
                "max_surface_z_m": round(float(max_surface_z_m), 6),
                "max_cluster_radius_m": round(float(max_cluster_radius_m), 6),
                "min_accepted_views": int(required_accepted),
            },
        },
    }



def multi_object_vl_locate(
    instruction: str,
    output_dir: str | Path | None = None,
    *,
    provider: str = "color_fixture",
    manual_regions: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    camera_width: int = 424,
    camera_height: int = 240,
    seed: int = 7,
    poses: list[str] | tuple[str, ...] = MULTI_VIEW_DEFAULT_POSES,
    max_parallel_vl: int = 4,
    min_accepted_views: int = 1,
    depth_variant: str = "raw",
    language_goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = _resolve_language_goal(instruction, language_goal)
    if parsed["status"] != "ok":
        return {
            "status": "failed",
            "scene_id": "gripper_multi_object_d435i",
            "skill": "multi_object_vl_locate",
            "language_goal": parsed,
            "reason": "language goal must resolve exactly one target object before VL grounding",
        }
    target_object = _object_by_name(str(parsed["target"]["object_name"]))
    object_half_height = _object_half_height_m(target_object)
    located = multi_view_vl_locate_object_3d(
        parsed["vl_prompt"],
        output_dir,
        provider=provider,
        manual_regions=manual_regions,
        model=model,
        config_path=config_path,
        camera_width=camera_width,
        camera_height=camera_height,
        seed=seed,
        poses=poses,
        max_parallel_vl=max_parallel_vl,
        min_valid_pixels=25,
        min_surface_z_m=TABLE_TOP_Z + object_half_height + 0.004,
        max_surface_z_m=TABLE_TOP_Z + object_half_height * 2.0 + 0.060,
        min_accepted_views=min_accepted_views,
        depth_variant=depth_variant,
        scene_id="gripper_multi_object_d435i",
    )
    return {
        "status": located["status"],
        "scene_id": "gripper_multi_object_d435i",
        "skill": "multi_object_vl_locate",
        "instruction": instruction,
        "language_goal": parsed,
        "target_object": target_object,
        "object_half_height_m": round(object_half_height, 6),
        "perception": located,
    }


def multi_target_vl_locate_once(
    target_names: list[str] | tuple[str, ...],
    output_dir: str | Path | None = None,
    *,
    provider: str = "openrouter_vision",
    model: str | None = None,
    config_path: str | Path | None = None,
    camera_width: int = 424,
    camera_height: int = 240,
    seed: int = 7,
    poses: list[str] | tuple[str, ...] = MULTI_VIEW_DEFAULT_POSES,
    max_parallel_vl: int = 4,
    min_accepted_views: int = 1,
    depth_variant: str = "raw",
) -> dict[str, Any]:
    if provider != "openrouter_vision":
        raise ValueError("multi_target_vl_locate_once currently supports provider='openrouter_vision'.")
    depth_file_key = _depth_file_key(depth_variant)
    selected_poses = tuple(str(pose) for pose in poses)
    for pose in selected_poses:
        if pose not in POSE_CHOICES:
            raise ValueError(f"Unknown D435i pose for multi-target VL: {pose}")

    output = _resolve_output_dir(output_dir, "multi_target_vl_region")
    target_objects = [_object_by_name(str(name)) for name in target_names]
    observations: dict[str, dict[str, Any]] = {}
    for index, pose in enumerate(selected_poses):
        view_dir = output / f"{index:02d}_{pose}"
        observations[pose] = render_d435i_preview(
            view_dir,
            width=camera_width,
            height=camera_height,
            seed=seed + index,
            pose=pose,
            scene_id="gripper_multi_object_d435i",
        )

    regions_by_pose: dict[str, dict[str, dict[str, Any]]] = {}
    workers = max(1, min(int(max_parallel_vl), len(selected_poses)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                locate_openrouter_vision_regions,
                ROOT / observations[pose]["files"]["rgb"],
                targets=target_objects,
                output_path=output / f"{index:02d}_{pose}" / f"{pose}_multi_target_vl_overlay.png",
                model=model,
                config_path=config_path,
            ): pose
            for index, pose in enumerate(selected_poses)
        }
        for future in as_completed(futures):
            pose = futures[future]
            try:
                regions_by_pose[pose] = future.result()
            except Exception as exc:
                regions_by_pose[pose] = {"__error__": {"error": str(exc)}}

    return multi_target_vl_results_from_observations(
        target_names,
        observations=observations,
        regions_by_pose=regions_by_pose,
        provider=provider,
        depth_variant=depth_variant,
        poses=selected_poses,
        min_accepted_views=min_accepted_views,
    )


def multi_target_vl_results_from_observations(
    target_names: list[str] | tuple[str, ...],
    *,
    observations: dict[str, dict[str, Any]],
    regions_by_pose: dict[str, dict[str, dict[str, Any]]],
    provider: str,
    depth_variant: str,
    poses: list[str] | tuple[str, ...],
    min_accepted_views: int = 1,
) -> dict[str, Any]:
    depth_file_key = _depth_file_key(depth_variant)
    selected_poses = tuple(str(pose) for pose in poses)
    target_objects = [_object_by_name(str(name)) for name in target_names]
    results: dict[str, Any] = {}
    required_accepted = max(int(min_accepted_views), 2)
    for target_object in target_objects:
        name = str(target_object["name"])
        half_height = _object_half_height_m(target_object)
        candidates: list[dict[str, Any]] = []
        for pose in selected_poses:
            observation = observations[pose]
            pose_regions = regions_by_pose.get(pose, {})
            if "__error__" in pose_regions:
                candidates.append(
                    {
                        "pose": pose,
                        "accepted": False,
                        "reject_reason": f"vl_failed: {pose_regions['__error__']['error']}",
                        "observation": observation,
                    }
                )
                continue
            region = pose_regions.get(name)
            if region is None:
                candidates.append(
                    {
                        "pose": pose,
                        "accepted": False,
                        "reject_reason": "target_missing_from_multi_target_vl_response",
                        "observation": observation,
                    }
                )
                continue
            depth_path = Path(observation["files"][depth_file_key])
            if not depth_path.is_absolute():
                depth_path = ROOT / depth_path
            try:
                estimate3d = estimate_vl_region_3d(
                    depth_path=depth_path,
                    intrinsics=observation["intrinsics"],
                    extrinsic_world_to_camera=observation["extrinsic_world_to_camera"],
                    region=region,
                    foreground_quantile=0.05,
                    foreground_margin_m=0.010,
                    bbox_expansion=1.25,
                    min_world_z_m=TABLE_TOP_Z + 0.004,
                )
                candidates.append(
                    {
                        "pose": pose,
                        "accepted": False,
                        "reject_reason": "not_scored",
                        "observation": observation,
                        "depth_variant": depth_variant,
                        "depth_file": _relative(depth_path),
                        "region": region,
                        "target_3d": region3d_to_dict(estimate3d),
                    }
                )
            except Exception as exc:
                candidates.append(
                    {
                        "pose": pose,
                        "accepted": False,
                        "reject_reason": f"depth_failed: {exc}",
                        "observation": observation,
                        "region": region,
                    }
                )

        min_z = TABLE_TOP_Z + half_height + 0.004
        max_z = TABLE_TOP_Z + half_height * 2.0 + 0.060
        candidates = _score_multi_view_candidates(
            candidates,
            min_valid_pixels=25,
            min_surface_z_m=min_z,
            max_surface_z_m=max_z,
            max_cluster_radius_m=0.040,
        )
        accepted = [candidate for candidate in candidates if candidate["accepted"]]
        if len(accepted) >= required_accepted:
            points = np.asarray([candidate["target_3d"]["center_world_m"] for candidate in accepted], dtype=float)
            fused = np.median(points, axis=0)
            mean_distance = float(np.mean(np.linalg.norm(points - fused, axis=1))) if len(points) else 0.0
            confidence = float(np.clip(1.0 - mean_distance / 0.040, 0.05, 1.0))
            status = "ok"
            fusion = {
                "fused_target_surface_world_m": _round_vector(fused),
                "used_views": [candidate["pose"] for candidate in accepted],
                "rejected_views": [candidate["pose"] for candidate in candidates if not candidate["accepted"]],
                "accepted_count": len(accepted),
                "rejected_count": len(candidates) - len(accepted),
                "mean_distance_to_fused_m": round(mean_distance, 6),
                "confidence": round(confidence, 4),
                "rules": {
                    "min_valid_pixels": 25,
                    "min_surface_z_m": round(float(min_z), 6),
                    "max_surface_z_m": round(float(max_z), 6),
                    "max_cluster_radius_m": 0.040,
                    "min_accepted_views": int(required_accepted),
                },
            }
        else:
            status = "failed"
            fusion = {
                "accepted_count": len(accepted),
                "rejected_count": len(candidates) - len(accepted),
                "used_views": [candidate["pose"] for candidate in accepted],
                "rejected_views": [candidate["pose"] for candidate in candidates if not candidate["accepted"]],
                "required_accepted_views": required_accepted,
                "reason": "not enough accepted views for multi-target VL grounding",
            }
        perception = {
            "status": status,
            "scene_id": "gripper_multi_object_d435i",
            "prompt": "Locate all requested cube targets in one pass.",
            "provider": provider,
            "depth_variant": depth_variant,
            "poses": list(selected_poses),
            "candidates": candidates,
            "fusion": fusion,
            "mode": "multi_target_one_request_per_view",
        }
        results[name] = {
            "status": status,
            "scene_id": "gripper_multi_object_d435i",
            "skill": "multi_target_vl_locate_once",
            "target_object": target_object,
            "object_half_height_m": round(half_height, 6),
            "perception": perception,
        }

    return {
        "status": "ok" if all(item["status"] == "ok" for item in results.values()) else "failed",
        "scene_id": "gripper_multi_object_d435i",
        "skill": "multi_target_vl_locate_once",
        "provider": provider,
        "depth_variant": depth_variant,
        "poses": list(selected_poses),
        "target_names": [str(name) for name in target_names],
        "observations": observations,
        "results": results,
    }


def language_multi_view_pick_and_place(
    instruction: str,
    output_dir: str | Path | None = None,
    *,
    provider: str = "color_fixture",
    manual_regions: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    camera_width: int = 424,
    camera_height: int = 240,
    seed: int = 7,
    poses: list[str] | tuple[str, ...] = MULTI_VIEW_DEFAULT_POSES,
    max_parallel_vl: int = 4,
    min_accepted_views: int = 1,
    render_gif: bool = True,
    output_path: str | Path | None = None,
    frames: int = 0,
    fps: int = 20,
    width: int = 960,
    height: int = 720,
    show_sites: bool = False,
    show_viewer: bool = False,
    depth_variant: str = "raw",
    language_goal: dict[str, Any] | None = None,
    _place_slot_index: int = 0,
) -> dict[str, Any]:
    pre_parsed = _resolve_language_goal(instruction, language_goal)
    if pre_parsed.get("status") == "ok" and pre_parsed.get("target", {}).get("quantifier") == "all":
        return _language_collection_pick_and_place(
            instruction,
            pre_parsed,
            output_dir,
            provider=provider,
            manual_regions=manual_regions,
            model=model,
            config_path=config_path,
            camera_width=camera_width,
            camera_height=camera_height,
            seed=seed,
            poses=poses,
            max_parallel_vl=max_parallel_vl,
            min_accepted_views=min_accepted_views,
            render_gif=render_gif,
            output_path=output_path,
            frames=frames,
            fps=fps,
            width=width,
            height=height,
            show_sites=show_sites,
            show_viewer=show_viewer,
            depth_variant=depth_variant,
        )
    located = multi_object_vl_locate(
        instruction,
        output_dir,
        provider=provider,
        manual_regions=manual_regions,
        model=model,
        config_path=config_path,
        camera_width=camera_width,
        camera_height=camera_height,
        seed=seed,
        poses=poses,
        max_parallel_vl=max_parallel_vl,
        min_accepted_views=min_accepted_views,
        depth_variant=depth_variant,
        language_goal=pre_parsed,
    )
    parsed = located["language_goal"]
    if located["status"] != "ok":
        return {
            "status": "failed",
            "scene_id": "gripper_multi_object_d435i",
            "skill": "language_multi_view_pick_and_place",
            "language_goal": parsed,
            "perception": located.get("perception"),
            "reason": located.get("reason", "multi-object VL localization failed"),
            "user_acceptance": "pending",
        }
    destination = parsed.get("destination")
    if not destination or destination.get("world_xy_m") is None:
        return {
            "status": "failed",
            "scene_id": "gripper_multi_object_d435i",
            "skill": "language_multi_view_pick_and_place",
            "language_goal": parsed,
            "perception": located["perception"],
            "reason": "pick-and-place requires a destination region in the instruction",
            "user_acceptance": "pending",
        }

    target_surface_world = np.asarray(located["perception"]["fusion"]["fused_target_surface_world_m"], dtype=float)
    target_object = located["target_object"]
    object_half_height = float(located["object_half_height_m"])
    model_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL)
    planned = plan_pick_place_trajectory(
        target_surface_world,
        _destination_place_xy(destination, _place_slot_index),
        object_half_height=object_half_height,
        placement_surface_z=_destination_surface_z(destination),
        source_model=model_path,
    )
    branch_selection = {"mode": "single_default", "selected": {"grasp_yaw_rad": 0.0, "seed": "default"}, "candidates": []}
    actual_frames = pick_place_required_frames(planned, frames=frames, fps=fps)
    result = simulate_pick_place(
        model_path,
        planned,
        object_name=str(target_object["name"]),
        frames=actual_frames,
        fps=fps,
    )
    status = "automatic_precheck_passed" if result.placed else "failed"
    response: dict[str, Any] = {
        "status": status,
        "scene_id": "gripper_multi_object",
        "skill": "language_multi_view_pick_and_place",
        "instruction": instruction,
        "language_goal": parsed,
        "target_object": target_object,
        "model_path": _relative(model_path),
        "perception": located["perception"],
        "target_surface_world_m": _round_vector(target_surface_world),
        "place_center_m": _round_vector(planned.place_center),
        "rrt_connect": {
            "segments": [_planned_segment_summary(segment) for segment in pick_place_segments(planned)],
            "playback_duration_s": round(float(planned.total_playback_duration), 3),
        },
        "ik_branch_selection": branch_selection,
        "metrics": _pick_place_metrics(result),
        "render": {
            "width": width,
            "height": height,
            "frames": actual_frames,
            "fps": fps,
            "show_sites": show_sites,
        },
        "user_acceptance": "pending",
    }
    if show_viewer:
        play_pick_place_viewer(model_path, planned_trajectory=planned, frames=actual_frames, fps=fps)
        response["viewer"] = {"shown": True, "mode": "mujoco_passive", "blocking": True}
    else:
        response["viewer"] = {"shown": False}
    if render_gif:
        output = (ROOT / "outputs" / "pick_place" / "language_multi_view_pick_and_place.gif") if output_path is None else Path(output_path)
        if not output.is_absolute():
            output = ROOT / output
        render_pick_place_gif(
            model_path,
            output,
            planned_trajectory=planned,
            width=width,
            height=height,
            frames=actual_frames,
            fps=fps,
            show_sites=show_sites,
        )
        validate_gif(output, expected_frames=actual_frames, expected_size=(width, height), expected_fps=fps)
        response["gif"] = _relative(output)
    return response


def _language_collection_pick_and_place(
    instruction: str,
    parsed: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    provider: str,
    manual_regions: dict[str, dict[str, Any]] | None,
    model: str | None,
    config_path: str | Path | None,
    camera_width: int,
    camera_height: int,
    seed: int,
    poses: list[str] | tuple[str, ...],
    max_parallel_vl: int,
    min_accepted_views: int,
    render_gif: bool,
    output_path: str | Path | None,
    frames: int,
    fps: int,
    width: int,
    height: int,
    show_sites: bool,
    show_viewer: bool,
    depth_variant: str,
) -> dict[str, Any]:
    destination = parsed.get("destination")
    target_names = list(parsed.get("target", {}).get("object_names") or [])
    if not destination or destination.get("world_xy_m") is None:
        return {
            "status": "failed",
            "scene_id": "gripper_multi_object_d435i",
            "skill": "language_multi_view_pick_and_place",
            "instruction": instruction,
            "language_goal": parsed,
            "reason": "collection pick-and-place requires a destination region in the instruction",
            "user_acceptance": "pending",
        }
    if not target_names:
        return {
            "status": "failed",
            "scene_id": "gripper_multi_object_d435i",
            "skill": "language_multi_view_pick_and_place",
            "instruction": instruction,
            "language_goal": parsed,
            "reason": "collection instruction did not resolve any target objects",
            "user_acceptance": "pending",
        }

    base_output_dir = ROOT / "outputs" / "pick_place" / "collect_to_tray" if output_dir is None else Path(output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = ROOT / base_output_dir

    model_path = write_d435i_multi_object_scene_model(DEFAULT_D435I_MULTI_OBJECT_MODEL)
    subtasks: list[dict[str, Any]] = []
    sequence_items: list[PickPlaceSequenceItem] = []
    used_tray_slots: list[int] = []
    for index, name in enumerate(target_names):
        target_object = _object_by_name(str(name))
        sub_language_goal = _single_object_language_goal(target_object, destination)
        sub_instruction = str(sub_language_goal["instruction"])
        sub_output_dir = base_output_dir / f"{index + 1:02d}_{target_object['name']}"
        located = multi_object_vl_locate(
            sub_instruction,
            sub_output_dir,
            provider=provider,
            manual_regions=manual_regions,
            model=model,
            config_path=config_path,
            camera_width=camera_width,
            camera_height=camera_height,
            seed=seed + index,
            poses=poses,
            max_parallel_vl=max_parallel_vl,
            min_accepted_views=min_accepted_views,
            depth_variant=depth_variant,
            language_goal=sub_language_goal,
        )
        sub_parsed = located.get("language_goal")
        if located.get("status") != "ok":
            return {
                "status": "failed",
                "scene_id": "gripper_multi_object_d435i",
                "skill": "language_multi_view_pick_and_place",
                "mode": "collection_continuous",
                "instruction": instruction,
                "language_goal": parsed,
                "subtasks": subtasks,
                "failed_subtask": {
                    "instruction": sub_instruction,
                    "language_goal": sub_parsed,
                    "perception": located.get("perception"),
                    "reason": located.get("reason", "multi-object VL localization failed"),
                },
                "user_acceptance": "pending",
            }

        sub_destination = sub_parsed.get("destination") if isinstance(sub_parsed, dict) else destination
        target_surface_world = np.asarray(located["perception"]["fusion"]["fused_target_surface_world_m"], dtype=float)
        object_half_height = float(located["object_half_height_m"])
        planned, branch_selection = _plan_collection_pick_place_item(
            model_path,
            sequence_items,
            object_name=str(target_object["name"]),
            target_surface_world=target_surface_world,
            place_xy=_destination_place_xy(sub_destination, index),
            place_xy_candidates=_destination_next_place_xy_candidate(sub_destination, used_tray_slots),
            object_half_height=object_half_height,
            placement_surface_z=_destination_surface_z(sub_destination),
            fps=fps,
        )
        selected_slot = branch_selection.get("selected", {}).get("place_slot_index")
        if isinstance(selected_slot, int) and selected_slot >= 0:
            used_tray_slots.append(selected_slot)
        sequence_items.append(PickPlaceSequenceItem(object_name=str(target_object["name"]), planned_trajectory=planned))
        subtasks.append(
            {
                "instruction": sub_instruction,
                "language_goal": sub_parsed,
                "target_object": target_object,
                "perception": located["perception"],
                "target_surface_world_m": _round_vector(target_surface_world),
                "place_center_m": _round_vector(planned.place_center),
                "rrt_connect": {
                    "segments": [_planned_segment_summary(segment) for segment in pick_place_segments(planned)],
                    "playback_duration_s": round(float(planned.total_playback_duration), 3),
                },
                "ik_branch_selection": branch_selection,
            }
        )

    actual_frames = pick_place_sequence_required_frames(sequence_items, frames=frames, fps=fps)
    result = simulate_pick_place_sequence(model_path, sequence_items, frames=frames, fps=fps)
    for subtask, item_result in zip(subtasks, result.item_results):
        subtask["status"] = "automatic_precheck_passed" if item_result.placed else "failed"
        subtask["metrics"] = _pick_place_metrics(item_result)

    ok_count = sum(1 for item in subtasks if item.get("status") == "automatic_precheck_passed")
    response: dict[str, Any] = {
        "status": "automatic_precheck_passed" if result.placed else "failed",
        "scene_id": "gripper_multi_object",
        "skill": "language_multi_view_pick_and_place",
        "mode": "collection_continuous",
        "instruction": instruction,
        "language_goal": parsed,
        "destination": destination,
        "target_count": len(target_names),
        "successful_count": ok_count,
        "model_path": _relative(model_path),
        "subtasks": subtasks,
        "sequence": {
            "continuous_simulation": True,
            "bridge_seconds": SEQUENCE_BRIDGE_SECONDS,
            "frames": int(result.total_frames),
            "fps": fps,
            "playback_duration_s": round(float(result.total_frames) / float(fps), 3),
        },
        "render": {
            "width": width,
            "height": height,
            "frames": int(result.total_frames),
            "fps": fps,
            "show_sites": show_sites,
        },
        "user_acceptance": "pending",
    }
    if show_viewer:
        play_pick_place_sequence_viewer(model_path, items=sequence_items, frames=actual_frames, fps=fps)
        response["viewer"] = {"shown": True, "mode": "mujoco_passive", "blocking": True}
    else:
        response["viewer"] = {"shown": False}
    if render_gif:
        output = _collection_sequence_gif_path(output_path)
        render_pick_place_sequence_gif(
            model_path,
            output,
            items=sequence_items,
            width=width,
            height=height,
            frames=actual_frames,
            fps=fps,
            show_sites=show_sites,
        )
        validate_gif(output, expected_frames=result.total_frames, expected_size=(width, height), expected_fps=fps)
        response["gif"] = _relative(output)
        response["gif_paths"] = [_relative(output)]
    else:
        response["gif_paths"] = []
    return response

def _plan_collection_pick_place_item(
    model_path: Path,
    sequence_items: list[PickPlaceSequenceItem],
    *,
    object_name: str,
    target_surface_world: np.ndarray,
    place_xy: list[float],
    place_xy_candidates: list[tuple[int, list[float]]] | None = None,
    object_half_height: float,
    placement_surface_z: float,
    fps: int,
    preview_sequence: bool = True,
) -> tuple[Any, dict[str, Any]]:
    previous = sequence_items[-1].planned_trajectory if sequence_items else None
    place_specs = place_xy_candidates or [(-1, place_xy)]
    candidate_specs: list[tuple[float, str, np.ndarray | None]] = []
    for yaw in PICK_PLACE_CANDIDATE_YAWS:
        candidate_specs.append((float(yaw), "default", None))
        if previous is not None:
            candidate_specs.append((float(yaw), "previous_retreat", previous.poses.q_retreat))

    candidates: list[dict[str, Any]] = []
    if len(place_specs) == 1:
        place_slot_index, candidate_place_xy = place_specs[0]
        first_planned = None
        first_summary = None
        for candidate_index, (yaw, seed_name, seed_q) in enumerate(candidate_specs):
            try:
                planned = plan_pick_place_trajectory(
                    target_surface_world,
                    candidate_place_xy,
                    object_half_height=object_half_height,
                    placement_surface_z=placement_surface_z,
                    source_model=model_path,
                    grasp_yaw=yaw,
                    seed_q=seed_q,
                )
                if previous is None:
                    bridge_delta = np.zeros(6, dtype=float)
                    joint_norm = 0.0
                    min_place_spacing = 0.20
                else:
                    bridge_waypoints = sequence_bridge_waypoints(previous.poses.q_retreat, previous, planned)
                    bridge_delta = bridge_waypoints[-1] - bridge_waypoints[0]
                    joint_norm = float(np.linalg.norm(bridge_delta))
                    min_place_spacing = min(
                        float(np.linalg.norm(planned.place_center[:2] - item.planned_trajectory.place_center[:2]))
                        for item in sequence_items
                    )
                candidate_summary = {
                    "index": candidate_index,
                    "place_slot_index": int(place_slot_index),
                    "place_xy_m": [round(float(value), 6) for value in candidate_place_xy],
                    "min_place_spacing_m": round(float(min_place_spacing), 6),
                    "grasp_yaw_rad": round(float(yaw), 6),
                    "seed": seed_name,
                    "placed_in_preview": None,
                    "bridge_delta_deg": [round(float(value), 3) for value in np.rad2deg(bridge_delta)],
                    "bridge_joint_norm_rad": round(joint_norm, 6),
                    "playback_duration_s": round(float(planned.total_playback_duration), 3),
                }
                candidates.append(candidate_summary)
                if not preview_sequence:
                    return planned, {"mode": "sequential_slot_first_feasible_no_preview", "selected": candidate_summary, "candidates": candidates}
                item = PickPlaceSequenceItem(object_name=object_name, planned_trajectory=planned)
                preview_items = [*sequence_items, item]
                sequence_result = simulate_pick_place_sequence(model_path, preview_items, fps=fps, frames=0)
                placed = bool(sequence_result.placed and sequence_result.item_results[-1].placed)
                candidate_summary["placed_in_preview"] = placed
                if first_planned is None:
                    first_planned = planned
                    first_summary = candidate_summary
                if placed:
                    return planned, {"mode": "sequential_slot_first_feasible", "selected": candidate_summary, "candidates": candidates}
            except Exception as exc:
                candidates.append(
                    {
                        "index": candidate_index,
                        "place_slot_index": int(place_slot_index),
                        "place_xy_m": [round(float(value), 6) for value in candidate_place_xy],
                        "grasp_yaw_rad": round(float(yaw), 6),
                        "seed": seed_name,
                        "placed_in_preview": False,
                        "score": None,
                        "reject_reason": str(exc),
                    }
                )
        if first_planned is not None:
            return first_planned, {"mode": "sequential_slot_first_feasible", "selected": first_summary, "candidates": candidates}
        raise RuntimeError(f"No feasible IK branch candidate for {object_name}.")

    feasible_records: list[tuple[float, int, Any, dict[str, Any]]] = []
    candidate_index = 0
    for place_slot_index, candidate_place_xy in place_specs:
        for yaw, seed_name, seed_q in candidate_specs:
            try:
                planned = plan_pick_place_trajectory(
                    target_surface_world,
                    candidate_place_xy,
                    object_half_height=object_half_height,
                    placement_surface_z=placement_surface_z,
                    source_model=model_path,
                    grasp_yaw=yaw,
                    seed_q=seed_q,
                )
                if previous is None:
                    bridge_delta = np.zeros(6, dtype=float)
                    base_delta = 0.0
                    wrist_flip_delta = 0.0
                    wrist_roll_delta = 0.0
                    joint_norm = 0.0
                    min_place_spacing = 0.20
                else:
                    bridge_waypoints = sequence_bridge_waypoints(previous.poses.q_retreat, previous, planned)
                    bridge_delta = bridge_waypoints[-1] - bridge_waypoints[0]
                    base_delta = abs(float(bridge_delta[0]))
                    wrist_flip_delta = abs(float(bridge_delta[4]))
                    wrist_roll_delta = abs(float(bridge_delta[5]))
                    joint_norm = float(np.linalg.norm(bridge_delta))
                    min_place_spacing = min(
                        float(np.linalg.norm(planned.place_center[:2] - item.planned_trajectory.place_center[:2]))
                        for item in sequence_items
                    )
                score = (
                    3.0 * base_delta
                    + 1.8 * wrist_flip_delta
                    + 0.8 * wrist_roll_delta
                    + 0.4 * joint_norm
                    + 0.03 * float(planned.total_playback_duration)
                    - 1.2 * min(min_place_spacing, 0.18)
                )
                candidate_summary = {
                    "index": candidate_index,
                    "place_slot_index": int(place_slot_index),
                    "place_xy_m": [round(float(value), 6) for value in candidate_place_xy],
                    "min_place_spacing_m": round(float(min_place_spacing), 6),
                    "grasp_yaw_rad": round(float(yaw), 6),
                    "seed": seed_name,
                    "placed_in_preview": None,
                    "score": round(float(score), 6),
                    "bridge_delta_deg": [round(float(value), 3) for value in np.rad2deg(bridge_delta)],
                    "bridge_joint_norm_rad": round(joint_norm, 6),
                    "playback_duration_s": round(float(planned.total_playback_duration), 3),
                }
                candidates.append(candidate_summary)
                feasible_records.append((float(score), candidate_index, planned, candidate_summary))
            except Exception as exc:
                candidates.append(
                    {
                        "index": candidate_index,
                        "place_slot_index": int(place_slot_index),
                        "place_xy_m": [round(float(value), 6) for value in candidate_place_xy],
                        "grasp_yaw_rad": round(float(yaw), 6),
                        "seed": seed_name,
                        "placed_in_preview": False,
                        "score": None,
                        "reject_reason": str(exc),
                    }
                )
            candidate_index += 1

    if not feasible_records:
        raise RuntimeError(f"No feasible IK branch candidate for {object_name}.")
    best_score, best_index, best_planned, selected = sorted(feasible_records, key=lambda item: item[0])[0]
    for score, index, planned, candidate_summary in sorted(feasible_records, key=lambda item: item[0]):
        item = PickPlaceSequenceItem(object_name=object_name, planned_trajectory=planned)
        preview_items = [*sequence_items, item]
        sequence_result = simulate_pick_place_sequence(model_path, preview_items, fps=fps, frames=0)
        placed = bool(sequence_result.placed and sequence_result.item_results[-1].placed)
        candidate_summary["placed_in_preview"] = placed
        if placed:
            best_score = score
            best_index = index
            best_planned = planned
            selected = candidate_summary
            break
    return best_planned, {"mode": "ranked_preview_scored_multi_branch", "selected": selected, "candidates": candidates}

def _destination_place_xy(destination: dict[str, Any], slot_index: int = 0) -> list[float]:
    if destination.get("type") == "tray":
        x, y = TRAY_PLACE_SLOTS[slot_index % len(TRAY_PLACE_SLOTS)]
        return [float(x), float(y)]
    return [float(value) for value in destination["world_xy_m"]]


def _destination_place_xy_candidates(destination: dict[str, Any], used_slots: list[int] | None = None) -> list[tuple[int, list[float]]] | None:
    if destination.get("type") != "tray":
        return None
    used = set(used_slots or [])
    candidates = []
    for slot_index, (x, y) in enumerate(TRAY_PLACE_SLOTS):
        if slot_index in used:
            continue
        candidates.append((slot_index, [float(x), float(y)]))
    return candidates or None


def _destination_next_place_xy_candidate(destination: dict[str, Any], used_slots: list[int] | None = None) -> list[tuple[int, list[float]]] | None:
    if destination.get("type") != "tray":
        return None
    used = set(used_slots or [])
    for slot_index, (x, y) in enumerate(TRAY_PLACE_SLOTS):
        if slot_index not in used:
            return [(slot_index, [float(x), float(y)])]
    return None


def _destination_surface_z(destination: dict[str, Any]) -> float:
    return float(TRAY_FLOOR_TOP_Z if destination.get("type") == "tray" else TABLE_TOP_Z)


def _single_object_language_goal(target_object: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    instruction = _single_object_instruction(target_object, destination)
    target = {
        "color": target_object.get("color"),
        "shape": target_object.get("shape"),
        "object_name": target_object.get("name"),
        "object_names": [target_object.get("name")],
        "quantifier": "one",
        "constraints": {
            "color": target_object.get("color"),
            "shape": target_object.get("shape"),
        },
    }
    goal = {
        "status": "ok",
        "instruction": instruction,
        "action": "pick_and_place",
        "target": target,
        "destination": dict(destination),
        "matched_objects": [_candidate_from_object(target_object)],
        "ambiguities": [],
        "warnings": [],
        "scene_id": "gripper_multi_object_d435i",
        "available_objects": object_specs_to_dicts(DEFAULT_MULTI_OBJECT_SPECS),
    }
    goal["vl_prompt"] = _build_goal_vl_prompt(goal)
    return goal

def _single_object_instruction(target_object: dict[str, Any], destination: dict[str, Any]) -> str:
    color = _color_text_cn(str(target_object.get("color", "")))
    shape = _shape_text_cn(str(target_object.get("shape", "")))
    if destination.get("type") == "tray":
        return f"把{color}{shape}放到托盘中"
    region_text = str(destination.get("region_text") or destination.get("region") or "目标区域")
    return f"把{color}{shape}放到{region_text}"


def _color_text_cn(color: str) -> str:
    return {
        "red": "红色",
        "blue": "蓝色",
        "green": "绿色",
        "yellow": "黄色",
        "purple": "紫色",
    }.get(color, color)


def _shape_text_cn(shape: str) -> str:
    return {
        "box": "方块",
        "cylinder": "圆柱体",
    }.get(shape, shape)


def _collection_sequence_gif_path(output_path: str | Path | None) -> Path:
    if output_path is None:
        return ROOT / "outputs" / "pick_place" / "collect_cylinders_to_tray" / "all_cylinders_continuous.gif"
    path = Path(output_path)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() == ".gif":
        return path
    return path / "all_cylinders_continuous.gif"
def _collection_gif_path(output_path: str | Path | None, index: int, object_name: str) -> Path:
    if output_path is None:
        return ROOT / "outputs" / "pick_place" / "collect_to_tray" / f"{index + 1:02d}_{object_name}.gif"
    path = Path(output_path)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() == ".gif":
        return path.with_name(f"{path.stem}_{index + 1:02d}_{object_name}{path.suffix}")
    return path / f"{index + 1:02d}_{object_name}.gif"
def simulate_pick_cube(frames: int = 120, fps: int = 20) -> dict[str, Any]:
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    result = simulate_pick(model_path, frames=frames, fps=fps)
    return {
        "status": "automatic_precheck_passed" if result.lifted else "failed",
        "scene_id": "gripper_pick_cube",
        "object": "grasp_cube",
        "model_path": _relative(model_path),
        "metrics": _pick_metrics(result),
        "user_acceptance": "pending",
    }


def render_pick_cube_gif(
    output_path: str | Path | None = None,
    *,
    frames: int = 160,
    fps: int = 20,
    width: int = 960,
    height: int = 720,
    show_sites: bool = False,
) -> dict[str, Any]:
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    output = DEFAULT_PICK_GIF if output_path is None else Path(output_path)
    if not output.is_absolute():
        output = ROOT / output

    render_pick_gif(
        model_path,
        output,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        show_sites=show_sites,
    )
    validate_gif(output, expected_frames=frames, expected_size=(width, height), expected_fps=fps)
    return {
        "status": "ok",
        "scene_id": "gripper_pick_cube",
        "gif": _relative(output),
        "render": {
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "show_sites": show_sites,
        },
        "user_acceptance": "pending",
    }


def pick_cube(
    *,
    render_gif: bool = True,
    output_path: str | Path | None = None,
    frames: int = 160,
    fps: int = 20,
    width: int = 960,
    height: int = 720,
    show_sites: bool = False,
) -> dict[str, Any]:
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    result = simulate_pick(model_path, frames=frames, fps=fps)
    status = "automatic_precheck_passed" if result.lifted else "failed"
    response: dict[str, Any] = {
        "status": status,
        "scene_id": "gripper_pick_cube",
        "skill": "pick_cube",
        "object": "grasp_cube",
        "model_path": _relative(model_path),
        "metrics": _pick_metrics(result),
        "user_acceptance": "pending",
    }
    if render_gif:
        gif_info = render_pick_cube_gif(
            output_path,
            frames=frames,
            fps=fps,
            width=width,
            height=height,
            show_sites=show_sites,
        )
        response["gif"] = gif_info["gif"]
        response["render"] = gif_info["render"]
    return response


def plan_pick_from_target_3d(
    target_3d: dict[str, Any] | list[float],
    *,
    render_gif: bool = True,
    output_path: str | Path | None = None,
    frames: int = 0,
    fps: int = 20,
    width: int = 960,
    height: int = 720,
    show_sites: bool = False,
) -> dict[str, Any]:
    target_surface_world = _target_3d_center(target_3d)
    planned = plan_pick_trajectory_from_target_3d(target_surface_world)
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    actual_frames = _pick_frames(planned, frames=frames, fps=fps)
    result = simulate_pick(model_path, frames=actual_frames, fps=fps, planned_trajectory=planned)
    response = _planned_pick_response(
        skill="plan_pick_from_target_3d",
        model_path=model_path,
        target_surface_world=target_surface_world,
        planned=planned,
        result=result,
        frames=actual_frames,
        fps=fps,
        width=width,
        height=height,
        show_sites=show_sites,
    )
    if render_gif:
        output = (ROOT / "outputs" / "gripper_pick" / "target_3d_planned_pick_cube.gif") if output_path is None else Path(output_path)
        if not output.is_absolute():
            output = ROOT / output
        render_pick_gif(
            model_path,
            output,
            width=width,
            height=height,
            frames=actual_frames,
            fps=fps,
            show_sites=show_sites,
            planned_trajectory=planned,
        )
        validate_gif(output, expected_frames=actual_frames, expected_size=(width, height), expected_fps=fps)
        response["gif"] = _relative(output)
        response["render"] = {
            "width": width,
            "height": height,
            "frames": actual_frames,
            "fps": fps,
            "show_sites": show_sites,
        }
    return response


def vl_pick_cube(
    prompt: str,
    output_dir: str | Path | None = None,
    *,
    provider: str = "color_fixture",
    manual_region: dict[str, Any] | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    camera_width: int = 424,
    camera_height: int = 240,
    seed: int = 7,
    pose: str = "scan",
    render_gif: bool = True,
    output_path: str | Path | None = None,
    frames: int = 0,
    fps: int = 20,
    width: int = 960,
    height: int = 720,
    show_sites: bool = False,
    depth_variant: str = "raw",
    scene_id: str = "gripper_pick_cube_d435i",
) -> dict[str, Any]:
    located = vl_locate_object_3d(
        prompt,
        output_dir,
        provider=provider,
        manual_region=manual_region,
        model=model,
        config_path=config_path,
        width=camera_width,
        height=camera_height,
        seed=seed,
        pose=pose,
        depth_variant=depth_variant,
    )
    target_surface_world = _target_3d_center(located["target_3d"])
    planned = plan_pick_trajectory_from_target_3d(target_surface_world)
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    actual_frames = _pick_frames(planned, frames=frames, fps=fps)
    result = simulate_pick(model_path, frames=actual_frames, fps=fps, planned_trajectory=planned)
    response = _planned_pick_response(
        skill="vl_pick_cube",
        model_path=model_path,
        target_surface_world=target_surface_world,
        planned=planned,
        result=result,
        frames=actual_frames,
        fps=fps,
        width=width,
        height=height,
        show_sites=show_sites,
    )
    response["perception"] = {
        "provider": provider,
        "prompt": prompt,
        "depth_variant": located["depth_variant"],
        "depth_file": located["depth_file"],
        "observation": located["observation"],
        "region": located["region"],
        "target_3d": located["target_3d"],
    }
    if render_gif:
        output = (ROOT / "outputs" / "end_to_end" / "vl_planned_pick_cube.gif") if output_path is None else Path(output_path)
        if not output.is_absolute():
            output = ROOT / output
        render_pick_gif(
            model_path,
            output,
            width=width,
            height=height,
            frames=actual_frames,
            fps=fps,
            show_sites=show_sites,
            planned_trajectory=planned,
        )
        validate_gif(output, expected_frames=actual_frames, expected_size=(width, height), expected_fps=fps)
        response["gif"] = _relative(output)
        response["render"] = {
            "width": width,
            "height": height,
            "frames": actual_frames,
            "fps": fps,
            "show_sites": show_sites,
        }
    return response


def multi_view_vl_pick_cube(
    prompt: str,
    output_dir: str | Path | None = None,
    *,
    provider: str = "color_fixture",
    manual_regions: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
    config_path: str | Path | None = None,
    camera_width: int = 424,
    camera_height: int = 240,
    seed: int = 7,
    poses: list[str] | tuple[str, ...] = MULTI_VIEW_DEFAULT_POSES,
    max_parallel_vl: int = 4,
    min_accepted_views: int = 1,
    render_gif: bool = True,
    output_path: str | Path | None = None,
    frames: int = 0,
    fps: int = 20,
    width: int = 960,
    height: int = 720,
    show_sites: bool = False,
    depth_variant: str = "raw",
    scene_id: str = "gripper_pick_cube_d435i",
) -> dict[str, Any]:
    located = multi_view_vl_locate_object_3d(
        prompt,
        output_dir,
        provider=provider,
        manual_regions=manual_regions,
        model=model,
        config_path=config_path,
        camera_width=camera_width,
        camera_height=camera_height,
        seed=seed,
        poses=poses,
        max_parallel_vl=max_parallel_vl,
        min_accepted_views=min_accepted_views,
        depth_variant=depth_variant,
    )
    if located["status"] != "ok":
        return {
            "status": "failed",
            "scene_id": scene_id,
            "skill": "multi_view_vl_pick_cube",
            "perception": located,
            "user_acceptance": "pending",
        }

    target_surface_world = np.asarray(located["fusion"]["fused_target_surface_world_m"], dtype=float)
    planned = plan_pick_trajectory_from_target_3d(target_surface_world)
    model_path = write_pick_scene_model(DEFAULT_PICK_MODEL)
    actual_frames = _pick_frames(planned, frames=frames, fps=fps)
    result = simulate_pick(model_path, frames=actual_frames, fps=fps, planned_trajectory=planned)
    response = _planned_pick_response(
        skill="multi_view_vl_pick_cube",
        model_path=model_path,
        target_surface_world=target_surface_world,
        planned=planned,
        result=result,
        frames=actual_frames,
        fps=fps,
        width=width,
        height=height,
        show_sites=show_sites,
    )
    response["perception"] = located
    if render_gif:
        output = (ROOT / "outputs" / "end_to_end" / "multi_view_vl_pick_cube.gif") if output_path is None else Path(output_path)
        if not output.is_absolute():
            output = ROOT / output
        render_pick_gif(
            model_path,
            output,
            width=width,
            height=height,
            frames=actual_frames,
            fps=fps,
            show_sites=show_sites,
            planned_trajectory=planned,
        )
        validate_gif(output, expected_frames=actual_frames, expected_size=(width, height), expected_fps=fps)
        response["gif"] = _relative(output)
        response["render"] = {
            "width": width,
            "height": height,
            "frames": actual_frames,
            "fps": fps,
            "show_sites": show_sites,
        }
    return response



def _object_by_name(name: str) -> dict[str, Any]:
    for item in object_specs_to_dicts(DEFAULT_MULTI_OBJECT_SPECS):
        if item["name"] == name:
            return item
    raise ValueError(f"Unknown multi-object scene object: {name}")


def _object_half_height_m(item: dict[str, Any]) -> float:
    if "half_size_m" in item:
        return float(item["half_size_m"][2])
    if "half_height_m" in item:
        return float(item["half_height_m"])
    raise ValueError(f"Object does not expose a half height: {item}")


def _pick_place_metrics(result) -> dict[str, Any]:
    return {
        "object_name": result.object_name,
        "initial_object_pos_m": _round_vector(result.initial_object_pos),
        "final_object_pos_m": _round_vector(result.final_object_pos),
        "max_object_z_m": round(float(result.max_object_z), 6),
        "place_center_m": _round_vector(result.place_center),
        "final_gripper_qpos_m": _round_vector(result.final_gripper_qpos),
        "lifted": bool(result.lifted),
        "placed": bool(result.placed),
        "automatic_check": "target object is lifted, moved, and ends near the requested place center",
    }
def _pick_metrics(result) -> dict[str, Any]:
    return {
        "initial_cube_pos_m": _round_vector(result.initial_cube_pos),
        "final_cube_pos_m": _round_vector(result.final_cube_pos),
        "max_cube_z_m": round(float(result.max_cube_z), 6),
        "final_gripper_qpos_m": _round_vector(result.final_gripper_qpos),
        "lifted": bool(result.lifted),
        "automatic_check": "cube final z and max z exceed lift thresholds",
    }


def _multi_view_candidate(
    *,
    prompt: str,
    pose: str,
    observation: dict[str, Any],
    output_dir: Path,
    provider: str,
    manual_region: dict[str, Any] | None,
    model: str | None,
    config_path: str | Path | None,
    depth_file_key: str,
    depth_variant: str,
) -> dict[str, Any]:
    region = _locate_region_from_observation(
        prompt=prompt,
        observation=observation,
        output_dir=output_dir,
        provider=provider,
        manual_region=manual_region,
        model=model,
        config_path=config_path,
        overlay_name=f"{pose}_vl_region_overlay.png",
    )
    depth_path = Path(observation["files"][depth_file_key])
    if not depth_path.is_absolute():
        depth_path = ROOT / depth_path
    estimate3d = estimate_vl_region_3d(
        depth_path=depth_path,
        intrinsics=observation["intrinsics"],
        extrinsic_world_to_camera=observation["extrinsic_world_to_camera"],
        region=region,
        foreground_quantile=0.05,
        foreground_margin_m=0.010,
        bbox_expansion=1.25,
        min_world_z_m=TABLE_TOP_Z + 0.004,
    )
    return {
        "pose": pose,
        "accepted": False,
        "reject_reason": "not_scored",
        "observation": observation,
        "depth_variant": depth_variant,
        "depth_file": _relative(depth_path),
        "region": region,
        "target_3d": region3d_to_dict(estimate3d),
    }


def _score_multi_view_candidates(
    candidates: list[dict[str, Any]],
    *,
    min_valid_pixels: int,
    min_surface_z_m: float,
    max_surface_z_m: float,
    max_cluster_radius_m: float,
) -> list[dict[str, Any]]:
    prelim: list[dict[str, Any]] = []
    for candidate in candidates:
        reasons = []
        target = candidate.get("target_3d")
        if not isinstance(target, dict):
            existing_reason = str(candidate.get("reject_reason", ""))
            if existing_reason and existing_reason != "not_scored":
                reasons.append(existing_reason)
            else:
                reasons.append("missing_target_3d")
        else:
            valid_pixels = int(target.get("valid_pixel_count", 0))
            center = np.asarray(target.get("center_world_m", []), dtype=float)
            if center.shape != (3,):
                reasons.append("invalid_target_shape")
            else:
                z = float(center[2])
                if valid_pixels < min_valid_pixels:
                    reasons.append(f"too_few_depth_pixels:{valid_pixels}")
                if z < min_surface_z_m:
                    reasons.append(f"surface_z_too_low:{z:.6f}")
                if z > max_surface_z_m:
                    reasons.append(f"surface_z_too_high:{z:.6f}")
        if reasons:
            candidate["accepted"] = False
            candidate["reject_reason"] = "; ".join(reasons)
        else:
            candidate["accepted"] = True
            candidate["reject_reason"] = ""
            prelim.append(candidate)

    if len(prelim) <= 1:
        for candidate in prelim:
            candidate["cluster_distance_m"] = 0.0
        return candidates

    selected = _dominant_spatial_cluster(prelim, max_cluster_radius_m=max_cluster_radius_m)
    selected_ids = {id(candidate) for candidate in selected}
    fused = np.median(
        np.asarray([candidate["target_3d"]["center_world_m"] for candidate in selected], dtype=float),
        axis=0,
    )
    for candidate in prelim:
        point = np.asarray(candidate["target_3d"]["center_world_m"], dtype=float)
        distance = float(np.linalg.norm(point - fused))
        candidate["cluster_distance_m"] = round(distance, 6)
        if id(candidate) not in selected_ids:
            candidate["accepted"] = False
            candidate["reject_reason"] = f"inconsistent_3d_cluster:{distance:.6f}"
        elif distance > max_cluster_radius_m:
            candidate["accepted"] = False
            candidate["reject_reason"] = f"outside_cluster_radius:{distance:.6f}"
        else:
            candidate["accepted"] = True
            candidate["reject_reason"] = ""
    return candidates


def _dominant_spatial_cluster(
    candidates: list[dict[str, Any]],
    *,
    max_cluster_radius_m: float,
) -> list[dict[str, Any]]:
    points = np.asarray([candidate["target_3d"]["center_world_m"] for candidate in candidates], dtype=float)
    best_indices: list[int] = []
    best_confidence = -1.0
    for index, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        indices = [i for i, distance in enumerate(distances) if float(distance) <= max_cluster_radius_m]
        confidence = float(sum(candidates[i].get("region", {}).get("confidence", 0.0) for i in indices))
        if len(indices) > len(best_indices) or (len(indices) == len(best_indices) and confidence > best_confidence):
            best_indices = indices
            best_confidence = confidence
    return [candidates[index] for index in best_indices]


def _planned_pick_response(
    *,
    skill: str,
    model_path: Path,
    target_surface_world: np.ndarray,
    planned,
    result,
    frames: int,
    fps: int,
    width: int,
    height: int,
    show_sites: bool,
) -> dict[str, Any]:
    status = "automatic_precheck_passed" if result.lifted else "failed"
    return {
        "status": status,
        "scene_id": "gripper_pick_cube",
        "skill": skill,
        "object": "grasp_cube",
        "model_path": _relative(model_path),
        "target_surface_world_m": _round_vector(target_surface_world),
        "planned_cube_center_m": _round_vector(planned.cube_center),
        "grasp_poses": {
            "q_ready": _round_vector(planned.poses.q_ready),
            "q_above": _round_vector(planned.poses.q_above),
            "q_grasp": _round_vector(planned.poses.q_grasp),
            "q_lift": _round_vector(planned.poses.q_lift),
        },
        "rrt_connect": {
            "segments": [_planned_segment_summary(segment) for segment in _planned_segments(planned)],
            "playback_duration_s": round(float(planned.total_playback_duration), 3),
        },
        "metrics": _pick_metrics(result),
        "render": {
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "show_sites": show_sites,
        },
        "user_acceptance": "pending",
    }


def _planned_segments(planned) -> tuple[Any, Any, Any]:
    return (planned.ready_to_above, planned.above_to_grasp, planned.grasp_to_lift)


def _planned_segment_summary(segment) -> dict[str, Any]:
    return {
        "name": segment.name,
        "reason": segment.reason,
        "iterations": int(segment.iterations),
        "raw_waypoints": len(segment.raw_path),
        "waypoints": len(segment.path),
        "duration_s": round(float(segment.trajectory.duration), 3),
    }


def _pick_frames(planned, *, frames: int, fps: int) -> int:
    required = int(np.ceil(planned.total_playback_duration * fps))
    return max(int(frames), required)


def _target_3d_center(target_3d: dict[str, Any] | list[float]) -> np.ndarray:
    if isinstance(target_3d, dict):
        values = target_3d.get("center_world_m")
    else:
        values = target_3d
    target = np.asarray(values, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"target_3d must contain a 3D center_world_m, got shape {target.shape}")
    return target


def _depth_file_key(depth_variant: str) -> str:
    normalized = str(depth_variant).strip().lower()
    if normalized not in DEPTH_VARIANTS:
        raise ValueError(f"Unknown depth_variant: {depth_variant}. Expected one of {DEPTH_VARIANTS}.")
    return f"{normalized}_depth"


def _round_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _read_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _resolve_output_dir(output_dir: str | Path | None, default_name: str) -> Path:
    output = (DEFAULT_D435I_OUTPUT_DIR / default_name) if output_dir is None else Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return output
