from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.vl_region import estimate_region_3d as estimate_vl_region_3d
from src.perception.vl_region import locate_ark_coding_vision_region
from src.perception.vl_region import locate_codex_vision_region
from src.perception.vl_region import locate_manual_region, locate_openai_vision_region, locate_openrouter_vision_region, locate_red_region_fixture
from src.perception.vl_region import region3d_to_dict
from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model
from src.sim.gripper_pick_motion import plan_pick_trajectory_from_target_3d, simulate_pick
from src.sim.gripper_pick_scene import CUBE_HALF_SIZE, DEFAULT_MULTI_OBJECT_MODEL, DEFAULT_MULTI_OBJECT_SPECS, DEFAULT_PICK_MODEL, TABLE_TOP_Z, object_specs_from_config, object_specs_to_dicts, write_multi_object_scene_model, write_pick_scene_model
from src.sim.d435i_model import DEFAULT_D435I_GRIPPER_MODEL, DEFAULT_D435I_MULTI_OBJECT_MODEL, DEFAULT_D435I_PICK_MODEL, write_d435i_multi_object_scene_model, write_d435i_pick_scene_model
from src.sim.render_d435i_preview import DEFAULT_OUTPUT_DIR as DEFAULT_D435I_OUTPUT_DIR
from src.sim.render_d435i_preview import POSE_CHOICES
from src.sim.render_d435i_preview import render_preview as render_d435i_camera_preview
from src.sim.render_gripper_pick_gif import DEFAULT_OUTPUT as DEFAULT_PICK_GIF
from src.sim.render_gripper_pick_gif import render_gif as render_pick_gif
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
        ],
        "validation_policy": "automatic pre-check passed; waiting for user GIF confirmation",
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
        model_path = write_multi_object_scene_model(DEFAULT_MULTI_OBJECT_MODEL)
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
            }
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


def generate_d435i_scene() -> dict[str, Any]:
    model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    return {
        "status": "ok",
        "model_path": _relative(model_path),
        "scene_id": "gripper_pick_cube_d435i",
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
) -> dict[str, Any]:
    if provider not in VL_PROVIDERS:
        raise ValueError(f"Unknown VL provider: {provider}")
    output = _resolve_output_dir(output_dir, "vl_region")
    observation = render_d435i_preview(output, width=width, height=height, seed=seed, pose=pose)
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
) -> dict[str, Any]:
    depth = Path(depth_path)
    if not depth.is_absolute():
        depth = ROOT / depth
    result = estimate_vl_region_3d(
        depth_path=depth,
        intrinsics=intrinsics,
        extrinsic_world_to_camera=extrinsic_world_to_camera,
        region=region,
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
        "scene_id": "gripper_pick_cube_d435i",
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
            "scene_id": "gripper_pick_cube_d435i",
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
            "scene_id": "gripper_pick_cube_d435i",
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
        "scene_id": "gripper_pick_cube_d435i",
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
            "scene_id": "gripper_pick_cube_d435i",
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


def _resolve_output_dir(output_dir: str | Path | None, default_name: str) -> Path:
    output = (DEFAULT_D435I_OUTPUT_DIR / default_name) if output_dir is None else Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return output
