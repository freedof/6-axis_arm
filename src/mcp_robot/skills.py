from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.vl_region import estimate_region_3d as estimate_vl_region_3d
from src.perception.vl_region import locate_red_region_fixture, region3d_to_dict
from src.sim.gripper_model import DEFAULT_GRIPPER_MODEL, write_gripper_model
from src.sim.gripper_pick_motion import simulate_pick
from src.sim.gripper_pick_scene import DEFAULT_PICK_MODEL, write_pick_scene_model
from src.sim.d435i_model import DEFAULT_D435I_GRIPPER_MODEL, DEFAULT_D435I_PICK_MODEL, write_d435i_pick_scene_model
from src.sim.render_d435i_preview import DEFAULT_OUTPUT_DIR as DEFAULT_D435I_OUTPUT_DIR
from src.sim.render_d435i_preview import render_preview as render_d435i_camera_preview
from src.sim.render_gripper_pick_gif import DEFAULT_OUTPUT as DEFAULT_PICK_GIF
from src.sim.render_gripper_pick_gif import render_gif as render_pick_gif
from src.sim.verify_render_gifs import validate_gif


SCENES: dict[str, dict[str, Any]] = {
    "cr5_simplified": {
        "name": "CR5 simplified baseline",
        "model_path": ROOT / "assets" / "dobot_cr5" / "mjcf" / "cr5_simplified.xml",
        "description": "6-axis CR5 baseline model without a gripper.",
        "dynamic_objects": [],
    },
    "cr5_with_gripper": {
        "name": "CR5 with simplified parallel gripper",
        "model_path": DEFAULT_GRIPPER_MODEL,
        "description": "Generated 8-actuator CR5 model with a simple parallel gripper.",
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
}


def get_robot_capabilities() -> dict[str, Any]:
    return {
        "robot": "Dobot CR5 simplified",
        "simulator": "MuJoCo",
        "models": {
            "baseline": _relative(SCENES["cr5_simplified"]["model_path"]),
            "gripper": _relative(DEFAULT_GRIPPER_MODEL),
            "pick_scene": _relative(DEFAULT_PICK_MODEL),
            "gripper_d435i": _relative(DEFAULT_D435I_GRIPPER_MODEL),
            "pick_scene_d435i": _relative(DEFAULT_D435I_PICK_MODEL),
        },
        "actuation": {
            "baseline_dof": 6,
            "gripper_model_dof": 8,
            "gripper_type": "simplified two-finger parallel gripper",
        },
        "available_skills": [
            "generate_gripper_model",
            "generate_pick_scene",
            "simulate_pick_cube",
            "render_pick_cube_gif",
            "pick_cube",
            "generate_d435i_scene",
            "render_d435i_preview",
            "vl_locate_object_region",
            "vl_locate_object_3d",
            "estimate_region_3d",
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
    if scene_id == "gripper_pick_cube_d435i":
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
) -> dict[str, Any]:
    model_path = write_d435i_pick_scene_model(DEFAULT_D435I_PICK_MODEL)
    output = DEFAULT_D435I_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    result = render_d435i_camera_preview(model_path, output, width=width, height=height, seed=seed, pose=pose)
    return {
        "status": "ok",
        "scene_id": "gripper_pick_cube_d435i",
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
    width: int = 424,
    height: int = 240,
    seed: int = 7,
    pose: str = "scan",
) -> dict[str, Any]:
    if provider != "color_fixture":
        raise ValueError("Only provider='color_fixture' is available locally; real VL providers should return the same region schema.")
    output = _resolve_output_dir(output_dir, "vl_region")
    observation = render_d435i_preview(output, width=width, height=height, seed=seed, pose=pose)
    rgb_path = ROOT / observation["files"]["rgb"]
    overlay_path = output / "vl_region_overlay.png"
    region = locate_red_region_fixture(rgb_path, prompt=prompt, output_path=overlay_path)
    region["overlay_path"] = _relative(Path(region["overlay_path"])) if region.get("overlay_path") else None
    return {
        "status": "ok",
        "provider": provider,
        "provider_note": "color_fixture is a deterministic stand-in for validating the VL-to-depth interface; replace it with a real VL model later.",
        "prompt": prompt,
        "observation": observation,
        "region": region,
    }


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
    width: int = 424,
    height: int = 240,
    seed: int = 7,
    pose: str = "scan",
) -> dict[str, Any]:
    located = vl_locate_object_region(
        prompt,
        output_dir,
        provider=provider,
        width=width,
        height=height,
        seed=seed,
        pose=pose,
    )
    observation = located["observation"]
    estimate = estimate_region_3d(
        region=located["region"],
        depth_path=observation["files"]["raw_depth"],
        intrinsics=observation["intrinsics"],
        extrinsic_world_to_camera=observation["extrinsic_world_to_camera"],
    )
    return {
        "status": "ok",
        "scene_id": "gripper_pick_cube_d435i",
        "prompt": prompt,
        "provider": provider,
        "observation": observation,
        "region": located["region"],
        "target_3d": estimate["target_3d"],
        "next_step": "Use target_3d as input for grasp-pose generation; it is not yet a complete pick execution.",
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


def _pick_metrics(result) -> dict[str, Any]:
    return {
        "initial_cube_pos_m": _round_vector(result.initial_cube_pos),
        "final_cube_pos_m": _round_vector(result.final_cube_pos),
        "max_cube_z_m": round(float(result.max_cube_z), 6),
        "final_gripper_qpos_m": _round_vector(result.final_gripper_qpos),
        "lifted": bool(result.lifted),
        "automatic_check": "cube final z and max z exceed lift thresholds",
    }


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
