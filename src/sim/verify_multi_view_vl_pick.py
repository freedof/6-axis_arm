from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp_robot import skills


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "end_to_end" / "multi_view_vl_pick"
DEFAULT_GIF = DEFAULT_OUTPUT_DIR / "multi_view_vl_pick_cube.gif"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify multi-view D435i/VL/depth fusion followed by RRT-Connect planned picking."
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Return one tight bounding box around only the target cube block itself. "
            "Exclude the table, robot gripper, shadows, floor, and background. Pick the red block."
        ),
    )
    parser.add_argument(
        "--provider",
        default="color_fixture",
        choices=["color_fixture", "manual_region", "openai_vision", "ark_coding_vision"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--poses", nargs="+", default=list(skills.MULTI_VIEW_DEFAULT_POSES))
    parser.add_argument("--max-parallel-vl", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = skills.multi_view_vl_pick_cube(
        args.prompt,
        _resolve_path(args.output_dir),
        provider=args.provider,
        model=args.model,
        config_path=args.config_path,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        seed=args.seed,
        poses=args.poses,
        max_parallel_vl=args.max_parallel_vl,
        render_gif=True,
        output_path=_resolve_path(args.gif),
        frames=args.frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    perception = result.get("perception", {})
    fusion = perception.get("fusion", {})
    print("provider:", args.provider)
    print("prompt:", args.prompt)
    print("poses:", args.poses)
    print("fusion:", fusion)
    print("candidates:", [_candidate_summary(candidate) for candidate in perception.get("candidates", [])])
    print("status:", result["status"])
    print("gif:", result.get("gif"))
    print("metrics:", result.get("metrics"))
    if result["status"] != "automatic_precheck_passed":
        raise RuntimeError(f"Expected multi-view planned pick to pass automatic pre-check: {result}")


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    target = candidate.get("target_3d", {})
    region = candidate.get("region", {})
    return {
        "pose": candidate.get("pose"),
        "accepted": candidate.get("accepted"),
        "reject_reason": candidate.get("reject_reason"),
        "bbox_xyxy": region.get("bbox_xyxy"),
        "overlay_path": region.get("overlay_path"),
        "center_world_m": target.get("center_world_m"),
        "valid_pixel_count": target.get("valid_pixel_count"),
        "cluster_distance_m": candidate.get("cluster_distance_m"),
    }


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
