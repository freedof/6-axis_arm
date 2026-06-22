from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp_robot import skills


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pick_place" / "multi_object_vl_pick_place"
DEFAULT_GIF = DEFAULT_OUTPUT_DIR / "blue_cube_right_pick_place.gif"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify language -> multi-object multi-view VL -> depth -> RRT-Connect pick-and-place."
    )
    parser.add_argument("--instruction", default="把蓝色方块放到桌面右侧")
    parser.add_argument("--provider", default="color_fixture", choices=list(skills.VL_PROVIDERS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--poses", nargs="+", default=list(skills.MULTI_VIEW_DEFAULT_POSES))
    parser.add_argument("--max-parallel-vl", type=int, default=4)
    parser.add_argument("--depth-variant", default="raw", choices=list(skills.DEPTH_VARIANTS))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    result = skills.language_multi_view_pick_and_place(
        args.instruction,
        _resolve_path(args.output_dir),
        provider=args.provider,
        model=args.model,
        config_path=args.config_path,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        seed=args.seed,
        poses=args.poses,
        max_parallel_vl=args.max_parallel_vl,
        depth_variant=args.depth_variant,
        render_gif=not args.no_gif,
        output_path=_resolve_path(args.gif),
        frames=args.frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    perception = result.get("perception", {})
    print("instruction:", args.instruction)
    print("provider:", args.provider)
    print("language_goal:", result.get("language_goal"))
    print("fusion:", perception.get("fusion"))
    print("target_object:", result.get("target_object"))
    print("place_center_m:", result.get("place_center_m"))
    print("metrics:", result.get("metrics"))
    print("gif:", result.get("gif"))
    print("status:", result["status"])
    if result["status"] != "automatic_precheck_passed":
        raise RuntimeError(f"Expected pick-and-place to pass automatic pre-check: {result}")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
