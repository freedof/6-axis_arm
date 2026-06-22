from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp_robot import skills


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pick_place" / "collect_cylinders_to_tray"
DEFAULT_GIF_DIR = DEFAULT_OUTPUT_DIR / "gifs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify language -> multi-object VL -> pick all cylinders -> place into tray."
    )
    parser.add_argument("--instruction", default="把所有圆柱体夹到托盘中")
    parser.add_argument("--provider", default="color_fixture", choices=list(skills.VL_PROVIDERS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gif-dir", type=Path, default=DEFAULT_GIF_DIR)
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
        output_path=_resolve_path(args.gif_dir),
        frames=args.frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    print("instruction:", args.instruction)
    print("provider:", args.provider)
    print("language_goal:", result.get("language_goal"))
    print("target_count:", result.get("target_count"))
    print("successful_count:", result.get("successful_count"))
    for index, subtask in enumerate(result.get("subtasks", []), start=1):
        print(f"subtask_{index}_object:", subtask.get("target_object", {}).get("name"))
        print(f"subtask_{index}_status:", subtask.get("status"))
        print(f"subtask_{index}_place_center_m:", subtask.get("place_center_m"))
        print(f"subtask_{index}_metrics:", subtask.get("metrics"))
        print(f"subtask_{index}_gif:", subtask.get("gif"))
    print("gif_paths:", result.get("gif_paths"))
    print("status:", result["status"])
    if result["status"] != "automatic_precheck_passed":
        raise RuntimeError(f"Expected collection pick-and-place to pass automatic pre-check: {result}")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
