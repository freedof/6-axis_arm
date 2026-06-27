from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMAND_PATH = ROOT / "outputs" / "live_session" / "command.json"
DEFAULT_STATUS_PATH = ROOT / "outputs" / "live_session" / "status.json"
DEFAULT_STDOUT_PATH = ROOT / "outputs" / "live_session" / "live_stdout.log"
DEFAULT_STDERR_PATH = ROOT / "outputs" / "live_session" / "live_stderr.log"
READY_STATUSES = {"waiting", "running"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Start or reuse the CR5 live MuJoCo session.")
    parser.add_argument("--command-path", type=Path, default=DEFAULT_COMMAND_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--provider", default="openrouter_vision")
    parser.add_argument("--model", default="google/gemini-3.5-flash")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--max-parallel-vl", type=int, default=4)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--no-hud", action="store_true", help="Disable the live MuJoCo native text overlay.")
    parser.add_argument("--wait-timeout", type=float, default=20.0)
    parser.add_argument("--new-console", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    command_path = _absolute(args.command_path)
    status_path = _absolute(args.status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_json(status_path)
    if _session_is_reusable(existing, args.provider, args.model, hud_enabled=not args.no_hud):
        _print_status("reused", existing)
        return
    _stop_non_reusable_session(existing)

    start_marker = time.time()
    _truncate_log(DEFAULT_STDOUT_PATH)
    _truncate_log(DEFAULT_STDERR_PATH)
    process = _start_live_session(args, command_path, status_path)
    status = _wait_for_ready(status_path, process, start_marker, args.wait_timeout)
    _print_status("started", status)


def _start_live_session(args: argparse.Namespace, command_path: Path, status_path: Path) -> subprocess.Popen:
    script_path = ROOT / "src" / "sim" / "live_robot_session.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--command-path",
        str(command_path),
        "--status-path",
        str(status_path),
        "--provider",
        str(args.provider),
        "--fps",
        str(args.fps),
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--max-parallel-vl",
        str(args.max_parallel_vl),
        "--hold-seconds",
        str(args.hold_seconds),
    ]
    if args.model:
        cmd.extend(["--model", str(args.model)])
    if args.config_path:
        cmd.extend(["--config-path", str(_absolute(args.config_path))])
    if args.no_hud:
        cmd.append("--no-hud")

    creationflags = 0
    if os.name == "nt":
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
        if args.new_console:
            creationflags |= subprocess.CREATE_NEW_CONSOLE

    stdout_handle = DEFAULT_STDOUT_PATH.open("a", encoding="utf-8")
    stderr_handle = DEFAULT_STDERR_PATH.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
            close_fds=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()


def _wait_for_ready(status_path: Path, process: subprocess.Popen, start_marker: float, wait_timeout: float) -> dict[str, Any]:
    deadline = time.time() + max(0.1, float(wait_timeout))
    latest: dict[str, Any] = {}
    while time.time() < deadline:
        if process.poll() is not None:
            latest = _read_json(status_path)
            raise RuntimeError(f"live session exited early with code {process.returncode}: {latest}")
        latest = _read_json(status_path)
        if latest.get("updated_at", 0.0) >= start_marker and latest.get("status") == "waiting":
            return latest
        if latest.get("updated_at", 0.0) >= start_marker and latest.get("status") == "failed":
            raise RuntimeError(f"live session failed during startup: {latest}")
        time.sleep(0.1)
    raise TimeoutError(f"live session did not reach waiting within {wait_timeout:.1f}s; latest status: {latest}")


def _session_is_reusable(status: dict[str, Any], provider: str, model: str | None, *, hud_enabled: bool) -> bool:
    pid = int(status.get("pid") or 0)
    if not pid or not _pid_is_alive(pid):
        return False
    if status.get("status") not in READY_STATUSES:
        return False
    if status.get("provider") != provider:
        return False
    if (status.get("vl_model") or None) != (model or None):
        return False
    if bool(status.get("hud_enabled", False)) != bool(hud_enabled):
        return False
    if not status.get("tray_memory_path"):
        return False
    return True


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_non_reusable_session(status: dict[str, Any]) -> None:
    pid = int(status.get("pid") or 0) if isinstance(status, dict) else 0
    if not pid or not _pid_is_alive(pid):
        return
    if status.get("status") == "running":
        raise RuntimeError(f"existing live session is running and is not reusable: {status}")
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, check=False)
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return


def _print_status(action: str, status: dict[str, Any]) -> None:
    summary = {
        "action": action,
        "status": status.get("status"),
        "pid": status.get("pid"),
        "provider": status.get("provider"),
        "vl_model": status.get("vl_model"),
        "model_path": status.get("model_path"),
        "command_path": status.get("command_path"),
        "hud_enabled": status.get("hud_enabled"),
        "tray_memory_path": status.get("tray_memory_path"),
        "startup_elapsed_s": status.get("startup_elapsed_s"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _truncate_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8"):
        pass


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
