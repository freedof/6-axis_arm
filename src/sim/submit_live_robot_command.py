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


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomically submit a command to the CR5 live MuJoCo session.")
    parser.add_argument("--command-path", type=Path, default=DEFAULT_COMMAND_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--command-file", type=Path, default=None, help="JSON command file. Defaults to stdin.")
    parser.add_argument("--require-openrouter-reachable", action="store_true")
    parser.add_argument("--skip-provider-preflight", action="store_true")
    parser.add_argument("--wait-consumed", action="store_true", help="Wait until command.json is consumed.")
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    args = parser.parse_args()

    command = _read_command(args.command_file)
    command_path = _absolute(args.command_path)
    status_path = _absolute(args.status_path)
    _ensure_session_waiting(status_path)
    if not args.skip_provider_preflight and (args.require_openrouter_reachable or command.get("provider") == "openrouter_vision"):
        _ensure_openrouter_reachable()
    _write_command_atomic(command_path, command)
    result: dict[str, Any] = {
        "status": "submitted",
        "command_path": str(command_path),
        "instruction": command.get("instruction"),
    }
    if args.wait_consumed:
        result["consumed"] = _wait_consumed(command_path, args.wait_timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def _read_command(command_file: Path | None) -> dict[str, Any]:
    if command_file is None:
        text = sys.stdin.read()
    else:
        text = _absolute(command_file).read_text(encoding="utf-8-sig")
    command = json.loads(text)
    if not isinstance(command, dict):
        raise ValueError("Live robot command must be a JSON object.")
    if not command.get("action"):
        raise ValueError("Live robot command must include an action field.")
    return command


def _ensure_session_waiting(status_path: Path) -> None:
    status = _read_json(status_path)
    if status.get("status") != "waiting":
        raise RuntimeError(f"Live session is not waiting; current status: {status}")


def _ensure_openrouter_reachable() -> None:
    attempts = (
        ["curl.exe", "-I", "--http1.1", "--max-time", "10", "https://openrouter.ai/api/v1"],
        ["curl.exe", "-I", "--ssl-no-revoke", "--http1.1", "--max-time", "10", "https://openrouter.ai/api/v1"],
    )
    errors = []
    for command in attempts:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return
        errors.append((result.stderr or result.stdout or "").strip())
    message = " | ".join(error for error in errors if error)
    if message:
        raise RuntimeError(f"OpenRouter preflight failed; command was not submitted: {message}")


def _write_command_atomic(command_path: Path, command: dict[str, Any]) -> None:
    command_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = command_path.with_name(f"{command_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, command_path)


def _wait_consumed(command_path: Path, timeout_s: float) -> bool:
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        if not command_path.exists():
            return True
        time.sleep(0.1)
    return not command_path.exists()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with _absolute(path).open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
