#!/usr/bin/env python3
"""Run the RoboTwin harness with a selectable camera render-load profile.

The harness remains the single source of rollout behavior.  L1 selects the
existing RoboTwin L515 camera profile through a temporary task config, keeping
the three camera names and camera collection flags unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS = PROJECT_ROOT / "scripts" / "robotwin_eval_harness.py"
TASK_CONFIG_ROOT = PROJECT_ROOT / "robotwin_repo" / "env_cfg" / "task_config"
BASE_TASK_CONFIG = TASK_CONFIG_ROOT / "demo_clean.yml"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "results"
DEFAULT_PARENT_PYTHON = (
    "/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python"
)
VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
HEARTBEAT_LIMIT_SECONDS = 120

LEVELS = {
    "L0": {
        "task_config": "demo_clean",
        "profile": "D435",
        "width": 320,
        "height": 240,
        "note": "current configuration",
    },
    "L1": {
        "task_config": None,
        "profile": "L515",
        "width": 320,
        "height": 180,
        "note": "reduced input-camera height via existing RoboTwin profile",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch RoboTwin render-load level L0 or L1 around the "
        "unchanged evaluation harness."
    )
    parser.add_argument("--level", choices=tuple(LEVELS), required=True)
    parser.add_argument(
        "--timeout",
        default="3600s",
        help="outer timeout duration passed to timeout (default: 3600s)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="absolute run log path; defaults to results/render_load_<level>_<UTC>.log",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=HEARTBEAT_LIMIT_SECONDS,
        help="fail when the run log is older than this many seconds (default: 120)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="construct and print the command without launching the harness",
    )
    parser.add_argument(
        "harness_args",
        nargs=argparse.REMAINDER,
        help="additional harness arguments, after --",
    )
    return parser.parse_args()


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_log(level: str) -> Path:
    return DEFAULT_LOG_ROOT / f"render_load_{level.lower()}_{_utc_stamp()}.log"


def _make_l1_task_config() -> tuple[str, Path]:
    if not BASE_TASK_CONFIG.is_file():
        raise FileNotFoundError(f"missing base task config: {BASE_TASK_CONFIG}")
    if not TASK_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"missing task config directory: {TASK_CONFIG_ROOT}")

    task_config_name = f"render_load_l1_{os.getpid()}_{int(time.time())}"
    task_config_path = TASK_CONFIG_ROOT / f"{task_config_name}.yml"
    text = BASE_TASK_CONFIG.read_text(encoding="utf-8")
    replacements = (
        ("head_camera_type: D435", "head_camera_type: L515"),
        ("wrist_camera_type: D435", "wrist_camera_type: L515"),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(
                f"expected exactly one {old!r} in {BASE_TASK_CONFIG}, "
                f"found {text.count(old)}"
            )
        text = text.replace(old, new)
    task_config_path.write_text(text, encoding="utf-8")
    return task_config_name, task_config_path


def _ps_snapshot(pid: int) -> tuple[str, str]:
    result = subprocess.run(
        ["ps", "-o", "etime=", "-o", "time=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    fields = result.stdout.strip().split()
    if len(fields) >= 2:
        return fields[0], fields[1]
    return "unknown", "unknown"


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
        return
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _log_watchdog(
    process: subprocess.Popen[bytes],
    log_path: Path,
    stale_seconds: float,
    stop_event: threading.Event,
    triggered: list[str],
) -> None:
    last_cpu_time = "unknown"
    while not stop_event.wait(10):
        if process.poll() is not None:
            return
        try:
            age = time.time() - log_path.stat().st_mtime
        except FileNotFoundError:
            age = stale_seconds + 1
        etime, cpu_time = _ps_snapshot(process.pid)
        if age <= stale_seconds:
            last_cpu_time = cpu_time
            continue
        triggered.append(
            f"watchdog timeout: log_age={age:.1f}s pid={process.pid} "
            f"etime={etime} cpu_time={cpu_time} previous_cpu_time={last_cpu_time}"
        )
        _kill_process_tree(process)
        return


def _build_command(
    level: str,
    task_config: str,
    harness_args: list[str],
) -> list[str]:
    parent_python = os.environ.get("ROBOTWIN_PARENT_PYTHON", DEFAULT_PARENT_PYTHON)
    if not Path(parent_python).is_file():
        parent_python = sys.executable
    extra = list(harness_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return [
        "timeout",
        "--kill-after=30",
        "3600s",
        parent_python,
        str(HARNESS),
        "--task-config",
        task_config,
        *extra,
    ]


def main() -> int:
    args = parse_args()
    level = LEVELS[args.level]
    temporary_config: Path | None = None

    if args.level == "L1":
        task_config, temporary_config = _make_l1_task_config()
    else:
        task_config = str(level["task_config"])

    log_path = args.log or _default_log(args.level)
    if not log_path.is_absolute():
        raise ValueError(f"--log must be absolute, got {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(args.level, task_config, args.harness_args)
    command[2] = args.timeout

    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = VK_ICD
    env["ROBOTWIN_HEARTBEAT_LOG"] = str(log_path)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with log_path.open("ab", buffering=0) as log_file:
            header = (
                f"RENDER_LOAD_LEVEL={args.level}\n"
                f"CAMERA_PROFILE={level['profile']}\n"
                f"CAMERA_SIZE={level['width']}x{level['height']}\n"
                f"EXPECTED_EFFECT={level['note']}\n"
                f"COMMAND={' '.join(command)}\n"
            )
            log_file.write(header.encode("utf-8"))
            if args.dry_run:
                print(header, end="")
                return 0

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            stop_event = threading.Event()
            watchdog_triggered: list[str] = []
            watchdog = threading.Thread(
                target=_log_watchdog,
                args=(
                    process,
                    log_path,
                    args.heartbeat_timeout,
                    stop_event,
                    watchdog_triggered,
                ),
                name="render_load_log_watchdog",
                daemon=True,
            )
            watchdog.start()
            return_code = process.wait()
            stop_event.set()
            watchdog.join(timeout=2)
            if watchdog_triggered:
                log_file.write(
                    ("\n" + watchdog_triggered[0] + "\n").encode("utf-8")
                )
                return 124
            return return_code
    finally:
        if temporary_config is not None:
            try:
                temporary_config.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
