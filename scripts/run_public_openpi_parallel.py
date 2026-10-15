#!/usr/bin/env python3
"""Run the official RoboTwin task list with bounded parallelism on one GPU.

Each worker is an unchanged robotwin_public_openpi_harness process.  Parallel
workers only overlap independent tasks; task order, seeds, episode limits,
checkpoint, and simulator protocol are unchanged.  Results are written per
task and the manifest is updated by this parent process.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task-config", choices=("demo_clean", "demo_randomized"), required=True)
    # Keep the project-wide diagnostic default aligned with the current
    # 10-trial/task evaluation configuration.  Official 100-trial runs must
    # opt in explicitly with ``--episodes 100``.
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--episode-steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tasks", help="optional comma-separated subset for diagnostics or sharding")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt-prefix", choices=("none", "scene"), default="none")
    p.add_argument("--normalization", choices=("zscore", "quantile"), default="quantile")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--noise-source", choices=("jax", "torch", "internal"), default="torch")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--robotwin-root", type=Path, default=Path(__file__).resolve().parents[1] / "robotwin_repo")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        raise SystemExit("episodes and workers must be positive")
    render_backend = os.environ.get("ROBOTWIN_RENDER_BACKEND", "rt")
    rt_denoiser = os.environ.get("ROBOTWIN_RT_DENOISER", "oidn")
    oidn_runtime = os.environ.get(
        "ROBOTWIN_OIDN_RUNTIME",
        str(Path(__file__).resolve().parents[1] / "third_party/oidn-2.5.1/lib"),
    )
    if render_backend == "rt" and args.workers != 1:
        raise SystemExit("RT evaluation requires --workers 1 on this host")

    args.output_root = args.output_root.expanduser().resolve()
    args.robotwin_root = args.robotwin_root.expanduser().resolve()

    task_list = yaml.safe_load((args.robotwin_root / "env_cfg/eval/all_tasks.yml").read_text())["tasks"]
    if args.tasks:
        requested = [x.strip() for x in args.tasks.split(",") if x.strip()]
        unknown = sorted(set(requested) - set(task_list))
        if unknown:
            raise SystemExit(f"unknown task(s): {unknown}")
        task_list = [x for x in task_list if x in requested]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task_config": args.task_config,
        "episodes_requested_per_task": args.episodes,
        "seed_start": args.seed,
        "episode_steps": args.episode_steps,
        "prompt_prefix": args.prompt_prefix,
        "normalization": args.normalization,
        "policy_seed": args.policy_seed,
        "noise_source": args.noise_source,
        "render_backend": render_backend,
        "rt_denoiser": rt_denoiser,
        "oidn_runtime": oidn_runtime if rt_denoiser == "oidn" else None,
        "workers": args.workers,
        "robotwin_root": str(args.robotwin_root),
        "official_task_list": task_list,
        "tasks": [],
    }
    if args.resume and manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text())
            keys = (
                "checkpoint", "task_config", "episodes_requested_per_task",
                "seed_start", "episode_steps", "prompt_prefix", "normalization", "policy_seed", "noise_source", "render_backend",
                "rt_denoiser",
                "oidn_runtime",
                "robotwin_root",
            )
            if all(old.get(k) == manifest[k] for k in keys):
                manifest["tasks"] = list(old.get("tasks", []))
        except (OSError, json.JSONDecodeError):
            pass

    runner = str(Path(__file__).resolve().with_name("robotwin_public_openpi_harness.py"))
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "OPENPI_DATA_HOME": "/root/autodl-tmp/openpi_cache", "PYTHONUNBUFFERED": "1"})

    pending: list[str] = []
    for task in task_list:
        result_path = args.output_root / task / "result.json"
        entry = next((x for x in manifest["tasks"] if x.get("task") == task), None)
        if args.resume and result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                if result.get("episodes_completed") == args.episodes:
                    if not entry or entry.get("returncode") != 0 or entry.get("episodes_completed") != args.episodes:
                        reports = list(result.get("reports", []))
                        successes = result.get("successes")
                        if successes is None:
                            successes = sum(bool(report.get("success")) for report in reports)
                        recovered = {
                            "task": task,
                            "returncode": 0,
                            "result": str(result_path),
                            "log": str(args.output_root / task / "run.log"),
                            "attempts": 0,
                            "episodes_completed": args.episodes,
                            "successes": successes,
                            "recovered_from_complete_result": True,
                        }
                        manifest["tasks"] = [
                            item for item in manifest["tasks"] if item.get("task") != task
                        ]
                        manifest["tasks"].append(recovered)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append(task)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run_task(task: str) -> dict[str, object]:
        task_dir = args.output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_dir / "result.json"
        log_path = task_dir / "run.log"
        cmd = [
            args.python, runner, "--checkpoint", str(Path(args.checkpoint).resolve()),
            "--task", task, "--task-config", args.task_config,
            "--episodes", str(args.episodes), "--seed", str(args.seed),
            "--prompt-prefix", args.prompt_prefix,
            "--normalization", args.normalization,
            "--policy-seed", str(args.policy_seed),
            "--noise-source", args.noise_source,
            "--episode-steps", str(args.episode_steps), "--output", str(result_path),
            "--robotwin-root", str(args.robotwin_root),
        ]
        if args.resume:
            cmd.append("--resume")
        print(f"START {task} {args.task_config}", flush=True)
        # A simulator-side scene error can terminate one harness process after
        # a valid partial result.  Retry that same task with --resume so the
        # requested accepted-episode target is reached instead of silently
        # freezing a short result into the manifest.  The retry limit prevents a fatal
        # model/configuration error from spinning forever.
        attempts = 0
        last_completed = -1
        proc_returncode = 1
        while True:
            attempts += 1
            with log_path.open("a" if (args.resume or attempts > 1) else "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=args.robotwin_root)
            proc_returncode = proc.returncode
            completed = None
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text())
                    completed = result.get("episodes_completed")
                except Exception:
                    completed = None
            if proc_returncode == 0 and completed == args.episodes:
                break
            if completed == args.episodes:
                break
            if attempts >= 10:
                break
            if proc_returncode != 0 and completed == last_completed:
                break
            last_completed = completed if completed is not None else last_completed
            print(
                f"RETRY {task} attempt={attempts + 1} "
                f"episodes={completed} rc={proc_returncode}",
                flush=True,
            )
            if "--resume" not in cmd:
                cmd.append("--resume")
        entry: dict[str, object] = {"task": task, "returncode": proc_returncode, "result": str(result_path), "log": str(log_path), "attempts": attempts}
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                entry.update({"episodes_completed": result.get("episodes_completed"), "successes": result.get("successes")})
            except Exception as exc:
                entry["result_error"] = repr(exc)
        print(f"DONE {task} rc={proc_returncode} episodes={entry.get('episodes_completed')} attempts={attempts}", flush=True)
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            entry = future.result()
            manifest["tasks"] = [x for x in manifest["tasks"] if x.get("task") != entry["task"]]
            manifest["tasks"].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if len(manifest["tasks"]) == len(task_list) and all(x.get("returncode") == 0 for x in manifest["tasks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
