#!/usr/bin/env python3
"""Run the public OpenPI adapter over RoboTwin's official all-task list.

This project's evaluation budget is 10 trials per task. The requested count
is recorded in the output manifest; task names and task configs always come
from `env_cfg/eval/all_tasks.yml`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task-config", choices=("demo_clean", "demo_randomized"), required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--episode-steps", type=int, default=2000)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt-prefix", choices=("none", "scene"), default="none")
    p.add_argument("--normalization", choices=("zscore", "quantile"), default="quantile")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--noise-source", choices=("jax", "torch"), default="torch")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--robotwin-root", type=Path, default=Path(__file__).resolve().parents[1] / "robotwin_repo")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip tasks with a complete existing result.json")
    args = p.parse_args()
    args.output_root = args.output_root.expanduser().resolve()

    task_list = yaml.safe_load((args.robotwin_root / "env_cfg/eval/all_tasks.yml").read_text())["tasks"]
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
        "render_backend": os.environ.get("ROBOTWIN_RENDER_BACKEND", "rt"),
        "rt_denoiser": os.environ.get("ROBOTWIN_RT_DENOISER", "oidn"),
        "oidn_runtime": os.environ.get(
            "ROBOTWIN_OIDN_RUNTIME",
            str(Path(__file__).resolve().parents[1] / "third_party/oidn-2.5.1/lib"),
        ),
        "official_task_list": task_list,
        "tasks": [],
    }
    if args.resume and manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            if (
                previous.get("checkpoint") == manifest["checkpoint"]
                and previous.get("task_config") == manifest["task_config"]
                and previous.get("episodes_requested_per_task") == manifest["episodes_requested_per_task"]
                and previous.get("seed_start") == manifest["seed_start"]
                and previous.get("episode_steps") == manifest["episode_steps"]
                and previous.get("prompt_prefix") == manifest["prompt_prefix"]
                and previous.get("normalization") == manifest["normalization"]
                and previous.get("policy_seed") == manifest["policy_seed"]
                and previous.get("noise_source") == manifest["noise_source"]
                and previous.get("render_backend") == manifest["render_backend"]
                and previous.get("rt_denoiser") == manifest["rt_denoiser"]
                and previous.get("oidn_runtime") == manifest["oidn_runtime"]
            ):
                manifest["tasks"] = list(previous.get("tasks", []))
        except (OSError, json.JSONDecodeError):
            pass
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "OPENPI_DATA_HOME": "/root/autodl-tmp/openpi_cache", "PYTHONUNBUFFERED": "1"})
    runner = str(Path(__file__).resolve().with_name("robotwin_public_openpi_harness.py"))
    for index, task in enumerate(task_list):
        task_dir = args.output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_dir / "result.json"
        log_path = task_dir / "run.log"
        existing = next((item for item in manifest["tasks"] if item.get("task") == task), None)
        if args.resume and existing and existing.get("returncode") == 0 and result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                if (
                    result.get("episodes_requested") == args.episodes
                    and result.get("episodes_completed") == args.episodes
                ):
                    print(f"[{index + 1}/{len(task_list)}] {task} {args.task_config} (resume: complete)", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        cmd = [
            args.python, runner,
            "--checkpoint", str(Path(args.checkpoint).resolve()),
            "--task", task,
            "--task-config", args.task_config,
            "--episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--prompt-prefix", args.prompt_prefix,
            "--normalization", args.normalization,
            "--policy-seed", str(args.policy_seed),
            "--noise-source", args.noise_source,
            "--episode-steps", str(args.episode_steps),
            "--output", str(result_path),
        ]
        if args.resume:
            cmd.append("--resume")
        print(f"[{index + 1}/{len(task_list)}] {task} {args.task_config}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=args.robotwin_root)
        entry = {"task": task, "returncode": proc.returncode, "result": str(result_path), "log": str(log_path)}
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                entry.update({"episodes_completed": result.get("episodes_completed"), "successes": result.get("successes")})
            except Exception as exc:
                entry["result_error"] = repr(exc)
        manifest["tasks"] = [item for item in manifest["tasks"] if item.get("task") != task]
        manifest["tasks"].append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if proc.returncode != 0 and args.stop_on_error:
            return proc.returncode
    return 0 if all(item["returncode"] == 0 for item in manifest["tasks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
