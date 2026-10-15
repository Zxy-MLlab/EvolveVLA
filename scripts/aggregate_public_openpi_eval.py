#!/usr/bin/env python3
"""Aggregate per-task public RoboTwin harness results into Stage-0 records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-config", required=True)
    p.add_argument("--robotwin-root", type=Path, default=Path(__file__).resolve().parents[1] / "robotwin_repo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--robotwin-commit", default="ea8b211")
    p.add_argument("--expected-trials-per-task", type=int, default=100)
    args = p.parse_args()
    tasks = yaml.safe_load((args.robotwin_root / "env_cfg/eval/all_tasks.yml").read_text())["tasks"]
    rows: list[dict[str, object]] = []
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for task in tasks:
        path = args.run_root / task / "result.json"
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
        result = json.loads(path.read_text())
        if result.get("task") != task:
            raise SystemExit(f"task metadata mismatch: {path}")
        if result.get("task_config") != args.task_config:
            raise SystemExit(f"task_config metadata mismatch: {path}")
        if result.get("episodes_requested") != args.expected_trials_per_task:
            raise SystemExit(f"episodes_requested mismatch: {path}")
        reports = list(result.get("reports", []))
        if result.get("episodes_completed") != len(reports):
            raise SystemExit(f"inconsistent result: {path}")
        n = len(reports)
        seeds = [int(x["seed"]) for x in reports]
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"duplicate accepted seed(s): {path}")
        for report in reports:
            steps = int(report.get("steps", 0))
            if not 1 <= steps <= 1200:
                raise SystemExit(f"invalid episode length in {path}: {steps}")
            extrema = (report.get("action_min"), report.get("action_max"))
            if any(value is None or not math.isfinite(float(value)) for value in extrema):
                raise SystemExit(f"non-finite/missing action extrema in {path}")
        s = sum(bool(x.get("success")) for x in reports)
        row = {
            "task": task, "setting": args.task_config, "num_trials": n,
            "num_success": s, "success_rate": (s / n if n else None),
            "average_episode_length": (sum(int(x.get("steps", 0)) for x in reports) / n if n else None),
            "seeds": seeds, "source": str(path),
        }
        rows.append(row)
        for report in reports:
            record = {"task": task, "setting": args.task_config, "checkpoint": args.checkpoint, "robotwin_commit": args.robotwin_commit, **report}
            (successes if report.get("success") else failures).append(record)
    total = sum(int(x["num_trials"]) for x in rows)
    good = sum(int(x["num_success"]) for x in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_task.json").write_text(json.dumps(rows, indent=2) + "\n")
    complete = len(rows) == 50 and all(x["num_trials"] == args.expected_trials_per_task for x in rows)
    if complete and args.expected_trials_per_task == 100:
        status = "official_50_task_complete"
    elif complete:
        status = f"50_task_complete_{args.expected_trials_per_task}_trials_nonofficial"
    else:
        status = "incomplete"
    summary = {
        "status": status,
        "checkpoint": args.checkpoint, "robotwin_commit": args.robotwin_commit,
        "task_config": args.task_config, "tasks": len(rows), "trials": total,
        "successes": good, "success_rate": (good / total if total else None),
        "task_average_success_rate": (sum(float(x["success_rate"]) for x in rows) / len(rows) if rows else None),
        "episode_steps": 1200,
        "episodes_per_task": args.expected_trials_per_task,
        "seed_protocol": (
            "start at seed 1; official expert/scene validation rejects invalid seeds; "
            f"continue until {args.expected_trials_per_task} accepted trials per task"
        ),
        "protocol": (
            "official setup_demo/play_once/check_success; OpenPI 32-step chunks; "
            + ("official 100-trial evaluation" if args.expected_trials_per_task == 100
               else f"diagnostic {args.expected_trials_per_task}-trial setting (not directly comparable to official reference)")
        ),
        "per_task": "per_task.json",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "success_episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in successes))
    (args.output_dir / "failure_episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in failures))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
