#!/usr/bin/env python3
"""Summarize the pre-registered five-task checkpoint diagnostic."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


REFERENCE_URL = "https://robotwin-platform.github.io/data/robotwin_leaderboard.json"
TASKS = (
    "lift_pot",
    "place_dual_shoes",
    "place_mouse_pad",
    "place_cans_plasticbox",
    "click_bell",
)


def task_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_run(root: Path) -> dict[str, dict[str, object]]:
    rows = {}
    for task in TASKS:
        path = root / task / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        reports = result.get("reports", [])
        if result.get("episodes_completed") != 10 or len(reports) != 10:
            raise SystemExit(f"incomplete 10-trial diagnostic: {path}")
        successes = sum(bool(report["success"]) for report in reports)
        rows[task] = {
            "successes": successes,
            "trials": 10,
            "percent": 10.0 * successes,
            "source": str(path.resolve()),
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-clean", type=Path, required=True)
    parser.add_argument("--candidate-randomized", type=Path, required=True)
    parser.add_argument("--baseline-clean", type=Path, required=True)
    parser.add_argument("--baseline-randomized", type=Path, required=True)
    parser.add_argument(
        "--candidate-label",
        default="candidate_59999",
        help="stable report key for the candidate checkpoint",
    )
    parser.add_argument(
        "--candidate-warning",
        default=(
            "The candidate checkpoint is independently trained and is not the "
            "RoboTwin Team leaderboard artifact."
        ),
        help="warning recorded next to the candidate result",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with urllib.request.urlopen(REFERENCE_URL, timeout=30) as response:
        board = json.load(response)
    entry = next(item for item in board["entries"] if item["id"] == "cotrain:Pi_05")
    board_indexes = {task_slug(name): index for index, name in enumerate(board["tasks"])}
    official = {
        task: {
            "clean_percent": float(entry["easy"][board_indexes[task]]),
            "randomized_percent": float(entry["hard"][board_indexes[task]]),
        }
        for task in TASKS
    }

    raw_runs = {
        "baseline_29999": {
            "clean": load_run(args.baseline_clean),
            "randomized": load_run(args.baseline_randomized),
        },
        args.candidate_label: {
            "clean": load_run(args.candidate_clean),
            "randomized": load_run(args.candidate_randomized),
        },
    }
    runs = {}
    for name, raw_run in raw_runs.items():
        runs[name] = {}
        for setting in ("clean", "randomized"):
            task_rows = raw_run[setting]
            runs[name][setting] = {
                "mean_percent": sum(
                    float(task_rows[task]["percent"]) for task in TASKS
                )
                / len(TASKS),
                "tasks": task_rows,
            }

    comparison = {}
    baseline_absolute_gaps = []
    candidate_absolute_gaps = []
    for setting, official_key in (
        ("clean", "clean_percent"),
        ("randomized", "randomized_percent"),
    ):
        official_mean = sum(float(official[task][official_key]) for task in TASKS) / len(TASKS)
        baseline_mean = float(runs["baseline_29999"][setting]["mean_percent"])
        candidate_mean = float(runs[args.candidate_label][setting]["mean_percent"])
        baseline_absolute_gap = abs(baseline_mean - official_mean)
        candidate_absolute_gap = abs(candidate_mean - official_mean)
        baseline_absolute_gaps.append(baseline_absolute_gap)
        candidate_absolute_gaps.append(candidate_absolute_gap)
        comparison[setting] = {
            "official_mean_percent": official_mean,
            "baseline_29999_mean_percent": baseline_mean,
            f"{args.candidate_label}_mean_percent": candidate_mean,
            f"{args.candidate_label}_minus_baseline_percentage_points": candidate_mean - baseline_mean,
            f"{args.candidate_label}_minus_official_percentage_points": candidate_mean - official_mean,
            "baseline_absolute_gap_percentage_points": baseline_absolute_gap,
            f"{args.candidate_label}_absolute_gap_percentage_points": candidate_absolute_gap,
            f"{args.candidate_label}_is_closer_to_official": candidate_absolute_gap < baseline_absolute_gap,
        }

    mean_baseline_gap = sum(baseline_absolute_gaps) / len(baseline_absolute_gaps)
    mean_candidate_gap = sum(candidate_absolute_gaps) / len(candidate_absolute_gaps)
    decision = {
        "rule": (
            "Proceed to 50 tasks only if the candidate five-task mean is closer to "
            "the official mean in both settings and its mean absolute gap across "
            "settings is at least 5 percentage points smaller than baseline."
        ),
        "baseline_mean_absolute_gap_percentage_points": mean_baseline_gap,
        f"{args.candidate_label}_mean_absolute_gap_percentage_points": mean_candidate_gap,
        "gap_reduction_percentage_points": mean_baseline_gap - mean_candidate_gap,
        "proceed_to_full_50_task_diagnostic": (
            all(
                bool(comparison[setting][f"{args.candidate_label}_is_closer_to_official"])
                for setting in ("clean", "randomized")
            )
            and mean_baseline_gap - mean_candidate_gap >= 5.0
        ),
    }

    payload = {
        "status": "complete",
        "purpose": "diagnose whether a 59999-step checkpoint explains the leaderboard gap",
        "warning": (
            args.candidate_warning
        ),
        "pre_registered_tasks": list(TASKS),
        "episodes_per_task": 10,
        "official_reference": {
            "url": REFERENCE_URL,
            "trials_per_task": 100,
            "tasks": official,
        },
        "runs": runs,
        "comparison": comparison,
        "pre_registered_full_run_decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
