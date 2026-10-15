#!/usr/bin/env python3
"""Parse and report pi05 competence JSON files.

Usage:
    python tools/eval_results_report.py RESULTS.json [MORE_RESULTS.json ...]

The input files use the pi0.5 competence schema with a top-level ``reports``
list. Each report contains a task name, rollout and success counts, and a
``results`` list of episode records. The CLI prints file metadata, one row
per task, and an aggregate across all input files. It uses only the Python
standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """An input file is missing, malformed, or has an unsupported schema."""


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError(f"{path}: file does not exist")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise InputError(f"{path}: could not read file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{path}: malformed JSON at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise InputError(f"{path}: top-level JSON value must be an object")
    if not isinstance(payload.get("reports"), list):
        raise InputError(f"{path}: top-level 'reports' must be a list")
    return payload


def _as_int(value: Any, field: str, path: Path, task: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{path}: task {task!r} has invalid {field}: {value!r}") from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "success", "succeeded"}
    return bool(value)


def _format_rate(successes: int, rollouts: int, reported: Any = None) -> str:
    if reported is None:
        fraction = successes / rollouts if rollouts else 0.0
    else:
        try:
            fraction = float(reported)
        except (TypeError, ValueError):
            fraction = successes / rollouts if rollouts else 0.0
        if fraction > 1.0:
            fraction /= 100.0
    return f"{fraction * 100:.1f}%"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "-"
    return str(value)


def _provisional_notes(payload: dict[str, Any], reports: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    if "provisional" in payload:
        notes.append(f"provisional={_format_value(payload['provisional'])}")
    if "note" in payload:
        notes.append(f"note={_format_value(payload['note'])}")
    for report in reports:
        task = str(report.get("task", "<unnamed task>"))
        if "provisional" in report:
            notes.append(f"{task}: provisional={_format_value(report['provisional'])}")
        if "note" in report:
            notes.append(f"{task}: note={_format_value(report['note'])}")
    return notes


def _task_row(
    report: dict[str, Any], path: Path, index: int
) -> tuple[str, int, int, str, str]:
    task = str(report.get("task", "")).strip()
    if not task:
        task = f"<unnamed task {index}>"

    results = report.get("results", [])
    if results is None:
        results = []
    if not isinstance(results, list):
        raise InputError(f"{path}: task {task!r} has non-list 'results'")

    observed_successes = 0
    steps: list[str] = []
    errors: list[str] = []
    for episode_index, episode in enumerate(results, start=1):
        if not isinstance(episode, dict):
            errors.append(f"episode {episode_index}: invalid entry")
            steps.append("error")
            continue
        if episode.get("status") == "error":
            errors.append(str(episode.get("error", "unspecified error")))
            steps.append("error")
            continue
        if _as_bool(episode.get("success", False)):
            observed_successes += 1
        if "steps" in episode and episode["steps"] is not None:
            steps.append(str(episode["steps"]))

    rollouts_value = report.get("rollouts", len(results))
    successes_value = report.get("successes", observed_successes)
    rollouts = _as_int(rollouts_value, "rollouts", path, task)
    successes = _as_int(successes_value, "successes", path, task)

    if report.get("status") == "error":
        errors.append(str(report.get("error", "unspecified error")))
    elif "error" in report and report["error"]:
        errors.append(str(report["error"]))

    steps_text = "[" + ", ".join(steps) + "]" if steps else "-"
    error_text = "none" if not errors else " | ".join(errors)
    rate = _format_rate(successes, rollouts, report.get("success_rate"))
    return task, successes, rollouts, rate, steps_text + "\t" + error_text


def _print_file(path: Path, payload: dict[str, Any]) -> tuple[int, int]:
    reports = payload["reports"]
    if not reports:
        raise InputError(f"{path}: 'reports' is empty")
    if not all(isinstance(report, dict) for report in reports):
        raise InputError(f"{path}: every entry in 'reports' must be an object")

    metadata = [
        f"obs_size={_format_value(payload.get('obs_size'))}",
        f"split={_format_value(payload.get('split'))}",
    ]
    for key in ("torch", "cuda", "device"):
        if key in payload:
            metadata.append(f"{key}={_format_value(payload[key])}")

    print(f"File: {path}")
    print("  " + " | ".join(metadata))
    for note in _provisional_notes(payload, reports):
        print(f"  PROVISIONAL: {note}")
    print("  task | successes/n | rate | steps | error status")

    total_successes = 0
    total_rollouts = 0
    for index, report in enumerate(reports, start=1):
        task, successes, rollouts, rate, detail = _task_row(report, path, index)
        steps_text, error_text = detail.split("\t", 1)
        print(f"  {task} | {successes}/{rollouts} | {rate} | {steps_text} | {error_text}")
        total_successes += successes
        total_rollouts += rollouts
    return total_successes, total_rollouts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print per-task and aggregate reports for pi0.5 competence JSON files."
    )
    parser.add_argument("files", nargs="+", type=Path, help="competence JSON file(s)")
    args = parser.parse_args(argv)

    aggregate_successes = 0
    aggregate_rollouts = 0
    for path in args.files:
        try:
            payload = _load_payload(path)
            successes, rollouts = _print_file(path, payload)
        except InputError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        aggregate_successes += successes
        aggregate_rollouts += rollouts

    aggregate_rate = _format_rate(aggregate_successes, aggregate_rollouts)
    print(
        f"Aggregate: {aggregate_successes}/{aggregate_rollouts} "
        f"= {aggregate_rate}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
