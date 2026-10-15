#!/usr/bin/env python3
"""Summarize D151 closed-loop JSONL traces without model or simulator imports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={statistics.fmean(values):.6f} "
        f"min={min(values):.6f} p25={percentile(values, 0.25):.6f} "
        f"median={percentile(values, 0.50):.6f} "
        f"p75={percentile(values, 0.75):.6f} "
        f"p95={percentile(values, 0.95):.6f} max={max(values):.6f}"
    )


def read_records(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}")
                grouped[str(record["model_path"])].append(record)
    return grouped


def read_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}")
    return records


def trace_label(path: Path) -> str:
    name = path.name.lower()
    if "official" in name:
        return "official"
    if "peer" in name:
        return "peer"
    return path.name


def mean_field(records: list[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(record[field]) for record in records)


def paired_output(paths: list[Path], first_n: int) -> None:
    """Print cross-trace summaries without changing the legacy output above."""
    traces = [(trace_label(path), path, read_trace(path)) for path in paths]
    print("PAIRED_ARM_NORM_TABLE")
    print(
        "| trace | signal | n | mean | p25 | median | p75 | p95 | min | max |"
        " post/raw |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, _path, records in traces:
        raw_mean = mean_field(records, "raw_action_arm_norm")
        post_mean = mean_field(records, "postprocessed_action_arm_norm")
        ratio = post_mean / raw_mean if raw_mean else float("nan")
        for signal, field in (
            ("raw", "raw_action_arm_norm"),
            ("post", "postprocessed_action_arm_norm"),
            ("executed", "executed_action_arm_norm"),
        ):
            values = [float(record[field]) for record in records]
            print(
                f"| {label} | {signal} | {len(values)} | "
                f"{statistics.fmean(values):.6f} | {percentile(values, 0.25):.6f} | "
                f"{percentile(values, 0.50):.6f} | {percentile(values, 0.75):.6f} | "
                f"{percentile(values, 0.95):.6f} | {min(values):.6f} | "
                f"{max(values):.6f} | {ratio:.6f} |"
            )

    print(f"\nPAIRED_FIRST_N_VS_STEADY first_n={first_n}")
    print("| trace | subset | n | mean | p25 | median | p75 | p95 | min | max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, _path, records in traces:
        for subset_name, subset in (
            ("first_n", records[:first_n]),
            ("steady_state", records[first_n:]),
        ):
            values = [float(record["executed_action_arm_norm"]) for record in subset]
            print(
                f"| {label} | {subset_name} | {len(values)} | "
                f"{statistics.fmean(values):.6f} | {percentile(values, 0.25):.6f} | "
                f"{percentile(values, 0.50):.6f} | {percentile(values, 0.75):.6f} | "
                f"{percentile(values, 0.95):.6f} | {min(values):.6f} | "
                f"{max(values):.6f} |"
            )

    print("\nPER_EPISODE_BREAKDOWN success=not_recorded_in_trace")
    print(
        "| trace | episode | success | n | raw_mean | post_mean | executed_mean |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    for label, _path, records in traces:
        episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            episodes[int(record["episode"])].append(record)
        for episode, episode_records in sorted(episodes.items()):
            print(
                f"| {label} | {episode} | not_recorded | {len(episode_records)} | "
                f"{mean_field(episode_records, 'raw_action_arm_norm'):.6f} | "
                f"{mean_field(episode_records, 'postprocessed_action_arm_norm'):.6f} | "
                f"{mean_field(episode_records, 'executed_action_arm_norm'):.6f} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--first-n", type=int, default=8)
    args = parser.parse_args()
    if args.first_n < 1:
        raise SystemExit("--first-n must be positive")

    for model_path, records in sorted(read_records(args.trace).items()):
        print(f"CHECKPOINT {model_path}")
        sources = sorted({str(record["frame_source"]) for record in records})
        print(f"FRAME_SOURCE values={sources} all_sim={sources == ['sim']}")
        for field in (
            "raw_action_arm_norm",
            "postprocessed_action_arm_norm",
            "executed_action_arm_norm",
        ):
            print(f"{field} DISTRIBUTION {describe([float(r[field]) for r in records])}")

        first = records[: args.first_n]
        steady = records[args.first_n :]
        print(
            f"CHUNK_RAMP first_n={len(first)} "
            f"queue_positions={[int(r['queue_chunk_position']) for r in first]}"
        )
        for label, subset in (("first_n", first), ("steady_state", steady)):
            values = [float(r["executed_action_arm_norm"]) for r in subset]
            positions = [int(r["queue_chunk_position"]) for r in subset]
            print(f"{label} executed_arm_norm {describe(values)}")
            print(f"{label} queue_positions={sorted(set(positions))}")
        print()
    if len(args.trace) >= 2:
        paired_output(args.trace, args.first_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
