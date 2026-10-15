#!/usr/bin/env python3
"""Validate and summarize the 10-trial official RoboTwin pi0.5 sweeps."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import urllib.request
from pathlib import Path

import yaml


REFERENCE_URL = "https://robotwin-platform.github.io/data/robotwin_leaderboard.json"
OFFICIAL_ENTRY = "cotrain:Pi_05"
PUBLIC_ARCHIVE_SHA256 = "957b07d4319746c70ed5173c2b0ae11b96872410d139931de603d9b4c974ee1c"


def task_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def episode_records(run_root: Path, setting: str, tasks: list[str]):
    for task in tasks:
        source = run_root / task / "result.json"
        result = json.loads(source.read_text(encoding="utf-8"))
        for report in result["reports"]:
            yield {
                "task": task,
                "setting": setting,
                **report,
                "source": str(source.resolve()),
            }


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (math.nan, math.nan)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (center - margin, center + margin)


def stratified_reference_sampling_sd(
    tasks: list[str],
    reference: dict[str, dict[str, float]],
    reference_key: str,
    trials_per_task: int,
) -> float:
    """SD of the suite mean if the leaderboard task rates were ground truth.

    RoboTwin reports a macro average over tasks.  With the same number of
    Bernoulli trials for every task, this is numerically equal to the pooled
    episode rate, but its sampling variance should retain the individual task
    probabilities instead of replacing them with one pooled probability.
    """
    task_count = len(tasks)
    variance = sum(
        (reference[task][reference_key] / 100.0)
        * (1.0 - reference[task][reference_key] / 100.0)
        / trials_per_task
        for task in tasks
    ) / (task_count * task_count)
    return math.sqrt(variance)


def pearson(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return math.nan
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if denominator == 0:
        return math.nan
    return sum(a * b for a, b in zip(centered_a, centered_b, strict=True)) / denominator


def load_reference(path: Path | None) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    if path is None:
        with urllib.request.urlopen(REFERENCE_URL, timeout=30) as response:
            payload = json.load(response)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    entry = next(item for item in payload["entries"] if item["id"] == OFFICIAL_ENTRY)
    by_task = {
        task_slug(task): {"clean": float(clean), "randomized": float(randomized)}
        for task, clean, randomized in zip(payload["tasks"], entry["easy"], entry["hard"], strict=True)
    }
    metadata = {
        "url": REFERENCE_URL,
        "board_updated": payload["board"]["updated"],
        "entry": OFFICIAL_ENTRY,
        "listed_at": entry["listed_at"],
        "clean_percent": float(entry["easy_mean"]),
        "randomized_percent": float(entry["hard_mean"]),
        "trials_per_task": 100,
    }
    return metadata, by_task


def aggregate_setting(
    run_root: Path,
    task_config: str,
    tasks: list[str],
    reference: dict[str, dict[str, float]],
    reference_key: str,
    expected_trials: int,
    noise_source: str = "jax",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pending_sidecars = sorted(run_root.glob("*/result.json.pending.json"))
    if pending_sidecars:
        raise SystemExit(
            "unfinished pending episode sidecars: "
            + ", ".join(str(path) for path in pending_sidecars)
        )
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_entries = {
        str(entry.get("task")): entry for entry in manifest.get("tasks", [])
    }
    if set(manifest_entries) != set(tasks):
        raise SystemExit(
            f"manifest task set mismatch in {manifest_path}: "
            f"expected {len(tasks)}, got {len(manifest_entries)}"
        )
    for task, entry in manifest_entries.items():
        if entry.get("returncode") != 0 or entry.get("episodes_completed") != expected_trials:
            raise SystemExit(f"incomplete/failed manifest entry for {task}: {entry}")

    rows: list[dict[str, object]] = []
    common: dict[str, object] | None = None
    expected = {
        "checkpoint_format": "openpi_orbax",
        "action_horizon": 50,
        "prompt_prefix": "none",
        "normalization": "quantile",
        "policy_seed": 0,
        "noise_source": noise_source,
        "render_backend": "rt",
        "rt_denoiser": "oidn",
    }
    for task in tasks:
        path = run_root / task / "result.json"
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("task") != task or result.get("task_config") != task_config:
            raise SystemExit(f"task metadata mismatch: {path}")
        if result.get("episodes_requested") != expected_trials:
            raise SystemExit(f"trial count mismatch: {path}")
        for key, value in expected.items():
            if result.get(key) != value:
                raise SystemExit(f"{key} mismatch in {path}: {result.get(key)!r} != {value!r}")
        reports = list(result.get("reports", []))
        if result.get("episodes_completed") != expected_trials or len(reports) != expected_trials:
            raise SystemExit(f"incomplete result: {path}")
        seeds = [int(report["seed"]) for report in reports]
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"duplicate accepted seed in {path}")
        if seeds != sorted(seeds) or seeds[0] < 100000:
            raise SystemExit(f"invalid accepted seed stream in {path}: {seeds}")
        if result.get("instruction_candidate_count") != 100:
            raise SystemExit(f"instruction candidate count mismatch in {path}")
        for report in reports:
            steps = int(report["steps"])
            limit = int(report["step_limit"])
            if not 1 <= steps <= limit:
                raise SystemExit(f"invalid episode length in {path}: {steps}/{limit}")
            if not all(math.isfinite(float(report[key])) for key in ("action_min", "action_max")):
                raise SystemExit(f"non-finite action range in {path}")
        successes = sum(bool(report["success"]) for report in reports)
        rate = successes / expected_trials
        official_percent = reference[task][reference_key]
        rows.append(
            {
                "task": task,
                "trials": expected_trials,
                "successes": successes,
                "success_rate": rate,
                "official_100_trial_percent": official_percent,
                "difference_percentage_points": 100.0 * rate - official_percent,
                "seeds": seeds,
                "average_episode_steps": sum(int(report["steps"]) for report in reports) / expected_trials,
                "source": str(path.resolve()),
            }
        )
        current = {key: result.get(key) for key in (*expected, "checkpoint", "episode_steps")}
        common = current if common is None else common
        if current != common:
            raise SystemExit(f"inference configuration differs across tasks: {path}")

    successes = sum(int(row["successes"]) for row in rows)
    trials = len(rows) * expected_trials
    low, high = wilson(successes, trials)
    official_rate = sum(reference[task][reference_key] for task in tasks) / (100.0 * len(tasks))
    reference_sampling_sd = stratified_reference_sampling_sd(
        tasks, reference, reference_key, expected_trials
    )
    observed_rate = successes / trials
    observed_task_rates = [100.0 * float(row["success_rate"]) for row in rows]
    official_task_rates = [float(row["official_100_trial_percent"]) for row in rows]
    task_differences = [
        observed - official
        for observed, official in zip(observed_task_rates, official_task_rates, strict=True)
    ]
    summary = {
        "status": "complete",
        "task_config": task_config,
        "tasks": len(rows),
        "episodes_per_task": expected_trials,
        "trials": trials,
        "successes": successes,
        "success_rate": observed_rate,
        "wilson_95_percent": [low, high],
        "official_100_trial_success_rate": official_rate,
        "difference_percentage_points": 100.0 * (observed_rate - official_rate),
        "reference_rate_sampling_sd_at_requested_trials": reference_sampling_sd,
        "difference_in_reference_sampling_sd": (
            (observed_rate - official_rate) / reference_sampling_sd
            if reference_sampling_sd > 0
            else math.nan
        ),
        "mean_absolute_task_difference_percentage_points": sum(abs(float(row["difference_percentage_points"])) for row in rows) / len(rows),
        "root_mean_square_task_difference_percentage_points": math.sqrt(
            sum(value * value for value in task_differences) / len(task_differences)
        ),
        "per_task_rate_pearson_correlation": pearson(
            observed_task_rates, official_task_rates
        ),
        "inference": common,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--randomized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--noise-source",
        choices=("jax", "internal"),
        default="jax",
        help="noise path recorded by the harness; internal matches official policy.infer()",
    )
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "robotwin_repo",
    )
    args = parser.parse_args()
    tasks = list(yaml.safe_load((args.robotwin_root / "env_cfg/eval/all_tasks.yml").read_text())["tasks"])
    reference_metadata, reference = load_reference(args.reference_json)
    missing = sorted(set(tasks) - set(reference))
    if missing:
        raise SystemExit(f"official reference is missing tasks: {missing}")

    clean, clean_rows = aggregate_setting(
        args.clean_root, "demo_clean", tasks, reference, "clean", args.episodes,
        args.noise_source,
    )
    randomized, randomized_rows = aggregate_setting(
        args.randomized_root,
        "demo_randomized",
        tasks,
        reference,
        "randomized",
        args.episodes,
        args.noise_source,
    )
    if clean["inference"] != randomized["inference"]:
        raise SystemExit("Clean and Randomized inference configurations differ")
    checkpoint = Path(str(clean["inference"]["checkpoint"])).resolve()
    deploy = yaml.safe_load(
        (args.robotwin_root / "XPolicyLab/policy/Pi_05/deploy.yml").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_step = int(checkpoint.name) if checkpoint.name.isdigit() else None
    deploy_step = int(deploy["checkpoint_num"])
    deploy_asset_id = str(deploy.get("repo_id"))
    archive_asset_ids = sorted(
        path.parent.name
        for path in (checkpoint / "assets").glob("*/norm_stats.json")
    )
    provenance = {
        "robotwin_commit": git_revision(args.robotwin_root),
        "xpolicylab_commit": git_revision(args.robotwin_root / "XPolicyLab"),
        "robotwin_root": str(args.robotwin_root.resolve()),
    }
    checkpoint_disclosure = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "public_archive_sha256": PUBLIC_ARCHIVE_SHA256,
        "official_deploy_checkpoint_num": deploy_step,
        "matches_official_deploy_step": checkpoint_step == deploy_step,
        "official_deploy_asset_id": deploy_asset_id,
        "public_archive_asset_ids": archive_asset_ids,
        "matches_official_deploy_asset_id": deploy_asset_id in archive_asset_ids,
        "official_train_num_steps": 60000,
        "note": (
            "The official public archive contains only step 29999, while the "
            "published Pi_05 deploy.yml requests step 59999 from a 60000-step "
            "training run. Its configured normalization asset id is also not "
            "present in the archive. The released artifact is therefore not "
            "byte/config identical to the leaderboard deployment artifact."
        ),
    }
    output = args.output_dir.resolve()
    (output / "clean").mkdir(parents=True, exist_ok=True)
    (output / "randomized").mkdir(parents=True, exist_ok=True)
    (output / "clean" / "per_task.json").write_text(json.dumps(clean_rows, indent=2) + "\n")
    (output / "clean" / "summary.json").write_text(json.dumps(clean, indent=2) + "\n")
    (output / "randomized" / "per_task.json").write_text(json.dumps(randomized_rows, indent=2) + "\n")
    (output / "randomized" / "summary.json").write_text(json.dumps(randomized, indent=2) + "\n")
    comparison = {
        "reference": reference_metadata,
        "provenance": provenance,
        "checkpoint_disclosure": checkpoint_disclosure,
        "clean": clean,
        "randomized": randomized,
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    random_by_task = {row["task"]: row for row in randomized_rows}
    lines = [
        "# RoboTwin pi0.5 official Orbax baseline",
        "",
        f"Checkpoint: `{clean['inference']['checkpoint']}`",
        f"RoboTwin: `{provenance['robotwin_commit']}`; XPolicyLab: `{provenance['xpolicylab_commit']}`.",
        f"Protocol: 50 tasks x {args.episodes} accepted episodes, seed stream from 100000, official expert prefilter, RT+OIDN, one GPU.",
        "",
        "## Checkpoint disclosure",
        "",
        f"The public archive contains step `{checkpoint_step}`, but the official Pi_05 deployment config requests `{deploy_step}`. "
        "The requested step is not present in the public archive, so these results validate the released 29999 artifact and cannot prove byte-equivalence to the leaderboard checkpoint.",
        f"The official deployment requests normalization asset `{deploy_asset_id}`; the public archive contains `{', '.join(archive_asset_ids)}`.",
        f"Public archive SHA-256: `{PUBLIC_ARCHIVE_SHA256}`.",
        "",
        "| Setting | This run | Official 100-trial result | Difference | 95% Wilson interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, summary in (("Clean", clean), ("Randomized", randomized)):
        interval = summary["wilson_95_percent"]
        lines.append(
            f"| {label} | {100 * summary['success_rate']:.2f}% | "
            f"{100 * summary['official_100_trial_success_rate']:.2f}% | "
            f"{summary['difference_percentage_points']:+.2f} pp | "
            f"{100 * interval[0]:.2f}% to {100 * interval[1]:.2f}% |"
        )
    lines += [
        "",
        "The 10-trial estimate is intentionally coarser than the official 100-trial result; each task can only move in 10-point increments.",
        (
            "Using the official per-task rates as fixed Bernoulli probabilities, the expected 10-trial suite-mean sampling SD is "
            f"{100 * clean['reference_rate_sampling_sd_at_requested_trials']:.2f} pp for Clean and "
            f"{100 * randomized['reference_rate_sampling_sd_at_requested_trials']:.2f} pp for Randomized. "
            "This quantifies sampling noise but does not erase a systematic checkpoint mismatch."
        ),
        "The explicit JAX noise stream was separately verified elementwise identical to OpenPI Policy.infer()'s internal seed-0 stream for 1,000 consecutive chunks.",
        (
            "Per-task agreement with the leaderboard remains measurable despite 10-point task-level quantization: "
            f"Pearson r={clean['per_task_rate_pearson_correlation']:.3f} for Clean and "
            f"r={randomized['per_task_rate_pearson_correlation']:.3f} for Randomized."
        ),
        "The often-cited 82.7%/76.8% values are from the older RoboTwin2.0-Full literature setting and are not the current clean-only co-train leaderboard entry represented by this checkpoint.",
        "",
        "| Task | Clean | Official | Randomized | Official |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in clean_rows:
        other = random_by_task[row["task"]]
        lines.append(
            f"| `{row['task']}` | {100 * row['success_rate']:.0f}% | "
            f"{row['official_100_trial_percent']:.0f}% | {100 * other['success_rate']:.0f}% | "
            f"{other['official_100_trial_percent']:.0f}% |"
        )
    (output / "final_report.md").write_text("\n".join(lines) + "\n")

    if args.artifact_dir is not None:
        artifact_dir = args.artifact_dir.resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "checkpoint_reference.json").write_text(
            json.dumps(
                {
                    **checkpoint_disclosure,
                    **provenance,
                    "reference": reference_metadata,
                    "inference": clean["inference"],
                },
                indent=2,
            )
            + "\n"
        )
        (artifact_dir / "eval_clean.json").write_text(
            json.dumps({"summary": clean, "per_task": clean_rows}, indent=2) + "\n"
        )
        (artifact_dir / "eval_randomized.json").write_text(
            json.dumps({"summary": randomized, "per_task": randomized_rows}, indent=2)
            + "\n"
        )
        episodes = [
            *episode_records(args.clean_root, "clean", tasks),
            *episode_records(args.randomized_root, "randomized", tasks),
        ]
        for success, filename in (
            (True, "success_episodes.jsonl"),
            (False, "failure_episodes.jsonl"),
        ):
            payload = "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in episodes
                if bool(record["success"]) is success
            )
            (artifact_dir / filename).write_text(payload, encoding="utf-8")
        (artifact_dir / "baseline_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
