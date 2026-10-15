#!/usr/bin/env python3
"""Validate and report a full 10-trial RoboTwin pi0.5 candidate sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

import aggregate_official_orbax_eval as core


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(checkpoint: Path) -> dict[str, object]:
    manifest = checkpoint / "SHA256SUMS"
    if not manifest.is_file():
        raise SystemExit(f"missing checkpoint hash manifest: {manifest}")
    verified: list[str] = []
    omitted: list[str] = []
    mismatches: list[dict[str, str]] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = raw_line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = checkpoint / relative
        if not path.is_file():
            # README is not needed for inference and was intentionally excluded
            # from the selective checkpoint download.
            if relative == "README.md":
                omitted.append(relative)
                continue
            raise SystemExit(f"missing checkpoint file listed in SHA256SUMS: {path}")
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(
                {"path": relative, "expected_sha256": expected, "actual_sha256": actual}
            )
        else:
            verified.append(relative)
    if mismatches:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {mismatches}")

    metadata_path = checkpoint / "rldx2_openpi.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    norm_paths = sorted((checkpoint / "assets").glob("*/norm_stats.json"))
    if len(norm_paths) != 1:
        raise SystemExit(f"expected exactly one normalization asset, got {norm_paths}")
    incomplete = sorted(
        str(path.relative_to(checkpoint))
        for path in (checkpoint / ".cache").rglob("*.incomplete")
    ) if (checkpoint / ".cache").is_dir() else []
    if incomplete:
        raise SystemExit(f"checkpoint still has incomplete downloads: {incomplete}")
    return {
        "checkpoint": str(checkpoint),
        "format": "openpi_orbax",
        "declared_config_name": metadata.get("config_name"),
        "declared_step": metadata.get("step"),
        "declared_asset_id": metadata.get("asset_id"),
        "normalization_asset": str(norm_paths[0].relative_to(checkpoint)),
        "sha256_manifest": str(manifest),
        "sha256_manifest_sha256": sha256_file(manifest),
        "verified_inference_files": len(verified),
        "intentionally_omitted_non_inference_files": omitted,
        "sha256_mismatches": mismatches,
        "incomplete_downloads": incomplete,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--randomized-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Optional immutable Generation-0-style export for this candidate.",
    )
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument(
        "--baseline-comparison",
        type=Path,
        default=Path(
            "/root/autodl-tmp/EvolveVLA/results/pi05_official_orbax_baseline/"
            "protocol_v11_ba0fd21_report/comparison.json"
        ),
    )
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "robotwin_repo_ba0fd21",
    )
    args = parser.parse_args()

    tasks = list(
        yaml.safe_load(
            (args.robotwin_root / "env_cfg/eval/all_tasks.yml").read_text(
                encoding="utf-8"
            )
        )["tasks"]
    )
    reference_metadata, reference = core.load_reference(args.reference_json)
    missing = sorted(set(tasks) - set(reference))
    if missing:
        raise SystemExit(f"official reference is missing tasks: {missing}")

    clean, clean_rows = core.aggregate_setting(
        args.clean_root,
        "demo_clean",
        tasks,
        reference,
        "clean",
        args.episodes,
        "jax",
    )
    randomized, randomized_rows = core.aggregate_setting(
        args.randomized_root,
        "demo_randomized",
        tasks,
        reference,
        "randomized",
        args.episodes,
        "jax",
    )
    if clean["inference"] != randomized["inference"]:
        raise SystemExit("Clean and Randomized inference configurations differ")

    overall_successes = int(clean["successes"]) + int(randomized["successes"])
    overall_trials = int(clean["trials"]) + int(randomized["trials"])
    overall_rate = overall_successes / overall_trials
    official_overall_rate = (
        float(clean["official_100_trial_success_rate"])
        + float(randomized["official_100_trial_success_rate"])
    ) / 2.0
    overall_low, overall_high = core.wilson(overall_successes, overall_trials)
    overall = {
        "trials": overall_trials,
        "successes": overall_successes,
        "success_rate": overall_rate,
        "wilson_95_percent": [overall_low, overall_high],
        "official_100_trial_success_rate": official_overall_rate,
        "difference_percentage_points": 100.0
        * (overall_rate - official_overall_rate),
    }

    checkpoint = args.checkpoint.resolve()
    if Path(str(clean["inference"]["checkpoint"])).resolve() != checkpoint:
        raise SystemExit("evaluation checkpoint does not match --checkpoint")
    integrity = verify_checkpoint(checkpoint)
    provenance = {
        "candidate_name": args.candidate_name,
        "candidate_source": args.candidate_source,
        "candidate_revision": args.candidate_revision,
        "warning": (
            "This is an independently trained public reproduction, not the "
            "RoboTwin Team leaderboard artifact."
        ),
        "robotwin_commit": core.git_revision(args.robotwin_root),
        "xpolicylab_commit": core.git_revision(args.robotwin_root / "XPolicyLab"),
        "robotwin_root": str(args.robotwin_root.resolve()),
    }

    baseline = None
    if args.baseline_comparison.is_file():
        baseline_payload = json.loads(
            args.baseline_comparison.read_text(encoding="utf-8")
        )
        baseline = {
            "source": str(args.baseline_comparison.resolve()),
            "clean_success_rate": baseline_payload["clean"]["success_rate"],
            "randomized_success_rate": baseline_payload["randomized"]["success_rate"],
            "clean_candidate_gain_percentage_points": 100.0
            * (
                float(clean["success_rate"])
                - float(baseline_payload["clean"]["success_rate"])
            ),
            "randomized_candidate_gain_percentage_points": 100.0
            * (
                float(randomized["success_rate"])
                - float(baseline_payload["randomized"]["success_rate"])
            ),
            "overall_success_rate": (
                float(baseline_payload["clean"]["success_rate"])
                + float(baseline_payload["randomized"]["success_rate"])
            )
            / 2.0,
            "overall_candidate_gain_percentage_points": 100.0
            * (
                overall_rate
                - (
                    float(baseline_payload["clean"]["success_rate"])
                    + float(baseline_payload["randomized"]["success_rate"])
                )
                / 2.0
            ),
        }

    output = args.output_dir.resolve()
    (output / "clean").mkdir(parents=True, exist_ok=True)
    (output / "randomized").mkdir(parents=True, exist_ok=True)
    write_json(output / "clean" / "per_task.json", clean_rows)
    write_json(output / "clean" / "summary.json", clean)
    write_json(output / "randomized" / "per_task.json", randomized_rows)
    write_json(output / "randomized" / "summary.json", randomized)
    write_json(
        output / "official_reference.json",
        {"metadata": reference_metadata, "per_task": reference},
    )
    comparison = {
        "status": "complete",
        "reference": reference_metadata,
        "provenance": provenance,
        "checkpoint_integrity": integrity,
        "protocol": {
            "tasks": len(tasks),
            "episodes_per_task": args.episodes,
            "seed_start": 100000,
            "expert_prefilter": "official",
            "instruction_generator": "official",
            "action_horizon": 50,
            "normalization": "quantile",
            "policy_seed": 0,
            "noise_source": "jax",
            "renderer": "rt+oidn",
            "gpu_workers": 1,
        },
        "clean": clean,
        "randomized": randomized,
        "overall": overall,
        "baseline_29999": baseline,
    }
    write_json(output / "comparison.json", comparison)

    random_by_task = {row["task"]: row for row in randomized_rows}
    lines = [
        f"# {args.candidate_name} RoboTwin 2.0 diagnostic",
        "",
        provenance["warning"],
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Source: `{args.candidate_source}@{args.candidate_revision}`",
        f"Declared step: `{integrity['declared_step']}`; normalization asset: `{integrity['declared_asset_id']}`.",
        f"RoboTwin: `{provenance['robotwin_commit']}`; XPolicyLab: `{provenance['xpolicylab_commit']}`.",
        f"Protocol: 50 tasks x {args.episodes} accepted episodes, seed stream from 100000, official expert prefilter/instructions, 50-step chunks, quantile normalization, JAX seed-0 noise, RT+OIDN, one GPU and one worker.",
        "",
        "| Setting | Candidate | Official 100-trial result | Difference | 95% Wilson interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, summary in (
        ("Clean", clean),
        ("Randomized", randomized),
        ("Overall", overall),
    ):
        interval = summary["wilson_95_percent"]
        lines.append(
            f"| {label} | {100 * summary['success_rate']:.2f}% | "
            f"{100 * summary['official_100_trial_success_rate']:.2f}% | "
            f"{summary['difference_percentage_points']:+.2f} pp | "
            f"{100 * interval[0]:.2f}% to {100 * interval[1]:.2f}% |"
        )
    if baseline is not None:
        lines += [
            "",
            "Compared with the verified public 29999 baseline, this candidate changes "
            f"Clean by {baseline['clean_candidate_gain_percentage_points']:+.2f} pp and "
            f"Randomized by {baseline['randomized_candidate_gain_percentage_points']:+.2f} pp; "
            f"the Overall change is {baseline['overall_candidate_gain_percentage_points']:+.2f} pp.",
        ]
    lines += [
        "",
        "The 10-trial estimate is intentionally coarser than the official 100-trial result; each task can only move in 10-point increments.",
        (
            "Using the official per-task rates as fixed Bernoulli probabilities, the expected 10-trial suite-mean sampling SD is "
            f"{100 * clean['reference_rate_sampling_sd_at_requested_trials']:.2f} pp for Clean and "
            f"{100 * randomized['reference_rate_sampling_sd_at_requested_trials']:.2f} pp for Randomized."
        ),
        f"Checkpoint integrity: {integrity['verified_inference_files']} inference files matched `SHA256SUMS`; no incomplete downloads or hash mismatches were present. README was intentionally omitted from the selective inference download.",
        "",
        "## Reproducibility",
        "",
        "The complete machine-readable inference configuration is:",
        "",
        "```json",
        json.dumps(clean["inference"], indent=2, sort_keys=True),
        "```",
        "",
        "The accepted seed lists below are after the official expert prefilter; rejected scene seeds are intentionally absent.",
        "",
        "| Task | Clean accepted seeds | Randomized accepted seeds |",
        "|---|---|---|",
    ]
    for row in clean_rows:
        other = random_by_task[row["task"]]
        clean_seeds = ", ".join(str(seed) for seed in row["seeds"])
        randomized_seeds = ", ".join(str(seed) for seed in other["seeds"])
        lines.append(f"| `{row['task']}` | {clean_seeds} | {randomized_seeds} |")
    lines += [
        "",
        "## Per-task results",
        "",
        "| Task | Clean | Official | Randomized | Official |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in clean_rows:
        other = random_by_task[row["task"]]
        lines.append(
            f"| `{row['task']}` | {100 * row['success_rate']:.0f}% | "
            f"{row['official_100_trial_percent']:.0f}% | "
            f"{100 * other['success_rate']:.0f}% | "
            f"{other['official_100_trial_percent']:.0f}% |"
        )
    report = "\n".join(lines) + "\n"
    (output / "final_report.md").write_text(report, encoding="utf-8")

    if args.artifact_dir is not None:
        artifact_dir = args.artifact_dir.resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            artifact_dir / "checkpoint_reference.json",
            {
                **provenance,
                "reference": reference_metadata,
                "checkpoint_integrity": integrity,
                "inference": clean["inference"],
            },
        )
        write_json(
            artifact_dir / "eval_clean.json",
            {"summary": clean, "per_task": clean_rows},
        )
        write_json(
            artifact_dir / "eval_randomized.json",
            {"summary": randomized, "per_task": randomized_rows},
        )
        write_json(
            artifact_dir / "official_reference.json",
            {"metadata": reference_metadata, "per_task": reference},
        )
        episodes = [
            *core.episode_records(args.clean_root, "clean", tasks),
            *core.episode_records(args.randomized_root, "randomized", tasks),
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
        (artifact_dir / "baseline_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
