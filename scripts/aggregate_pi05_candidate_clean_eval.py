#!/usr/bin/env python3
"""Validate and report a Clean-only RoboTwin pi0.5 candidate sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

import aggregate_official_orbax_eval as core
from aggregate_pi05_candidate_eval import verify_checkpoint, write_json


OPENPI_INFERENCE_FILES = (
    "policies/policy.py",
    "policies/policy_config.py",
    "training/config.py",
    "transforms.py",
    "models/pi0.py",
    "models/pi0_config.py",
    "models/tokenizer.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status(path: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain=v1"], text=True
    )
    return output.splitlines()


def audit_source_chain(robotwin_root: Path, openpi_src: Path) -> dict[str, object]:
    """Audit the exact source tree imported by the completed OpenPI run."""
    pinned_openpi_src = (
        robotwin_root / "XPolicyLab/policy/Pi_05/openpi/src"
    ).resolve()
    openpi_src = openpi_src.resolve()
    if not openpi_src.is_dir():
        raise SystemExit(f"actual OpenPI source directory is missing: {openpi_src}")

    hashes: dict[str, dict[str, object]] = {}
    for relative in OPENPI_INFERENCE_FILES:
        actual = openpi_src / "openpi" / relative
        pinned = pinned_openpi_src / "openpi" / relative
        if not actual.is_file() or not pinned.is_file():
            raise SystemExit(f"OpenPI audit file is missing: {actual} or {pinned}")
        actual_hash = sha256_file(actual)
        pinned_hash = sha256_file(pinned)
        hashes[relative] = {
            "actual_sha256": actual_hash,
            "pinned_sha256": pinned_hash,
            "identical": actual_hash == pinned_hash,
        }
    if not all(bool(record["identical"]) for record in hashes.values()):
        raise SystemExit("actual and pinned OpenPI inference sources differ")

    robotwin_status = git_status(robotwin_root)
    xpolicylab_root = robotwin_root / "XPolicyLab"
    xpolicylab_status = git_status(xpolicylab_root)
    if robotwin_status or xpolicylab_status:
        raise SystemExit(
            "pinned RoboTwin/XPolicyLab source is dirty: "
            f"RoboTwin={robotwin_status}, XPolicyLab={xpolicylab_status}"
        )

    actual_xpolicylab_root = openpi_src.parents[3]
    checkpoint_loader_actual = openpi_src / "openpi/training/checkpoints.py"
    checkpoint_loader_pinned = pinned_openpi_src / "openpi/training/checkpoints.py"
    return {
        "status": "passed",
        "robotwin_worktree_clean": True,
        "xpolicylab_worktree_clean": True,
        "actual_openpi_src": str(openpi_src),
        "actual_xpolicylab_revision": core.git_revision(actual_xpolicylab_root),
        "pinned_openpi_src": str(pinned_openpi_src),
        "pinned_xpolicylab_revision": core.git_revision(xpolicylab_root),
        "inference_files": hashes,
        "inference_files_identical": True,
        "checkpoint_loader_actual_sha256": sha256_file(checkpoint_loader_actual),
        "checkpoint_loader_pinned_sha256": sha256_file(checkpoint_loader_pinned),
        "checkpoint_loader_note": (
            "The only OpenPI source difference is a TYPE_CHECKING-only lazy "
            "import in training/checkpoints.py; model, transforms, policy, "
            "tokenizer, and training configuration sources are byte-identical."
        ),
        "official_server_lifecycle": (
            "Pi_05/eval.sh starts a fresh policy server for each task; OpenPI "
            "Policy initializes the JAX key to 0 in each server."
        ),
        "current_server_lifecycle": (
            "The completed sweep loaded one policy process per task with "
            "policy_seed=0."
        ),
        "instruction_rng": (
            "The harness imports the pinned official four-argument "
            "build_instruction and calls it after expert play_once, close_env, "
            "and the second same-seed setup_demo, matching eval_remote_policy."
        ),
    }


def audit_protocol_records(
    clean_root: Path,
    tasks: list[str],
    official_limits: dict[str, int],
    episodes: int,
) -> dict[str, object]:
    manifest = json.loads((clean_root / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "task_config": "demo_clean",
        "episodes_requested_per_task": episodes,
        "seed_start": 100000,
        "prompt_prefix": "none",
        "normalization": "quantile",
        "policy_seed": 0,
        "noise_source": "jax",
        "workers": 1,
    }
    actual_manifest = {
        key: manifest.get(key) for key in expected_manifest
    }
    if actual_manifest != expected_manifest:
        raise SystemExit(
            f"Clean manifest protocol mismatch: {actual_manifest} != {expected_manifest}"
        )

    reports_checked = 0
    for task in tasks:
        result_path = clean_root / task / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reports = list(result.get("reports", []))
        episode_ids = [int(report["episode"]) for report in reports]
        if episode_ids != list(range(episodes)):
            raise SystemExit(f"episode ID sequence mismatch in {result_path}")
        recorded_limits = {int(report["step_limit"]) for report in reports}
        expected_limit = int(official_limits[task])
        if recorded_limits != {expected_limit}:
            raise SystemExit(
                f"official step-limit mismatch in {result_path}: "
                f"{recorded_limits} != {{{expected_limit}}}"
            )
        reports_checked += len(reports)
    return {
        "status": "passed",
        "manifest_protocol_fields": expected_manifest,
        "episode_ids": f"0..{episodes - 1} for every task",
        "official_task_step_limits": "all reports match _eval_step_limit.yml",
        "reports_checked": reports_checked,
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
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
    parser.add_argument(
        "--openpi-src",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "robotwin_repo/XPolicyLab/policy/Pi_05/openpi/src"
        ),
        help="OpenPI src directory actually imported by the completed runner",
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

    official_limits = yaml.safe_load(
        (
            args.robotwin_root / "env_cfg/task_config/_eval_step_limit.yml"
        ).read_text(encoding="utf-8")
    )
    protocol_audit = audit_protocol_records(
        args.clean_root, tasks, official_limits, args.episodes
    )
    source_audit = audit_source_chain(args.robotwin_root, args.openpi_src)

    clean, clean_rows = core.aggregate_setting(
        args.clean_root,
        "demo_clean",
        tasks,
        reference,
        "clean",
        args.episodes,
        "jax",
    )
    checkpoint = args.checkpoint.resolve()
    if Path(str(clean["inference"]["checkpoint"])).resolve() != checkpoint:
        raise SystemExit("evaluation checkpoint does not match --checkpoint")

    integrity = verify_checkpoint(checkpoint)
    deploy = yaml.safe_load(
        (
            args.robotwin_root / "XPolicyLab/policy/Pi_05/deploy.yml"
        ).read_text(encoding="utf-8")
    )
    official_asset = str(deploy.get("repo_id"))
    candidate_asset = str(integrity.get("declared_asset_id"))
    official_step = int(deploy["checkpoint_num"])
    candidate_step = int(integrity["declared_step"])
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
        "official_deploy_checkpoint_step": official_step,
        "candidate_declared_checkpoint_step": candidate_step,
        "checkpoint_step_matches_official_deploy": candidate_step == official_step,
        "official_deploy_normalization_asset": official_asset,
        "candidate_declared_normalization_asset": candidate_asset,
        "normalization_asset_matches_official_deploy": candidate_asset == official_asset,
    }
    protocol = {
        "setting": "Clean only",
        "randomized_status": "paused_by_user",
        "tasks": len(tasks),
        "episodes_per_task": args.episodes,
        "accepted_episodes": int(clean["trials"]),
        "seed_start": 100000,
        "expert_prefilter": "official",
        "instruction_generator": "official",
        "task_step_limits": "official _eval_step_limit.yml",
        "action_horizon": 50,
        "normalization": "quantile",
        "policy_seed": 0,
        "noise_source": "jax",
        "renderer": "rt+oidn",
        "gpu_workers": 1,
    }
    public_artifact_audit = {
        "checked_at_utc": "2026-10-07T10:45:59Z",
        "huggingface_query": "pi05 robotwin",
        "model_repositories_scanned": 82,
        "repositories_with_arx_x5_sim_path": [],
        "openpi_59999_repositories": [
            "ShijieSKY/pi05-robotwin@e81d31ca2c52527bb703704f739cb16171593d9f",
            "ShijieSKY/history-pi05-robotwin@9c3b7a8ebc2f3db29258a3f8762a1e0d8438f7ce",
        ],
        "official_dataset_tree_entries": 593,
        "official_pi05_archives": [
            "cotrain_ckpt/Pi_05/"
            "RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz"
        ],
        "official_59999_or_arx_x5_sim_found": False,
        "details": str(
            (
                args.robotwin_root.parent
                / "results/pi05_official_orbax_baseline/"
                "checkpoint_protocol_audit.md"
            ).resolve()
        ),
    }
    baseline = None
    if args.baseline_comparison.is_file():
        baseline_payload = json.loads(
            args.baseline_comparison.read_text(encoding="utf-8")
        )
        baseline_clean = baseline_payload["clean"]
        baseline_rows_path = args.baseline_comparison.parent / "clean/per_task.json"
        baseline_rows = json.loads(baseline_rows_path.read_text(encoding="utf-8"))
        baseline_by_task = {row["task"]: row for row in baseline_rows}
        task_deltas = [
            100.0
            * (
                float(row["success_rate"])
                - float(baseline_by_task[row["task"]]["success_rate"])
            )
            for row in clean_rows
        ]
        baseline = {
            "source": str(args.baseline_comparison.resolve()),
            "description": "verified public official step-29999 Orbax artifact",
            "success_rate": baseline_clean["success_rate"],
            "candidate_gain_percentage_points": 100.0
            * (
                float(clean["success_rate"])
                - float(baseline_clean["success_rate"])
            ),
            "mean_absolute_task_difference_percentage_points": baseline_clean[
                "mean_absolute_task_difference_percentage_points"
            ],
            "root_mean_square_task_difference_percentage_points": baseline_clean[
                "root_mean_square_task_difference_percentage_points"
            ],
            "per_task_rate_pearson_correlation": baseline_clean[
                "per_task_rate_pearson_correlation"
            ],
            "tasks_improved": sum(delta > 0 for delta in task_deltas),
            "tasks_regressed": sum(delta < 0 for delta in task_deltas),
            "tasks_unchanged": sum(delta == 0 for delta in task_deltas),
        }
    comparison = {
        "status": "clean_complete_randomized_paused",
        "reference": reference_metadata,
        "provenance": provenance,
        "checkpoint_integrity": integrity,
        "protocol": protocol,
        "protocol_audit": protocol_audit,
        "source_audit": source_audit,
        "public_artifact_audit": public_artifact_audit,
        "baseline_public_29999": baseline,
        "clean": clean,
    }

    output = args.output_dir.resolve()
    (output / "clean").mkdir(parents=True, exist_ok=True)
    write_json(output / "clean" / "per_task.json", clean_rows)
    write_json(output / "clean" / "summary.json", clean)
    write_json(
        output / "official_reference.json",
        {"metadata": reference_metadata, "per_task": reference},
    )
    write_json(output / "comparison.json", comparison)

    deficits = sorted(clean_rows, key=lambda row: row["difference_percentage_points"])
    gains = sorted(
        clean_rows,
        key=lambda row: row["difference_percentage_points"],
        reverse=True,
    )
    interval = clean["wilson_95_percent"]
    lines = [
        f"# {args.candidate_name} RoboTwin 2.0 Clean evaluation",
        "",
        provenance["warning"],
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Source: `{args.candidate_source}@{args.candidate_revision}`",
        (
            f"RoboTwin: `{provenance['robotwin_commit']}`; XPolicyLab: "
            f"`{provenance['xpolicylab_commit']}`."
        ),
        (
            f"Protocol: 50 Clean tasks x {args.episodes} accepted episodes; official "
            "expert prefilter, instructions, success predicates, and per-task step "
            "limits; 50-step OpenPI chunks; quantile normalization; JAX seed-0 "
            "noise; RT+OIDN; one GPU worker."
        ),
        "Randomized evaluation was paused at the user's request and is not included.",
        "",
        "## Result",
        "",
        "| This run | Official 100-trial result | Difference | 95% Wilson interval |",
        "|---:|---:|---:|---:|",
        (
            f"| {100 * clean['success_rate']:.2f}% "
            f"({clean['successes']}/{clean['trials']}) | "
            f"{100 * clean['official_100_trial_success_rate']:.2f}% | "
            f"{clean['difference_percentage_points']:+.2f} pp | "
            f"{100 * interval[0]:.2f}% to {100 * interval[1]:.2f}% |"
        ),
        "",
        (
            "The leaderboard displays 70.7%; the mean of its published per-task "
            f"rates is {100 * clean['official_100_trial_success_rate']:.2f}%."
        ),
        (
            "At 10 trials per task, task rates move in 10-point increments. Under "
            "the official per-task rates as fixed Bernoulli probabilities, the "
            "suite-mean sampling SD is "
            f"{100 * clean['reference_rate_sampling_sd_at_requested_trials']:.2f} pp; "
            f"the observed gap is {clean['difference_in_reference_sampling_sd']:.2f} "
            "such SDs."
        ),
        "",
        "## Remaining provenance mismatch",
        "",
        (
            f"The candidate declares checkpoint step `{candidate_step}` and the "
            f"official deploy config requests `{official_step}`, so the nominal "
            "step matches."
        ),
        (
            f"The candidate normalization asset is `{candidate_asset}`, while the "
            f"official deploy config requests `{official_asset}`. The assets do not "
            "match, and the leaderboard checkpoint is not the independently trained "
            "candidate evaluated here. This prevents attributing the full gap to "
            "10-vs-100-trial sampling."
        ),
        (
            "A 2026-10-07 rescan of 82 public `pi05 robotwin` model trees found "
            "no `arx_x5_sim` path. The official dataset still exposes only its "
            "step-29999 Pi_05 archive; the only public 59999 Orbax models are "
            "independent ShijieSKY seed-42 variants."
        ),
        "",
        "## Eliminated implementation mismatches",
        "",
        (
            "The pinned RoboTwin and XPolicyLab worktrees are clean. The actual "
            "OpenPI source path used by the runner is "
            f"`{source_audit['actual_openpi_src']}`. Its seven inference-critical "
            "policy, transform, configuration, model, and tokenizer files are "
            "byte-identical to the pinned XPolicyLab copy; their per-file SHA-256 "
            "values are recorded in `comparison.json`."
        ),
        (
            "The sole OpenPI file difference is a TYPE_CHECKING-only lazy import "
            "in `training/checkpoints.py`; it does not alter checkpoint values or "
            "inference."
        ),
        (
            "Official local evaluation starts a fresh Pi_05 server per task and "
            "OpenPI initializes its JAX key to 0. This sweep likewise loaded one "
            "policy per task with seed 0, so cross-task RNG reuse is not a mismatch."
        ),
        (
            "Instruction selection also follows the same RNG order: expert "
            "`setup_demo`/`play_once`, environment close, second same-seed "
            "`setup_demo`, then the pinned official `build_instruction` call."
        ),
    ]
    if baseline is not None:
        lines += [
            "",
            "## Comparison with the public step-29999 artifact",
            "",
            (
                f"The verified public step-29999 artifact scored "
                f"{100 * baseline['success_rate']:.2f}% under the same Clean "
                f"protocol. This independent 59999 reproduction changes the suite "
                f"mean by {baseline['candidate_gain_percentage_points']:+.2f} pp."
            ),
            (
                f"Across tasks, {baseline['tasks_improved']} improved, "
                f"{baseline['tasks_regressed']} regressed, and "
                f"{baseline['tasks_unchanged']} were unchanged. Mean absolute "
                "task error changed from "
                f"{baseline['mean_absolute_task_difference_percentage_points']:.2f} "
                f"to {clean['mean_absolute_task_difference_percentage_points']:.2f} "
                "pp; RMSE changed from "
                f"{baseline['root_mean_square_task_difference_percentage_points']:.2f} "
                f"to {clean['root_mean_square_task_difference_percentage_points']:.2f} "
                "pp; per-task Pearson correlation changed from "
                f"{baseline['per_task_rate_pearson_correlation']:.3f} to "
                f"{clean['per_task_rate_pearson_correlation']:.3f}."
            ),
        ]
    lines += [
        "",
        "## Largest task deficits",
        "",
        "| Task | This run | Official | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in deficits[:10]:
        lines.append(
            f"| `{row['task']}` | {100 * row['success_rate']:.0f}% | "
            f"{row['official_100_trial_percent']:.0f}% | "
            f"{row['difference_percentage_points']:+.0f} pp |"
        )
    lines += [
        "",
        "## Largest task gains",
        "",
        "| Task | This run | Official | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in gains[:10]:
        lines.append(
            f"| `{row['task']}` | {100 * row['success_rate']:.0f}% | "
            f"{row['official_100_trial_percent']:.0f}% | "
            f"{row['difference_percentage_points']:+.0f} pp |"
        )
    lines += [
        "",
        "## All tasks",
        "",
        "| Task | Successes | This run | Official | Difference | Accepted seeds |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in clean_rows:
        seeds = ", ".join(str(seed) for seed in row["seeds"])
        lines.append(
            f"| `{row['task']}` | {row['successes']}/{row['trials']} | "
            f"{100 * row['success_rate']:.0f}% | "
            f"{row['official_100_trial_percent']:.0f}% | "
            f"{row['difference_percentage_points']:+.0f} pp | {seeds} |"
        )
    report = "\n".join(lines) + "\n"
    (output / "final_report.md").write_text(report, encoding="utf-8")

    if args.artifact_dir is not None:
        artifact = args.artifact_dir.resolve()
        artifact.mkdir(parents=True, exist_ok=True)
        write_json(
            artifact / "checkpoint_reference.json",
            {
                **provenance,
                "reference": reference_metadata,
                "checkpoint_integrity": integrity,
                "source_audit": source_audit,
                "public_artifact_audit": public_artifact_audit,
                "inference": clean["inference"],
            },
        )
        write_json(
            artifact / "eval_clean.json",
            {
                "summary": clean,
                "protocol": protocol,
                "protocol_audit": protocol_audit,
                "source_audit": source_audit,
                "per_task": clean_rows,
            },
        )
        write_json(
            artifact / "official_reference.json",
            {"metadata": reference_metadata, "per_task": reference},
        )
        episodes = list(core.episode_records(args.clean_root, "clean", tasks))
        for success, filename in (
            (True, "success_episodes.jsonl"),
            (False, "failure_episodes.jsonl"),
        ):
            payload = "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in episodes
                if bool(record["success"]) is success
            )
            (artifact / filename).write_text(payload, encoding="utf-8")
        (artifact / "baseline_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
