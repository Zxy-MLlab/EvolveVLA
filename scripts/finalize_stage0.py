#!/usr/bin/env python3
"""Validate full Stage-0 sweeps and freeze the Generation-0 records."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "pi05_public_baseline"
ROBOTWIN = ROOT / "robotwin_repo"
CKPT = ROOT / "checkpoints" / "pi05_robotwin2_public"
AGG = ROOT / "scripts" / "aggregate_public_openpi_eval.py"
PYTHON = "/root/autodl-tmp/conda_envs/openpi_public/bin/python"
MODEL_BYTES = 7_473_091_464
MODEL_SHA256 = "de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb"
ROBOTWIN_COMMIT = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
XPOLICYLAB_COMMIT = "fa431ecd893ee706883e64fe5fe1464ec8cd928d"
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> dict[str, str]:
    model = CKPT / "model.safetensors"
    metadata = CKPT / "metadata.pt"
    norm = CKPT / "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
    for required in (model, metadata, norm):
        if not required.is_file():
            raise SystemExit(f"missing required inference asset: {required}")
    if model.stat().st_size != MODEL_BYTES:
        raise SystemExit(f"model size mismatch: {model.stat().st_size} != {MODEL_BYTES}")
    model_hash = sha256_file(model)
    if model_hash != MODEL_SHA256:
        raise SystemExit(f"model SHA-256 mismatch: {model_hash}")
    if metadata.stat().st_size <= 0:
        raise SystemExit("metadata.pt is empty")
    try:
        with zipfile.ZipFile(metadata) as archive:
            pickle_members = [name for name in archive.namelist() if name.endswith("data.pkl")]
            metadata_bytes = b"".join(archive.read(name) for name in pickle_members)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"metadata.pt is not a readable torch archive: {exc}") from exc
    if SCENE_PREFIX.encode("utf-8") not in metadata_bytes:
        raise SystemExit("metadata.pt does not contain the published RoboTwin scene prompt prefix")
    norm_payload = json.loads(norm.read_text())
    if not isinstance(norm_payload, dict) or not norm_payload:
        raise SystemExit("normalization statistics are empty or malformed")
    robotwin_commit = subprocess.check_output(
        ["git", "-C", str(ROBOTWIN), "rev-parse", "HEAD"], text=True
    ).strip()
    xpolicylab_commit = subprocess.check_output(
        ["git", "-C", str(ROBOTWIN / "XPolicyLab"), "rev-parse", "HEAD"], text=True
    ).strip()
    if robotwin_commit != ROBOTWIN_COMMIT:
        raise SystemExit(f"RoboTwin commit mismatch: {robotwin_commit}")
    if xpolicylab_commit != XPOLICYLAB_COMMIT:
        raise SystemExit(f"XPolicyLab commit mismatch: {xpolicylab_commit}")
    adapter = ROOT / "scripts/pi05_openpi_adapter.py"
    runner = ROOT / "scripts/robotwin_public_openpi_harness.py"
    simulator_harness = ROOT / "scripts/robotwin_eval_harness.py"
    instruction_source = ROOT / "scripts/robotwin_instruction_source.py"
    parallel_runner = ROOT / "scripts/run_public_openpi_parallel.py"
    return {
        "model_sha256": model_hash,
        "robotwin_commit": robotwin_commit,
        "xpolicylab_commit": xpolicylab_commit,
        "adapter_sha256": sha256_file(adapter),
        "runner_sha256": sha256_file(runner),
        "simulator_harness_sha256": sha256_file(simulator_harness),
        "instruction_source_sha256": sha256_file(instruction_source),
        "parallel_runner_sha256": sha256_file(parallel_runner),
    }


def run_aggregate(setting: str, run_root: Path, expected_trials: int) -> dict:
    out = RESULTS / setting
    subprocess.run([
        PYTHON, str(AGG), "--run-root", str(run_root), "--output-dir", str(out),
        "--task-config", "demo_clean" if setting == "clean" else "demo_randomized",
        "--checkpoint", "motus-robotics/pi0.5_robotwin2@effc6e1a99bd4d0edf9447e8ba313def67649cfe",
        "--robotwin-commit", "ea8b211",
        "--expected-trials-per-task", str(expected_trials),
    ], check=True)
    return json.loads((out / "summary.json").read_text())


def select_sweeps() -> tuple[Path, Path, int]:
    """Select a matched clean/randomized pair without mistaking old partial data for a run."""
    # The current user-requested run is the 10-trial diagnostic pair.  Keep
    # the official pair as a fallback so a later 100-trial run still freezes
    # through the same entry point.
    for suffix, trials in (("_10", 10), ("", 100)):
        clean = RESULTS / f"full_clean{suffix}"
        randomized = RESULTS / f"full_randomized{suffix}"
        if clean.is_dir() and randomized.is_dir():
            return clean, randomized, trials
    raise SystemExit("no matched full clean/randomized sweep pair found")


def main() -> int:
    verified = verify_sources()
    clean_root, randomized_root, expected_trials = select_sweeps()
    clean = run_aggregate("clean", clean_root, expected_trials)
    randomized = run_aggregate("randomized", randomized_root, expected_trials)
    required = (clean, randomized)
    expected_total = 50 * expected_trials
    valid_statuses = {"official_50_task_complete", f"50_task_complete_{expected_trials}_trials_nonofficial"}
    if any(x.get("status") not in valid_statuses or x.get("tasks") != 50 or x.get("trials") != expected_total for x in required):
        raise SystemExit("full evaluation incomplete; Generation-0 was not frozen")

    artifact = ROOT / "artifacts" / "gen0"
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint_ref = {
        "source": "motus-robotics/pi0.5_robotwin2",
        "revision": "effc6e1a99bd4d0edf9447e8ba313def67649cfe",
        "checkpoint_dir": str(CKPT),
        "model_bytes": MODEL_BYTES,
        "model_sha256": verified["model_sha256"],
        "robotwin_commit": verified["robotwin_commit"],
        "xpolicylab_openpi_commit": verified["xpolicylab_commit"],
        "openpi_compatibility_edit": "robotwin_repo/XPolicyLab/policy/Pi_05/openpi/src/openpi/training/checkpoints.py (lazy TYPE_CHECKING-only training-data import)",
        "adapter": "EvolveVLA/scripts/pi05_openpi_adapter.py",
        "adapter_sha256": verified["adapter_sha256"],
        "runner": "EvolveVLA/scripts/robotwin_public_openpi_harness.py",
        "runner_sha256": verified["runner_sha256"],
        "simulator_harness": "EvolveVLA/scripts/robotwin_eval_harness.py",
        "simulator_harness_sha256": verified["simulator_harness_sha256"],
        "instruction_source_sha256": verified["instruction_source_sha256"],
        "parallel_runner_sha256": verified["parallel_runner_sha256"],
        "action_horizon": 32, "model_action_dim": 32, "simulator_action_dim": 14,
        "camera_mapping": {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"},
        "normalization": "assets/pi0.5_clean_randomize_joint_training/norm_stats.json",
        "prompt_prefix": SCENE_PREFIX,
        "episodes_per_task": expected_trials,
        "evaluation_status": clean["status"],
        "official_reference_comparable": expected_trials == 100,
        "wandb": "disabled", "gpu": "single visible GPU 0",
    }
    (artifact / "checkpoint_reference.json").write_text(json.dumps(checkpoint_ref, indent=2) + "\n")
    (artifact / "eval_clean.json").write_text(json.dumps(clean, indent=2) + "\n")
    (artifact / "eval_randomized.json").write_text(json.dumps(randomized, indent=2) + "\n")
    (artifact / "failure_episodes.jsonl").write_text(
        (RESULTS / "clean" / "failure_episodes.jsonl").read_text() +
        (RESULTS / "randomized" / "failure_episodes.jsonl").read_text()
    )
    (artifact / "success_episodes.jsonl").write_text(
        (RESULTS / "clean" / "success_episodes.jsonl").read_text() +
        (RESULTS / "randomized" / "success_episodes.jsonl").read_text()
    )
    clean_rows = json.loads((RESULTS / "clean" / "per_task.json").read_text())
    rand_rows = json.loads((RESULTS / "randomized" / "per_task.json").read_text())
    rand_map = {x["task"]: x for x in rand_rows}
    lines = [
        "# EvolveVLA Generation-0 public π0.5 baseline",
        "",
        "This record freezes the unmodified public `motus-robotics/pi0.5_robotwin2` checkpoint in the official RoboTwin 2.0 environment.",
        "",
        f"- RoboTwin commit: `ea8b211`",
        f"- XPolicyLab/OpenPI commit: `{XPOLICYLAB_COMMIT}`",
        f"- Checkpoint revision: `effc6e1a99bd4d0edf9447e8ba313def67649cfe`",
        "- Inference-only compatibility edit: lazy `TYPE_CHECKING`-only training-data import in `openpi/training/checkpoints.py`; checkpoint weights, transforms, and protocol are unchanged.",
        f"- Protocol: 50 official tasks × {expected_trials} accepted episodes per setting, deterministic seed stream starting at 1; official expert/scene validation rejects invalid seeds and sampling continues until {expected_trials} accepted trials per task. The cap is 1,200 steps; execution is single-GPU with wandb disabled.",
        "- OpenPI config: `Pi0Config(action_dim=32, action_horizon=32, pi05=True)` with `LeRobotAlohaDataConfig(adapt_to_pi=True, use_delta_joint_actions=True)`.",
        "- Normalization: `assets/pi0.5_clean_randomize_joint_training/norm_stats.json`; state is 14-D and model actions are 32-D, converted to 14-D simulator joint actions.",
        "- Prompt contract: the public Motus/RoboTwin scene prefix is prepended before tokenization, matching the training converter and published deployment path.",
        f"- Harness source hashes: adapter `{verified['adapter_sha256']}`, parent runner `{verified['runner_sha256']}`, simulator bridge `{verified['simulator_harness_sha256']}`, parallel scheduler `{verified['parallel_runner_sha256']}`.",
        "- Cameras: `cam_high←head_camera`, `cam_left_wrist←left_camera`, `cam_right_wrist←right_camera`; each returned 32-step action chunk is executed sequentially.",
        "",
        "| Task | Clean SR | Randomized SR |",
        "|---|---:|---:|",
    ]
    for row in clean_rows:
        lines.append(f"| `{row['task']}` | {100*row['success_rate']:.2f}% | {100*rand_map[row['task']]['success_rate']:.2f}% |")
    lines += [
        "",
        f"Clean overall: **{100*clean['success_rate']:.2f}%** ({clean['successes']}/{clean['trials']}); randomized overall: **{100*randomized['success_rate']:.2f}%** ({randomized['successes']}/{randomized['trials']}).",
        (f"Reference π0.5 Full values are 82.70% clean and 76.80% randomized; differences are {100*(clean['success_rate']-0.827):+.2f} and {100*(randomized['success_rate']-0.768):+.2f} percentage points."
         if expected_trials == 100 else
         "These 10-trial/task results are diagnostic and are not directly comparable to the official 100-trial reference percentages."),
    ]
    (artifact / "baseline_report.md").write_text("\n".join(lines) + "\n")
    # Keep the user-facing report in the requested results directory in sync
    # with the frozen artifact.  The pre-evaluation report intentionally says
    # that the full sweep is pending; once both sweeps pass the gate above,
    # append the authoritative full-benchmark section rather than leaving a
    # stale pending statement as the primary result.
    report_path = RESULTS / "final_report.md"
    previous = report_path.read_text() if report_path.is_file() else ""
    marker = f"\n## 50-task evaluation (Generation-0; {expected_trials} trials/task)\n"
    full_report = marker + "\n" + "\n".join(lines[4:]) + "\n"
    # Remove a prior generated section (including the historical official
    # marker) before appending the current authoritative sweep.
    for old_marker in (marker, "\n## Official 50-task evaluation (Generation-0)\n"):
        if old_marker in previous:
            previous = previous.split(old_marker, 1)[0].rstrip() + "\n"
    report_path.write_text(previous.rstrip() + "\n" + full_report)
    print(json.dumps({"status": "gen0_frozen", "clean": clean, "randomized": randomized}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
