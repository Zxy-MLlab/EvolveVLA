"""Official RoboCasa baseline evaluation runner.

Implements the D005 baseline reproduction sequence:
  Step 0: Random rollouts (sanity check, no model needed)
  Step 1: Diffusion Policy baseline (1.7GB checkpoint)
  Step 2: Official pi0.5 baseline (44.7GB fine-tuned checkpoint)

Usage:
  python -m evolvevla.eval.robocasa_baseline_eval --step 0
  python -m evolvevla.eval.robocasa_baseline_eval --step 1 --checkpoint <path>
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from evolvevla.config import (
    DIFFUSION_POLICY_CHECKPOINT_TEMPLATE,
    ENV_RENDER_KWARGS,
    MAX_EPISODE_STEPS,
    NUM_ROLLOUTS_PER_TASK,
    OFFICIAL_BASELINE_CHECKPOINTS,
    PI05_ROBOCASA_CHECKPOINT_TEMPLATE,
    ROBOCASA365_SPLITS,
)
from evolvevla.eval.eval_runner import EvalConfig, EvalRunner, PolicyWrapper

logger = logging.getLogger(__name__)

# pretrain is required for RoboCasa leaderboard comparability.
ROBOCASA_LEADERBOARD_SPLIT = "pretrain"

# These lists are deliberately mutable: RoboCasa versions expose their
# benchmark task registry differently. ``get_task_splits`` fills them from the
# installed registry when possible, while CLI ``--tasks`` remains supported.
ATOMIC_SEEN_TASKS: list[str] = list(
    ROBOCASA365_SPLITS["atomic_seen"]["task_names"]
)
COMPOSITE_SEEN_TASKS: list[str] = list(
    ROBOCASA365_SPLITS["composite_seen"]["task_names"]
)
COMPOSITE_UNSEEN_TASKS: list[str] = list(
    ROBOCASA365_SPLITS["composite_unseen"]["task_names"]
)
TASK_SPLITS = {
    "atomic_seen": ATOMIC_SEEN_TASKS,
    "composite_seen": COMPOSITE_SEEN_TASKS,
    "composite_unseen": COMPOSITE_UNSEEN_TASKS,
}


def _registry_candidates(registry: Any) -> Iterable[Mapping[str, Any]]:
    """Yield mapping-like task registries from a RoboCasa module."""
    for name in (
        "TASK_GROUPS",
        "TASK_SETS",
        "TASK_SET",
        "BENCHMARK_TASKS",
        "ROBOCASA365_SPLITS",
    ):
        value = getattr(registry, name, None)
        if isinstance(value, Mapping):
            yield value
    for name in ("get_task_groups", "get_task_sets", "get_benchmark_tasks"):
        function = getattr(registry, name, None)
        if callable(function):
            try:
                value = function()
            except TypeError:
                continue
            if isinstance(value, Mapping):
                yield value


def get_task_splits() -> Dict[str, list[str]]:
    """Resolve the standard 50-task set from the installed RoboCasa registry."""
    resolved = {name: list(tasks) for name, tasks in TASK_SPLITS.items()}
    try:
        from robocasa.utils import dataset_registry
    except ImportError:
        return resolved

    for candidate in _registry_candidates(dataset_registry):
        for split_name in resolved:
            value = candidate.get(split_name)
            if isinstance(value, Mapping):
                value = value.get("task_names") or value.get("tasks")
            if isinstance(value, Sequence) and not isinstance(value, str):
                resolved[split_name] = [str(task) for task in value]
    for name, tasks in resolved.items():
        TASK_SPLITS[name][:] = tasks
    return resolved


def _make_random_env(task_name: str, split: str, seed: int):
    """Create a Gymnasium RoboCasa task for the random-rollout check."""
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401  # registers RoboCasa environments
    except ImportError as exc:
        raise RuntimeError(
            "Gymnasium and RoboCasa are required for random rollouts"
        ) from exc
    try:
        return gym.make(task_name, split=split, seed=seed)
    except TypeError:
        return gym.make(task_name, split=split)


def run_random_rollouts_check(
    task_name: str = "robocasa/PickPlaceCounterToCabinet",
    split: str = ROBOCASA_LEADERBOARD_SPLIT,
    num_rollouts: int = 3,
    num_steps: int = 100,
    video_path: str | None = None,
) -> Any:
    """Run RoboCasa's built-in random rollout environment sanity check."""
    try:
        from robocasa.utils.env_utils import run_random_rollouts
    except ImportError as exc:
        raise RuntimeError(
            "RoboCasa env_utils.run_random_rollouts is unavailable"
        ) from exc

    env = _make_random_env(task_name, split, seed=0)
    try:
        kwargs = {
            "num_rollouts": num_rollouts,
            "num_steps": num_steps,
        }
        if video_path is not None:
            kwargs["video_path"] = video_path
        supported = inspect.signature(run_random_rollouts).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
        return run_random_rollouts(env, **kwargs)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def _aggregate_split(task_results: Mapping[str, Mapping[str, Any]]) -> float:
    """Average per-task success rates, matching leaderboard aggregation."""
    if not task_results:
        return 0.0
    return sum(
        float(result["success_rate"]) for result in task_results.values()
    ) / len(task_results)


def _load_diffusion_policy_checkpoint(checkpoint: str, device: str) -> Any:
    """Load a compatible official Diffusion Policy object when available.

    A Lightning ``.ckpt`` containing only a state dict is not sufficient to
    reconstruct the architecture without the official Diffusion Policy
    repository/configuration. Never pass such a checkpoint to PI05 by
    accident; require the caller to provide an instantiated compatible policy.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to load a Diffusion Policy checkpoint"
        ) from exc

    try:
        checkpoint_data = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Diffusion Policy checkpoint {checkpoint!r}"
        ) from exc

    candidates = [checkpoint_data]
    if isinstance(checkpoint_data, Mapping):
        candidates.extend(
            checkpoint_data.get(key)
            for key in ("policy", "model", "ema_model")
            if checkpoint_data.get(key) is not None
        )
    for candidate in candidates:
        if callable(candidate) or any(
            hasattr(candidate, name)
            for name in ("predict", "infer", "select_action")
        ):
            if hasattr(candidate, "to"):
                candidate.to(device)
            if hasattr(candidate, "eval"):
                candidate.eval()
            return candidate
    raise RuntimeError(
        "The Diffusion Policy checkpoint contains weights but no loaded "
        "policy object. Install the official RoboCasa Diffusion Policy "
        "repository and construct its configured policy before calling "
        "run_baseline_eval."
    )


def _build_json_results(
    task_results: Mapping[str, Mapping[str, Any]],
    task_splits: Mapping[str, Sequence[str]],
    checkpoint: str,
    baseline_name: str,
) -> Dict[str, Any]:
    """Build a stable, leaderboard-style JSON result document."""
    split_results: Dict[str, Any] = {}
    for split_name, task_names in task_splits.items():
        per_task = {
            task: dict(task_results[task])
            for task in task_names
            if task in task_results
        }
        split_results[split_name] = {
            "num_tasks": len(per_task),
            "average_success_rate": _aggregate_split(per_task),
            "tasks": per_task,
        }
    overall = _aggregate_split(task_results)
    return {
        "benchmark": "RoboCasa365",
        "baseline": baseline_name,
        "checkpoint": checkpoint,
        "num_rollouts_per_task": NUM_ROLLOUTS_PER_TASK,
        "splits": split_results,
        "overall_success_rate": overall,
        "task_results": dict(task_results),
    }


def run_baseline_eval(
    checkpoint: str,
    baseline_name: str = "pi05",
    task_splits: Mapping[str, Sequence[str]] | None = None,
    num_rollouts: int = NUM_ROLLOUTS_PER_TASK,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    output_path: str | None = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Evaluate a baseline on 50 rollouts per task.

    The pi0.5 path uses the same three-camera, 16-D state, 50x32 action and
    RoboCasa conversion contract as ``EvalRunner``. Diffusion Policy is
    accepted as a backend name so an official adapter can be injected through
    ``PolicyWrapper.policy_factory`` when the benchmark repository is present.
    """
    if not checkpoint:
        raise ValueError("A baseline checkpoint path is required")
    resolved_splits = dict(task_splits or get_task_splits())
    tasks = [
        task
        for split_tasks in resolved_splits.values()
        for task in split_tasks
    ]
    if not tasks:
        raise RuntimeError(
            "No RoboCasa task names were found. Install RoboCasa's dataset "
            "registry or pass task names explicitly."
        )

    baseline_name = baseline_name.lower()
    if baseline_name not in {"pi05", "diffusion_policy", "diffusion"}:
        raise ValueError(f"Unsupported baseline: {baseline_name}")

    if baseline_name in {"diffusion_policy", "diffusion"}:
        loaded_policy = _load_diffusion_policy_checkpoint(checkpoint, device)
        policy = PolicyWrapper(
            checkpoint,
            device=device,
            policy=loaded_policy,
        )
    else:
        policy = PolicyWrapper(checkpoint, device=device)
    config = EvalConfig(
        checkpoint_path=checkpoint,
        task_names=tasks,
        num_rollouts_per_task=num_rollouts,
        max_episode_steps=max_episode_steps,
        output_dir=str(Path(output_path).parent) if output_path else "results",
        device=device,
        split=ROBOCASA_LEADERBOARD_SPLIT,
    )
    runner = EvalRunner(config, policy=policy)
    task_results = runner.run_all()
    results = _build_json_results(
        task_results,
        resolved_splits,
        checkpoint,
        baseline_name,
    )

    target = Path(output_path or "results/robocasa_baseline_results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Baseline results saved to %s", target)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--output", default="results/robocasa_baseline_results.json")
    parser.add_argument("--num-rollouts", type=int, default=NUM_ROLLOUTS_PER_TASK)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-task", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--video-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for key, value in ENV_RENDER_KWARGS.items():
        os.environ.setdefault(key, value)

    if args.step == 0:
        run_random_rollouts_check(
            task_name=args.random_task,
            split=ROBOCASA_LEADERBOARD_SPLIT,
            num_rollouts=3,
            num_steps=min(args.max_steps, 100),
            video_path=args.video_path,
        )
        return

    baseline = args.baseline or ("diffusion_policy" if args.step == 1 else "pi05")
    checkpoint = args.checkpoint or OFFICIAL_BASELINE_CHECKPOINTS[baseline]
    if args.tasks:
        splits = {"custom": args.tasks}
    else:
        splits = get_task_splits()
    run_baseline_eval(
        checkpoint=checkpoint,
        baseline_name=baseline,
        task_splits=splits,
        num_rollouts=args.num_rollouts,
        max_episode_steps=args.max_steps,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
