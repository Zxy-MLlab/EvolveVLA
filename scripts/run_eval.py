#!/usr/bin/env python3
"""Run pi0.5 evaluation on RoboCasa365 tasks.

Usage:
    python scripts/run_eval.py --split atomic_seen --num-rollouts 50
    python scripts/run_eval.py --tasks PnPCounterToCab --num-rollouts 10
"""

import argparse
import logging
import os
import sys

# Ensure the project root is on the Python path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolvevla.config import (  # noqa: E402
    ENV_RENDER_KWARGS,
    MAX_EPISODE_STEPS,
    NUM_ROLLOUTS_PER_TASK,
    PI05_CHECKPOINT,
    ROBOCASA365_SPLITS,
    TRAJECTORY_DIR,
)
from evolvevla.eval.eval_runner import EvalConfig, EvalRunner  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="EvolveVLA Evaluation Runner"
    )
    parser.add_argument(
        "--checkpoint",
        default=PI05_CHECKPOINT,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--split",
        default="atomic_seen",
        choices=list(ROBOCASA365_SPLITS.keys()),
        help="RoboCasa365 evaluation split",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific task names (overrides --split)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=NUM_ROLLOUTS_PER_TASK,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_EPISODE_STEPS,
    )
    parser.add_argument(
        "--output-dir",
        default=str(TRAJECTORY_DIR),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    # Set rendering environment variables before environment construction.
    for key, value in ENV_RENDER_KWARGS.items():
        os.environ.setdefault(key, value)

    if args.tasks:
        task_names = args.tasks
    else:
        split = ROBOCASA365_SPLITS[args.split]
        task_names = split["task_names"]
        if not task_names:
            logging.error(
                "Split '%s' has no tasks populated yet. "
                "Use --tasks to specify tasks manually.",
                args.split,
            )
            sys.exit(1)

    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        task_names=task_names,
        num_rollouts_per_task=args.num_rollouts,
        max_episode_steps=args.max_steps,
        save_trajectory=True,
        save_video=args.save_video,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )

    results = EvalRunner(config).run_all()

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for task, result in results.items():
        print(
            f"  {task}: {result['success_rate']:.1%} "
            f"({result['successes']}/{result['num_rollouts']})"
        )

    total_successes = sum(result["successes"] for result in results.values())
    total_rollouts = sum(
        result["num_rollouts"] for result in results.values()
    )
    overall = total_successes / max(total_rollouts, 1)
    print(f"\n  Overall: {overall:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
