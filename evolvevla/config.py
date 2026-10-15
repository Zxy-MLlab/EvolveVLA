"""Central configuration for EvolveVLA evaluation and evolution runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Base paths: all project data and checkpoints live on the persistent volume.
PROJECT_ROOT = Path(
    os.environ.get("EVOLVEVLA_ROOT", "/root/autodl-tmp/EvolveVLA")
)
HF_CACHE = Path(
    os.environ.get("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
)
BASELINE_CACHE = Path("/root/autodl-tmp/.cache")

CHECKPOINT_DIR = HF_CACHE / "hub" / "models--lerobot--pi05_base"
TRAJECTORY_DIR = PROJECT_ROOT / "trajectories"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"

# RoboCasa to pi0.5 camera mapping. Keep this order stable.
CAMERA_AGENTVIEW_LEFT = "robot0_agentview_left"
CAMERA_EYE_IN_HAND = "robot0_eye_in_hand"
CAMERA_AGENTVIEW_RIGHT = "robot0_agentview_right"
CAMERA_NAMES = (
    CAMERA_AGENTVIEW_LEFT,
    CAMERA_EYE_IN_HAND,
    CAMERA_AGENTVIEW_RIGHT,
)
POLICY_CAMERA_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

# pi0.5/RoboCasa dimensions.
STATE_DIM_INPUT = 16
STATE_DIM_PADDED = 32
ACTION_DIM_MODEL = 32
# Environment action dimension must be read from env.action_dim at runtime.
ACTION_HORIZON = 50
REPLAN_STEPS = 5
IMAGE_SIZE = (224, 224)

# RoboCasa365 standard evaluation splits.
ROBOCASA365_SPLITS = {
    "atomic_seen": {
        "description": "18 atomic tasks seen during pretraining",
        "num_tasks": 18,
        "task_names": [],
    },
    "composite_seen": {
        "description": "16 composite tasks seen during pretraining",
        "num_tasks": 16,
        "task_names": [],
    },
    "composite_unseen": {
        "description": "16 composite tasks held out from pretraining",
        "num_tasks": 16,
        "task_names": [],
    },
}

# The official checkpoint repository contains substantially more than the
# selected files. These paths intentionally point at individual baseline
# artifacts under the persistent cache rather than the overlay filesystem.
DIFFUSION_POLICY_CHECKPOINT_TEMPLATE = str(
    BASELINE_CACHE
    / "robocasa365_checkpoints"
    / "diffusion_policy"
    / "17.40.09_train_diffusion_transformer_hybrid_pretrain_human300"
    / "checkpoints"
    / "epoch=0500-test_mean_score=-1.000.ckpt"
)
PI05_ROBOCASA_CHECKPOINT_TEMPLATE = str(
    BASELINE_CACHE
    / "robocasa365_checkpoints"
    / "pi05_pretrain_human300"
    / "multitask_learning"
    / "75000"
)
PI0_ROBOCASA_CHECKPOINT_TEMPLATE = str(
    BASELINE_CACHE
    / "robocasa365_checkpoints"
    / "pi0"
    / "pi0_robocasa_pretrain_human300"
    / "multitask_learning"
    / "75000"
)
PI05_CHECKPOINT = str(CHECKPOINT_DIR)
OFFICIAL_BASELINE_CHECKPOINTS = {
    "diffusion_policy": DIFFUSION_POLICY_CHECKPOINT_TEMPLATE,
    "pi0": PI0_ROBOCASA_CHECKPOINT_TEMPLATE,
    "pi05": PI05_ROBOCASA_CHECKPOINT_TEMPLATE,
}

NUM_ROLLOUTS_PER_TASK = 50
MAX_EPISODE_STEPS = 1000
PI05_DEVICE = "cuda:0"

ENV_RENDER_KWARGS = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
}


@dataclass
class EvolutionConfig:
    """Configuration for one evolution cycle."""

    cycle_id: int = 0
    base_checkpoint: str = PI05_CHECKPOINT
    num_rollouts: int = NUM_ROLLOUTS_PER_TASK
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    batch_size: int = 4
    # RoboCasa split used for leaderboard-comparable evaluation.
    eval_split: str = "pretrain"
    diagnosis_model: str = ""
    min_failure_samples: int = 10
    num_demos_per_failure: int = 20
    output_dir: str = ""

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(PROJECT_ROOT / f"cycle_{self.cycle_id:03d}")
