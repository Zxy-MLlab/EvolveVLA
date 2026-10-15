#!/usr/bin/env python3
"""RoboTwin pi0.5 adapter for public OpenPI PyTorch and Orbax checkpoints.

This keeps OpenPI's internal action representation private. The checkpoint's
RoboTwin statistics and Aloha transforms convert the first 14 output
dimensions back to the simulator's joint action convention. Public fine-tunes
use different action horizons, so the horizon always comes from the matching
OpenPI model config.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# The checkpoint metadata carries this as a default prompt. OpenPI only injects
# defaults when an observation has no prompt; RoboTwin always supplies one.
# Keep the text available for a controlled compatibility A/B, but do not
# concatenate it unless explicitly requested.
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


@dataclasses.dataclass(frozen=True)
class _DataConfigOverride:
    """Override serialized training settings missing from the public code config."""

    delegate: Any
    use_quantile_norm: bool

    def create(self, assets_dirs: Path, model_config: Any) -> Any:
        created = self.delegate.create(assets_dirs, model_config)
        return dataclasses.replace(
            created,
            use_quantile_norm=self.use_quantile_norm,
        )


class PublicPi05Adapter:
    """Load and run the public PyTorch OpenPI checkpoint on RoboTwin observations."""

    def __init__(
        self,
        checkpoint: str | os.PathLike[str],
        device: str = "cuda:0",
        prompt_prefix: str = "none",
        normalization: str = "quantile",
        policy_seed: int = 0,
        noise_source: str = "torch",
    ):
        import openpi.transforms as _transforms
        from openpi.models import pi0_config
        from openpi.policies import policy_config
        from openpi.training import config as training_config

        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        if prompt_prefix not in {"none", "scene"}:
            raise ValueError("prompt_prefix must be 'none' or 'scene'")
        if normalization not in {"zscore", "quantile"}:
            raise ValueError("normalization must be 'zscore' or 'quantile'")
        if noise_source not in {"jax", "torch", "internal"}:
            raise ValueError("noise_source must be 'jax', 'torch', or 'internal'")
        self.prompt_prefix = prompt_prefix
        self.normalization = normalization
        self.policy_seed = int(policy_seed)
        self.noise_source = noise_source
        has_pytorch_weights = (self.checkpoint / "model.safetensors").is_file()
        has_orbax_weights = (self.checkpoint / "params").is_dir()
        if has_pytorch_weights == has_orbax_weights:
            raise FileNotFoundError(
                f"expected exactly one OpenPI weight format under {self.checkpoint}: "
                "model.safetensors or params/"
            )
        self.checkpoint_format = "openpi_pytorch" if has_pytorch_weights else "openpi_orbax"
        asset_root = self.checkpoint / "assets"
        norm_paths = sorted(asset_root.glob("*/norm_stats.json"))
        if len(norm_paths) != 1:
            raise RuntimeError(
                f"expected one normalization asset under {asset_root}, got {norm_paths}"
            )
        asset_id = norm_paths[0].parent.name
        norm_path = asset_root / asset_id / "norm_stats.json"

        base = training_config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")
        if has_pytorch_weights:
            model = dataclasses.replace(
                base.model,
                action_dim=32,
                action_horizon=32,
                max_token_len=200,
                pi05=True,
                discrete_state_input=True,
                # The converted PyTorch config defaults to max-autotune, which
                # spends several minutes benchmarking kernels on first use.
                pytorch_compile_mode=None,
            )
            data_factory = dataclasses.replace(
                base.data,
                repo_id="clean_randomized_joint_training",
                assets=dataclasses.replace(
                    base.data.assets,
                    assets_dir=str(asset_root),
                    asset_id=asset_id,
                ),
                adapt_to_pi=True,
                use_delta_joint_actions=True,
                default_prompt=SCENE_PREFIX,
            )
            repack = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "high_image",
                                "cam_left_wrist": "left_wrist_image",
                                "cam_right_wrist": "right_wrist_image",
                            },
                            "state": "state",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        else:
            # This is the exact config used by the official RoboTwin Orbax
            # release. In particular, it uses 50-step chunks and does not
            # apply the Aloha-to-PI joint sign/gripper conversion.
            model = base.model
            model_config = self.checkpoint / "model_config.json"
            if model_config.is_file():
                verification = json.loads((self.checkpoint / "verification.json").read_text())
                if (verification.get("status") != "passed"
                        or verification.get("kind") != "full_lora_model_unmerged"
                        or verification.get("production_restore_exact") is not True
                        or verification.get("normalization_exact") is not True):
                    raise ValueError("LoRA checkpoint verification missing")
                recorded = json.loads(model_config.read_text())
                if (recorded.get("paligemma_variant") != "gemma_2b_lora"
                        or recorded.get("action_expert_variant") != "gemma_300m_lora"
                        or recorded.get("action_horizon") != 50
                        or recorded.get("action_dim") != 32):
                    raise ValueError("Unexpected LoRA model configuration")
                model = dataclasses.replace(base.model, **recorded)
            data_factory = dataclasses.replace(
                base.data,
                assets=dataclasses.replace(
                    base.data.assets,
                    assets_dir=str(asset_root),
                    asset_id=asset_id,
                ),
            )
            repack = None
        self.action_horizon = int(model.action_horizon)
        self.model_action_dim = int(model.action_dim)
        # metadata.pt serializes False in the factory's base DataConfig, but
        # LeRobotAlohaDataConfig.create() overrides it to True for PI0.5.  A
        # same-seed rollout confirmed that z-score produces invalid joint
        # targets, so quantile is the effective training/inference setting.
        data = _DataConfigOverride(
            data_factory,
            use_quantile_norm=(normalization == "quantile"),
        )
        self.config = dataclasses.replace(
            base,
            model=model,
            data=data,
            wandb_enabled=False,
        )
        self.policy = policy_config.create_trained_policy(
            self.config,
            self.checkpoint,
            repack_transforms=repack,
            pytorch_device=device,
        )
        # Both reference paths start every task process from seed/key 0. The
        # converted PyTorch path otherwise inherits an OS-random seed.
        if has_pytorch_weights:
            import torch

            torch.manual_seed(self.policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.policy_seed)
        else:
            import jax

            self.policy._rng = jax.random.key(self.policy_seed)
        if self.noise_source == "jax":
            import jax

            self._jax_rng = jax.random.key(self.policy_seed)

    def advance_rng(self, num_inferences: int) -> None:
        """Restore RNG state after resuming completed policy inferences."""
        if num_inferences < 0:
            raise ValueError("num_inferences must be non-negative")
        if num_inferences == 0:
            return
        if self.checkpoint_format == "openpi_pytorch" and self.noise_source == "torch":
            raise RuntimeError(
                "deterministic resume is not implemented for PyTorch-generated noise"
            )

        import jax

        def advance(key: Any) -> Any:
            for _ in range(num_inferences):
                key, _ = jax.random.split(key)
            return key

        # JAX Policy.infer advances its private key even when explicit noise is
        # supplied. Explicit JAX noise uses a second, independent key stream.
        if self.checkpoint_format == "openpi_orbax":
            self.policy._rng = advance(self.policy._rng)
        if self.noise_source == "jax":
            self._jax_rng = advance(self._jax_rng)

    def infer_chunk(
        self,
        *,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        # OpenPI's AlohaInputs consumes channel-first uint8 images. The public
        # PyTorch fine-tune serialized a repack transform with flat image keys;
        # the native RoboTwin Orbax config consumes the images mapping directly.
        chw_images = {
            key: np.transpose(np.asarray(value), (2, 0, 1))
            for key, value in images.items()
        }
        if self.checkpoint_format == "openpi_orbax":
            obs = {
                "images": chw_images,
                "state": np.asarray(state, dtype=np.float32),
                "prompt": str(prompt),
            }
        else:
            obs = {
                "high_image": chw_images["cam_high"],
                "left_wrist_image": chw_images["cam_left_wrist"],
                "right_wrist_image": chw_images["cam_right_wrist"],
                "state": np.asarray(state, dtype=np.float32),
                "prompt": str(prompt),
            }
        prompt_text = str(prompt)
        if self.prompt_prefix == "scene" and not prompt_text.startswith(SCENE_PREFIX):
            prompt_text = SCENE_PREFIX + prompt_text
        obs["prompt"] = prompt_text
        noise = None
        if self.noise_source == "jax":
            import jax

            self._jax_rng, sample_rng = jax.random.split(self._jax_rng)
            noise = np.asarray(
                jax.random.normal(
                    sample_rng,
                    (self.action_horizon, self.model_action_dim),
                ),
                dtype=np.float32,
            )
        result = self.policy.infer(obs, noise=noise)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"public policy output must be [horizon,14], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("public policy returned NaN/Inf actions")
        return actions


def checkpoint_summary(checkpoint: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(checkpoint)
    norm_paths = sorted((root / "assets").glob("*/norm_stats.json"))
    if len(norm_paths) != 1:
        raise RuntimeError(f"expected one normalization asset under {root / 'assets'}")
    norm = norm_paths[0]
    stats = json.loads(norm.read_text())
    pytorch_weights = root / "model.safetensors"
    checkpoint_format = "openpi_pytorch" if pytorch_weights.is_file() else "openpi_orbax"
    model_config = root / "model_config.json"
    action_horizon = (json.loads(model_config.read_text())["action_horizon"]
                      if model_config.is_file() else
                      (32 if checkpoint_format == "openpi_pytorch" else 50))
    if checkpoint_format == "openpi_pytorch":
        model_bytes = pytorch_weights.stat().st_size
    else:
        model_bytes = sum(path.stat().st_size for path in (root / "params").rglob("*") if path.is_file())
    return {
        "checkpoint": str(root.resolve()),
        "checkpoint_format": checkpoint_format,
        "model_bytes": model_bytes,
        "normalization_asset": str(norm.relative_to(root)),
        "state_dim": len(stats["norm_stats"]["state"]["mean"]),
        "action_dim": len(stats["norm_stats"]["actions"]["mean"]),
        "action_horizon": action_horizon,
    }
