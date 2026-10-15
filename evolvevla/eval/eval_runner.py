"""Evaluation runner for pi0.5 policies on RoboCasa environments.

The runner implements the RoboCasa benchmark observation/action contract:

* three camera streams mapped to pi0.5's base and wrist feature names;
* 224x224 resize-with-padding with HWC uint8 RGB images;
* a 16-dimensional state vector;
* 50-step model chunks with a five-step receding-horizon execution window;
* RoboCasa's ``convert_action`` for the runtime environment action dimension.

The heavy RoboCasa and LeRobot imports are intentionally lazy so that this
module remains importable for configuration and syntax checks on a fresh
machine.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping

import numpy as np

from evolvevla.config import (
    ACTION_DIM_MODEL,
    ACTION_HORIZON,
    CAMERA_AGENTVIEW_LEFT,
    CAMERA_AGENTVIEW_RIGHT,
    CAMERA_EYE_IN_HAND,
    CAMERA_NAMES,
    IMAGE_SIZE,
    MAX_EPISODE_STEPS,
    REPLAN_STEPS,
    STATE_DIM_INPUT,
)

logger = logging.getLogger(__name__)


def resize_with_pad(
    image: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Resize an image without changing aspect ratio and pad to target size.

    Args:
        image: HWC RGB input. Values are preserved as uint8.
        target_size: ``(height, width)`` output size.

    Returns:
        HWC uint8 RGB image with exactly ``target_size`` spatial dimensions.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)

    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Invalid target size: {target_size}")

    from PIL import Image

    source_height, source_width = array.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = np.asarray(
        Image.fromarray(array, mode="RGB").resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )

    output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


def _as_numpy(value: Any) -> np.ndarray:
    """Convert torch/numpy-like values to a detached numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _extract_step(result: Any) -> tuple[Mapping[str, Any], float, bool, Dict[str, Any]]:
    """Normalize Gym and Gymnasium step return conventions."""
    if not isinstance(result, (tuple, list)) or len(result) not in (4, 5):
        raise RuntimeError(
            "Environment step must return 4 or 5 values "
            "(obs, reward, done[, truncated], info)"
        )
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info or {})
    obs, reward, terminated, truncated, info = result
    return obs, float(reward), bool(terminated or truncated), dict(info or {})


def _extract_reset(result: Any) -> Mapping[str, Any]:
    """Normalize RoboSuite and Gymnasium reset return conventions."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0]
    return result


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""

    checkpoint_path: str = ""
    task_names: List[str] = field(default_factory=list)
    num_rollouts_per_task: int = 50
    max_episode_steps: int = MAX_EPISODE_STEPS
    render: bool = False
    save_video: bool = False
    save_trajectory: bool = True
    output_dir: str = "trajectories"
    seed: int = 42
    device: str = "cuda:0"
    prompt_template: str = "{task_name}"
    # pretrain is required for RoboCasa leaderboard comparability.
    split: str = "pretrain"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


class PolicyWrapper:
    """Generic policy adapter with a LeRobot PI05 implementation.

    ``predict`` always accepts a normalized policy observation and returns a
    numpy action chunk. The preferred LeRobot path uses ``PI05Policy`` and its
    ``predict_action_chunk`` method. A callable or an object exposing
    ``predict``, ``infer``, or ``predict_action_chunk`` can also be injected
    for tests and alternate policy backends.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        policy: Any | None = None,
        policy_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.policy = policy
        self.policy_factory = policy_factory
        self._processor = None

    def load(self) -> None:
        """Load a LeRobot PI05 policy or the configured generic backend."""
        if self.policy is not None:
            return
        if not self.checkpoint_path:
            raise ValueError("A policy checkpoint path is required")

        try:
            if self.policy_factory is not None:
                self.policy = self.policy_factory(
                    self.checkpoint_path,
                    self.device,
                )
            else:
                from lerobot.policies.pi05 import PI05Policy

                self.policy = PI05Policy.from_pretrained(self.checkpoint_path)
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot PI05Policy is unavailable. Install LeRobot with its "
                "pi0/transformers dependencies or inject a policy_factory."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not load PI05 policy from {self.checkpoint_path!r}"
            ) from exc

        if hasattr(self.policy, "to"):
            self.policy.to(self.device)
        if hasattr(self.policy, "eval"):
            self.policy.eval()

        # Newer LeRobot releases expose processors separately. If present,
        # use them; otherwise PI05Policy receives an already prepared batch.
        processor = getattr(self.policy, "processor", None)
        if processor is not None:
            self._processor = processor

    def reset(self) -> None:
        """Clear policy-side action/history state between episodes."""
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def _prepare_lerobot_batch(
        self,
        observation: Mapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Build a LeRobot-compatible single-observation batch.

        Images remain HWC uint8 until LeRobot's own processor/model path
        handles conversion. A processor is preferred when the installed
        release provides one.
        """
        batch: MutableMapping[str, Any] = {
            "observation.images.base_0_rgb": observation["base_0_rgb"],
            "observation.images.left_wrist_0_rgb": observation[
                "left_wrist_0_rgb"
            ],
            "observation.images.right_wrist_0_rgb": observation[
                "right_wrist_0_rgb"
            ],
            "observation.state": observation["state"],
            "task": [str(observation["prompt"])],
        }
        if self._processor is not None:
            try:
                processed = self._processor(batch)
                if isinstance(processed, Mapping):
                    return dict(processed)
            except (TypeError, KeyError, ValueError) as exc:
                logger.debug("LeRobot processor fallback: %s", exc)
        return batch

    def _call_policy(self, batch: Mapping[str, Any]) -> Any:
        """Call the first compatible generic policy method."""
        if callable(self.policy):
            return self.policy(batch)
        for method_name in (
            "predict_action_chunk",
            "predict",
            "infer",
            "select_action",
        ):
            method = getattr(self.policy, method_name, None)
            if method is not None:
                return method(batch)
        raise TypeError(
            "Policy must be callable or expose predict_action_chunk, predict, "
            "infer, or select_action"
        )

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Predict a 50-step, 32-dimensional model action chunk."""
        if self.policy is None:
            self.load()
        batch = self._prepare_lerobot_batch(observation)
        try:
            output = self._call_policy(batch)
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                "Policy inference failed. Verify checkpoint feature names, "
                "task prompt, and LeRobot preprocessing configuration."
            ) from exc

        if isinstance(output, Mapping):
            for key in ("actions", "action", "predicted_actions"):
                if key in output:
                    output = output[key]
                    break
        actions = _as_numpy(output)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(
                f"Expected a 2D action chunk, got shape {actions.shape}"
            )
        if actions.shape[0] < REPLAN_STEPS:
            raise ValueError(
                f"Policy returned only {actions.shape[0]} action steps; "
                f"need at least {REPLAN_STEPS}"
            )
        if actions.shape[1] != ACTION_DIM_MODEL:
            logger.warning(
                "Policy returned %d dimensions instead of configured %d; "
                "EvalRunner will slice actions to env.action_dim at runtime",
                actions.shape[1],
                ACTION_DIM_MODEL,
            )
        return actions


class EvalRunner:
    """Run RoboCasa evaluation rollouts and collect trajectories."""

    def __init__(self, config: EvalConfig, policy: PolicyWrapper | None = None):
        self.config = config
        self.policy = policy
        self.action_dim: int | None = None

    def setup(self) -> None:
        """Initialize the policy."""
        logger.info("Setting up evaluation with config: %s", asdict(self.config))
        if self.policy is None:
            self.policy = PolicyWrapper(
                self.config.checkpoint_path,
                self.config.device,
            )
        self.policy.load()

    def create_env(self, task_name: str):
        """Create a RoboCasa Gymnasium environment for leaderboard eval."""
        try:
            import gymnasium as gym
            import robocasa  # noqa: F401  # registers RoboCasa task environments
        except ImportError as exc:
            raise RuntimeError(
                "Gymnasium/RoboCasa is unavailable; install the environment "
                "before creating evaluation rollouts."
            ) from exc

        env_kwargs = dict(self.config.env_kwargs)
        env_id = (
            task_name
            if task_name.startswith("robocasa/")
            else f"robocasa/{task_name}"
        )
        try:
            env = gym.make(env_id, split=self.config.split, **env_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not create RoboCasa task environment {task_name!r}"
            ) from exc
        action_dim = getattr(env, "action_dim", None)
        if action_dim is None:
            close = getattr(env, "close", None)
            if close is not None:
                close()
            raise RuntimeError(
                f"RoboCasa task environment {task_name!r} has no action_dim"
            )
        self.action_dim = int(action_dim)
        logger.info("Using runtime action_dim=%d for %s", self.action_dim, task_name)
        return env

    @staticmethod
    def _state_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
        """Extract the required 16-dimensional RoboCasa state."""
        required = (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_gripper_qpos",
        )
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Observation is missing state keys: {missing}")
        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"]),
                np.asarray(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_base_pos"]),
                np.asarray(obs["robot0_base_quat"]),
                np.asarray(obs["robot0_gripper_qpos"]),
            ]
        ).astype(np.float32, copy=False)
        if state.shape != (STATE_DIM_INPUT,):
            raise ValueError(
                f"Expected {STATE_DIM_INPUT}-D state, got shape {state.shape}"
            )
        return state

    def _policy_observation(
        self,
        obs: Mapping[str, Any],
        task_name: str,
    ) -> Dict[str, Any]:
        """Map a RoboCasa observation to the pi0.5 policy contract."""
        camera_map = {
            "base_0_rgb": CAMERA_AGENTVIEW_LEFT,
            "left_wrist_0_rgb": CAMERA_EYE_IN_HAND,
            "right_wrist_0_rgb": CAMERA_AGENTVIEW_RIGHT,
        }
        missing = [key for key in camera_map.values() if key not in obs]
        if missing:
            raise KeyError(f"Observation is missing camera keys: {missing}")
        prompt = obs.get("annotation.human.task_description", task_name)
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0]
        return {
            role: resize_with_pad(np.asarray(obs[camera_key]))
            for role, camera_key in camera_map.items()
        } | {
            "state": self._state_from_obs(obs),
            "prompt": self.config.prompt_template.format(
                task_name=str(prompt),
            ),
        }

    def _convert_action(self, action: Any) -> Any:
        """Convert one flat action using RoboCasa's official helper."""
        if self.action_dim is None:
            raise RuntimeError("Environment action_dim has not been initialized")
        vector = np.asarray(action)
        if vector.ndim != 1:
            raise ValueError(f"Expected a 1D action vector, got shape {vector.shape}")
        if vector.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, got {vector.shape[-1]}"
            )
        try:
            from robocasa.utils.env_utils import convert_action
        except ImportError as exc:
            raise RuntimeError(
                "RoboCasa env_utils.convert_action is unavailable"
            ) from exc
        return convert_action(vector)

    def run_episode(
        self,
        env: Any,
        task_name: str,
        episode_idx: int,
    ) -> Dict[str, Any]:
        """Run one receding-horizon episode."""
        if self.policy is None:
            raise RuntimeError("Call setup() before running episodes")

        env_action_dim = getattr(env, "action_dim", None)
        if env_action_dim is None:
            raise RuntimeError(f"Environment for {task_name!r} has no action_dim")
        env_action_dim = int(env_action_dim)
        if self.action_dim is None:
            self.action_dim = env_action_dim
        elif self.action_dim != env_action_dim:
            raise RuntimeError(
                f"Configured action_dim {self.action_dim} does not match "
                f"environment action_dim {env_action_dim}"
            )

        trajectory: List[Dict[str, Any]] = []
        obs = _extract_reset(env.reset())
        self.policy.reset()
        action_queue: deque[np.ndarray] = deque()
        total_reward = 0.0
        success = False
        num_steps = 0

        for step in range(self.config.max_episode_steps):
            if not action_queue:
                policy_obs = self._policy_observation(obs, task_name)
                action_chunk = self.policy.predict(policy_obs)
                if action_chunk.shape[1] < env_action_dim:
                    raise ValueError(
                        f"Expected at least environment action dimension "
                        f"{env_action_dim}, got {action_chunk.shape[1]}"
                    )
                action_queue.extend(
                    np.asarray(action_chunk[:ACTION_HORIZON, :env_action_dim]).copy()
                )
                if len(action_queue) > REPLAN_STEPS:
                    action_queue = deque(
                        list(action_queue)[:REPLAN_STEPS],
                    )

            action = np.asarray(action_queue.popleft())
            assert action.shape[-1] == env.action_dim, (
                f"Action dimension {action.shape[-1]} does not match "
                f"environment action_dim {env.action_dim}"
            )
            env_action = self._convert_action(action)
            next_obs, reward, done, info = _extract_step(env.step(env_action))
            total_reward += reward
            num_steps = step + 1
            success = bool(info.get("success", False))

            if self.config.save_trajectory:
                trajectory.append(
                    {
                        "step": step,
                        "reward": reward,
                        "done": done,
                        "success": success,
                        "info": {
                            key: value
                            for key, value in info.items()
                            if isinstance(value, (int, float, str, bool))
                        },
                    }
                )

            obs = next_obs
            if success or done:
                break

        result: Dict[str, Any] = {
            "task": task_name,
            "episode": episode_idx,
            "success": success,
            "total_reward": total_reward,
            "num_steps": num_steps,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.config.save_trajectory and not success:
            result["trajectory"] = trajectory
        return result

    def run_task(self, task_name: str) -> Dict[str, Any]:
        """Run all configured rollouts for one task."""
        logger.info(
            "Evaluating task: %s (%d rollouts)",
            task_name,
            self.config.num_rollouts_per_task,
        )
        env = self.create_env(task_name)
        episodes: List[Dict[str, Any]] = []
        successes = 0
        try:
            for episode_idx in range(self.config.num_rollouts_per_task):
                result = self.run_episode(env, task_name, episode_idx)
                episodes.append(result)
                successes += int(result["success"])
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

        rollouts = max(self.config.num_rollouts_per_task, 1)
        return {
            "task": task_name,
            "num_rollouts": self.config.num_rollouts_per_task,
            "successes": successes,
            "success_rate": successes / rollouts,
            "episodes": episodes,
        }

    def run_all(self) -> Dict[str, Any]:
        """Evaluate every configured task and save JSON results."""
        self.setup()
        all_results: Dict[str, Any] = {}
        for task_name in self.config.task_names:
            all_results[task_name] = self.run_task(task_name)
            logger.info(
                "Task %s: %.1f%% success rate",
                task_name,
                100.0 * all_results[task_name]["success_rate"],
            )

        output_path = Path(self.config.output_dir) / "eval_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        logger.info("Results saved to %s", output_path)
        return all_results
