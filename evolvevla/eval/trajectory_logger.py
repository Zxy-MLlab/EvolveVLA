"""Structured trajectory logging for failure analysis.

Records observations, actions, rewards, and metadata per step.
Supports both full trajectory storage (for failure analysis) and
lightweight metric-only logging (for aggregate statistics).

Outputs: JSON files per failed episode, summary CSVs per task
"""

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Single timestep record."""

    step: int
    action: Optional[List[float]] = None
    reward: float = 0.0
    done: bool = False
    success: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    # Observation metadata (not raw pixels).
    robot_state: Optional[List[float]] = None
    gripper_state: Optional[float] = None
    # Failure indicators.
    collision: bool = False
    out_of_bounds: bool = False
    timeout: bool = False


@dataclass
class EpisodeRecord:
    """Full episode trajectory record."""

    task_name: str
    episode_idx: int
    seed: int
    success: bool = False
    total_reward: float = 0.0
    num_steps: int = 0
    wall_time_seconds: float = 0.0
    steps: List[StepRecord] = field(default_factory=list)
    failure_mode: str = ""  # populated by failure_analyzer
    failure_step: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    """Log evaluation trajectories to disk for failure analysis."""

    def __init__(self, output_dir: str, task_name: str):
        self.output_dir = Path(output_dir) / task_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.current_episode: Optional[EpisodeRecord] = None
        self._start_time = 0.0

    def begin_episode(self, episode_idx: int, seed: int) -> None:
        """Start recording a new episode."""
        self.current_episode = EpisodeRecord(
            task_name=self.task_name,
            episode_idx=episode_idx,
            seed=seed,
        )
        self._start_time = time.time()

    def log_step(
        self,
        step: int,
        action: Any,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        robot_state: Any = None,
        gripper_state: Any = None,
    ) -> None:
        """Record one environment step."""
        if self.current_episode is None:
            raise RuntimeError("Call begin_episode first")

        record = StepRecord(
            step=step,
            action=action.tolist() if hasattr(action, "tolist") else action,
            reward=float(reward),
            done=bool(done),
            success=bool(info.get("success", False)),
            info={
                key: value
                for key, value in info.items()
                if isinstance(value, (int, float, str, bool))
            },
            robot_state=(
                robot_state.tolist()
                if hasattr(robot_state, "tolist")
                else robot_state
            ),
            gripper_state=(
                float(gripper_state) if gripper_state is not None else None
            ),
            collision=bool(info.get("collision", False)),
            out_of_bounds=bool(info.get("out_of_bounds", False)),
            timeout=(
                bool(done)
                and not bool(info.get("success", False))
                and step >= info.get("horizon", float("inf")) - 1
            ),
        )
        self.current_episode.steps.append(record)

    def end_episode(self, success: bool, total_reward: float) -> EpisodeRecord:
        """Finalize and save an episode record."""
        if self.current_episode is None:
            raise RuntimeError("No active episode")

        episode = self.current_episode
        episode.success = bool(success)
        episode.total_reward = float(total_reward)
        episode.num_steps = len(episode.steps)
        episode.wall_time_seconds = time.time() - self._start_time

        # Save full trajectories for failed episodes.
        if not episode.success:
            self._save_episode(episode)

        # Always append to the aggregate summary.
        self._append_summary(episode)

        self.current_episode = None
        return episode

    def _save_episode(self, episode: EpisodeRecord) -> None:
        """Save a full trajectory to a JSON file."""
        filepath = self.output_dir / f"episode_{episode.episode_idx:04d}.json"
        with filepath.open("w") as handle:
            json.dump(asdict(episode), handle, indent=2)

    def _append_summary(self, episode: EpisodeRecord) -> None:
        """Append an episode summary to CSV."""
        filepath = self.output_dir / "summary.csv"
        write_header = not filepath.exists()

        row = {
            "episode": episode.episode_idx,
            "success": episode.success,
            "total_reward": episode.total_reward,
            "num_steps": episode.num_steps,
            "wall_time_s": round(episode.wall_time_seconds, 2),
            "failure_mode": episode.failure_mode,
            "failure_step": episode.failure_step,
        }

        with filepath.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
