"""AI-driven failure diagnosis for VLA evaluation trajectories.

The analyzer turns failed rollout JSON files into structured diagnoses.  It is
deliberately independent of a particular simulator or provider SDK: trajectory
feature extraction supports the project's ``TrajectoryLogger`` schema and
common nested trajectory variants, while the model call can be connected to
the Agent-level Provider API relay when a project-scoped provider credential
is configured.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Canonical failure mode taxonomy for RoboCasa kitchen tasks. The descriptions
# are written to support one primary label per episode.
FAILURE_MODES = {
    "grasp_failure": "No stable grasp is established on the target before lift or transport",
    "grasp_slip": "A previously established grasp loses the object during manipulation",
    "navigation_error": "The robot fails to reach the intended approach or manipulation location",
    "collision": "The robot makes an unsafe collision or contact with the environment or an object",
    "wrong_object": "The robot selects or manipulates an object other than the target",
    "wrong_receptacle": "The robot places or attempts to place the target in the wrong receptacle",
    "sequence_error": "The robot performs required subtasks in an invalid order",
    "timeout": "The episode ends because the time or step budget is exhausted",
    "partial_completion": "The episode ends with required subtasks unfinished and no earlier specific cause is supported",
    "unknown": "The available trajectory evidence does not support a more specific label",
}

FAILURE_ANALYSIS_PROMPT = """You are an expert robot manipulation analyst. Analyze the following failed robot manipulation trajectory and provide a structured diagnosis.

## Task
{task_description}

## Trajectory Summary
- Total steps: {num_steps}
- Final reward: {total_reward}
- Episode terminated by: {termination_reason}

## Key Events Timeline
{events_timeline}

## Robot State at Failure
- End-effector position: {ee_position}
- Gripper state: {gripper_state}
- Base position: {base_position}

## Reward Curve
{reward_curve_summary}

## Instructions
Classify this failure into exactly ONE primary category:
{failure_categories}

Use only evidence present in the trajectory summary. If several categories
could apply, choose the earliest causal failure that best explains the
unfinished task. Use "partial_completion" only when the task is unfinished
but no earlier specific failure is supported. Use "timeout" only when the
episode ended because its time or step budget was exhausted. Use "unknown"
when the evidence is insufficient. Set failure_step to the first step where
the selected failure manifested, or -1 when that step cannot be identified.

Respond in this exact JSON format:
{{
  "failure_mode": "<category_name>",
  "confidence": <0.0-1.0>,
  "failure_step": <step_number>,
  "description": "<what happened>",
  "root_cause": "<why it happened>",
  "suggested_data": "<what training data would help>"
}}
"""

_DEMONSTRATION_DESCRIPTIONS = {
    "grasp_failure": (
        "Show complete approach, alignment, gripper closure, and lift verification "
        "for the target object, including recovery from near-miss grasps."
    ),
    "grasp_slip": (
        "Show stable grasps through lifting and transport with varied object poses, "
        "contact points, gripper widths, and motion speeds."
    ),
    "navigation_error": (
        "Show collision-free navigation from varied starting poses to the correct "
        "workspace location while preserving the manipulation plan."
    ),
    "collision": (
        "Show safe approach and retreat trajectories around obstacles and nearby "
        "objects, including tight-clearance and recovery scenarios."
    ),
    "wrong_object": (
        "Show visual identification and selection of the target among distractors "
        "with varied layouts, colors, and object proximity."
    ),
    "wrong_receptacle": (
        "Show placement into the correct receptacle with varied receptacle poses, "
        "approach directions, and final alignment checks."
    ),
    "sequence_error": (
        "Show the required subtask order from observation through placement, "
        "including demonstrations with intermediate state verification."
    ),
    "timeout": (
        "Show efficient end-to-end executions with varied initial states and "
        "demonstrate decisive actions near the time limit."
    ),
    "partial_completion": (
        "Show full multi-stage task completion and transitions between subtasks, "
        "including recovery after an intermediate stage succeeds."
    ),
    "unknown": (
        "Show diverse successful and failed trajectories with explicit state and "
        "outcome annotations to improve failure disambiguation."
    ),
}

_SCENARIO_HINTS = {
    "grasp_failure": [
        "grasp from different angles",
        "objects at varied distances and heights",
        "near-miss approach followed by correction",
    ],
    "grasp_slip": [
        "lift fragile or low-friction objects",
        "transport with different gripper widths",
        "vary object mass, pose, and motion speed",
    ],
    "navigation_error": [
        "navigate from different starting poses",
        "reach the target around obstacles",
        "recover from an initially incorrect heading",
    ],
    "collision": [
        "navigate around obstacles",
        "approach objects in tight clearances",
        "recover safely after contact detection",
    ],
    "wrong_object": [
        "select the target among distractors",
        "vary object colors and locations",
        "re-identify the target after camera viewpoint changes",
    ],
    "wrong_receptacle": [
        "place into visually similar receptacles",
        "approach from different directions",
        "verify the receptacle before release",
    ],
    "sequence_error": [
        "execute subtasks in the required order",
        "verify state after each subtask",
        "resume correctly after an interrupted step",
    ],
    "timeout": [
        "complete the task from varied initial states",
        "use efficient motion near the time limit",
        "avoid indecisive loops and redundant actions",
    ],
    "partial_completion": [
        "complete all subtasks rather than stopping after the first",
        "transition between manipulation stages",
        "recover after an intermediate success",
    ],
    "unknown": [
        "include diverse object layouts",
        "annotate collisions, contact, and termination causes",
        "pair failures with matched successful demonstrations",
    ],
}


@dataclass
class FailureDiagnosis:
    """Structured diagnosis of a single failed episode."""

    episode_idx: int
    task_name: str
    failure_mode: str
    confidence: float
    failure_step: int
    description: str
    root_cause: str
    suggested_data: str
    raw_analysis: str = ""


@dataclass
class TaskDiagnosisReport:
    """Aggregated diagnosis for all failures in a task."""

    task_name: str
    total_episodes: int
    failed_episodes: int
    success_rate: float
    failure_mode_distribution: Dict[str, int] = field(default_factory=dict)
    diagnoses: List[FailureDiagnosis] = field(default_factory=list)
    data_generation_priorities: List[Dict[str, Any]] = field(default_factory=list)


class FailureAnalyzer:
    """Analyze failed trajectories using a frontier AI model."""

    def __init__(
        self,
        provider_model: str = "",
        project: str = "zhouxueyangapexinai",
        provider_key_id: str = "",
        *,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            provider_model: Exact model name passed to the Agent-level Provider
                API relay. This is metadata, not a credential.
            project: Project identifier passed to the managed provider relay.
            provider_key_id: Metadata-only provider key identifier. This is not
                an API key and is never read from the environment.
            model_name: Backward-compatible alias for ``provider_model``.
        """
        if model_name is not None:
            if provider_model and provider_model != model_name:
                raise ValueError(
                    "provider_model and model_name must match when both are set"
                )
            provider_model = model_name
        self.provider_model = provider_model
        self.project = project
        self.provider_key_id = provider_key_id

    def analyze_episode(self, trajectory_path: str) -> FailureDiagnosis:
        """Analyze one failed episode JSON file.

        Args:
            trajectory_path: Path to an episode JSON file from
                ``TrajectoryLogger`` or a compatible exporter.

        Returns:
            A validated ``FailureDiagnosis``.

        Raises:
            ValueError: If the model response is not valid diagnosis JSON.
            OSError: If the trajectory file cannot be read.
        """
        path = Path(trajectory_path)
        with path.open(encoding="utf-8") as handle:
            episode_data = json.load(handle)

        features = self._extract_trajectory_features(episode_data)
        task_name = str(
            self._first_value(
                episode_data if isinstance(episode_data, Mapping) else {},
                ("task_description", "task_name", "task"),
            )
            or path.parent.name
        )
        prompt = self._format_analysis_prompt(features, fallback_task=task_name)

        raw_response = self._call_ai_model(prompt)
        analysis = self._parse_and_validate_response(raw_response)
        episode_idx = self._coerce_int(
            self._first_value(
                episode_data if isinstance(episode_data, Mapping) else {},
                ("episode_idx", "episode", "index"),
            ),
            default=-1,
        )
        return FailureDiagnosis(
            episode_idx=episode_idx,
            task_name=task_name,
            failure_mode=analysis["failure_mode"],
            confidence=analysis["confidence"],
            failure_step=analysis["failure_step"],
            description=analysis["description"],
            root_cause=analysis["root_cause"],
            suggested_data=analysis["suggested_data"],
            raw_analysis=raw_response,
        )

    def _extract_trajectory_features(self, trajectory: Any) -> Dict[str, Any]:
        """Extract prompt-ready events, rewards, and terminal state features.

        The primary supported schema is ``EpisodeRecord`` serialized by
        ``TrajectoryLogger``. For compatibility, ``trajectory`` may also be a
        raw list of step dictionaries or contain a ``trajectory``/``records``
        list.
        """
        root: Mapping[str, Any] = (
            trajectory if isinstance(trajectory, Mapping) else {}
        )
        steps = self._get_steps(trajectory)
        rewards = [
            self._coerce_float(step.get("reward"), default=0.0)
            for step in steps
        ]
        num_steps = self._coerce_int(
            root.get("num_steps"),
            default=len(steps),
        )
        if num_steps <= 0:
            num_steps = len(steps)

        total_reward_value = root.get("total_reward")
        total_reward = (
            self._coerce_float(total_reward_value, default=sum(rewards))
            if total_reward_value is not None
            else sum(rewards)
        )
        events = self._extract_events(steps)
        terminal = self._extract_terminal_state(root, steps)
        termination_reason = self._classify_termination(root, steps)
        task_description = str(
            self._first_value(
                root,
                ("task_description", "task_name", "task"),
            )
            or ""
        )
        return {
            "task_description": task_description,
            "num_steps": num_steps,
            "total_reward": round(total_reward, 6),
            "events": events,
            "events_timeline": self._format_events(events),
            "reward_curve": rewards,
            "reward_curve_summary": self._summarize_rewards(rewards),
            "terminal_state": terminal,
            "termination_reason": termination_reason,
            "ee_position": terminal.get("ee_position"),
            "gripper_state": terminal.get("gripper_state"),
            "base_position": terminal.get("base_position"),
            "object_positions": terminal.get("object_positions", {}),
        }

    def _format_analysis_prompt(
        self,
        features: Mapping[str, Any],
        fallback_task: str = "",
    ) -> str:
        """Format the canonical prompt from extracted trajectory features."""
        return FAILURE_ANALYSIS_PROMPT.format(
            task_description=features["task_description"] or fallback_task,
            num_steps=features["num_steps"],
            total_reward=features["total_reward"],
            termination_reason=features["termination_reason"],
            events_timeline=features["events_timeline"],
            ee_position=self._format_value(features["ee_position"]),
            gripper_state=self._format_value(features["gripper_state"]),
            base_position=self._format_value(features["base_position"]),
            reward_curve_summary=features["reward_curve_summary"],
            failure_categories=self._format_failure_categories(),
        )

    def _call_ai_model(self, prompt: str) -> str:
        """Call the frontier AI model for analysis.

        In production, the Agent calls the Provider API relay with
        ``provider_api_request(project=..., key_id=..., method="POST",
        path="chat/completions", json_body={...})``. The relay receives the
        metadata-only ``provider_key_id`` and ``provider_model``; no raw
        provider key is read or passed by this module.

        The current implementation intentionally returns a valid placeholder
        response so local parsing, validation, and batch workflows can be
        tested without provider access.
        """
        del prompt

        # Production integration pattern:
        #
        # from agent_ml_research import provider_api_request
        #
        # response = provider_api_request(
        #     project=self.project,
        #     key_id=self.provider_key_id,
        #     method="POST",
        #     path="chat/completions",
        #     json_body={
        #         "model": self.provider_model,
        #         "messages": [
        #             {
        #                 "role": "system",
        #                 "content": "Return only the requested JSON diagnosis.",
        #             },
        #             {"role": "user", "content": prompt},
        #         ],
        #         "temperature": 0.0,
        #         "response_format": {"type": "json_object"},
        #     },
        # )
        # return response["choices"][0]["message"]["content"]

        return json.dumps(
            {
                "failure_mode": "unknown",
                "confidence": 0.2,
                "failure_step": -1,
                "description": "Placeholder analysis; no frontier model was called.",
                "root_cause": "Provider API integration is not enabled.",
                "suggested_data": (
                    "Enable the provider relay and provide labeled successful "
                    "and failed trajectories."
                ),
            }
        )

    def analyze_task_batch(
        self,
        trajectory_dir: str,
        task_name: str = "",
        max_workers: int = 4,
    ) -> TaskDiagnosisReport:
        """Analyze failed episode JSON files concurrently.

        Args:
            trajectory_dir: Directory containing failed episode JSON files.
            task_name: Optional task name. If omitted, infer it from episode
                metadata or the directory name.
            max_workers: Maximum number of concurrent model calls.

        Returns:
            A report with a failure distribution sorted by descending count.
            Individual unreadable or invalid episodes are logged and skipped.
        """
        directory = Path(trajectory_dir)
        episode_files = sorted(directory.glob("episode_*.json"))
        if not episode_files:
            episode_files = sorted(
                path
                for path in directory.glob("*.json")
                if path.name != "eval_results.json"
            )
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        diagnoses: List[FailureDiagnosis] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze_episode, str(path)): path
                for path in episode_files
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    diagnoses.append(future.result())
                except Exception:
                    logger.exception("Skipping failed episode analysis: %s", path)

        diagnoses.sort(key=lambda item: (item.episode_idx, item.task_name))
        counts: Dict[str, int] = {}
        for diagnosis in diagnoses:
            counts[diagnosis.failure_mode] = (
                counts.get(diagnosis.failure_mode, 0) + 1
            )
        sorted_counts = dict(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )

        summary_total, summary_failed = self._read_summary_counts(directory)
        total_episodes = summary_total or len(episode_files)
        failed_episodes = summary_failed or len(episode_files)
        inferred_task = (
            task_name
            or (diagnoses[0].task_name if diagnoses else "")
            or directory.name
        )
        priorities = self._build_priorities(sorted_counts, failed_episodes)
        return TaskDiagnosisReport(
            task_name=inferred_task,
            total_episodes=total_episodes,
            failed_episodes=failed_episodes,
            success_rate=(
                max(total_episodes - failed_episodes, 0)
                / max(total_episodes, 1)
            ),
            failure_mode_distribution=sorted_counts,
            diagnoses=diagnoses,
            data_generation_priorities=priorities,
        )

    def analyze_task(
        self,
        trajectory_dir: str,
        task_name: str,
        max_workers: int = 4,
    ) -> TaskDiagnosisReport:
        """Analyze the legacy ``trajectory_dir/task_name`` layout."""
        task_dir = Path(trajectory_dir) / task_name
        if not task_dir.exists():
            task_dir = Path(trajectory_dir)
        return self.analyze_task_batch(
            str(task_dir),
            task_name=task_name,
            max_workers=max_workers,
        )

    def generate_data_spec(self, report: TaskDiagnosisReport) -> Dict[str, Any]:
        """Generate targeted demonstration requirements from a report.

        Only failure modes accounting for at least ten percent of diagnosed
        failures are included. Demonstration counts use a 100-example planning
        budget, so the suggested count is proportional to observed frequency
        with a minimum of ten examples for any selected mode.
        """
        denominator = max(report.failed_episodes, 1)
        target_modes: List[Dict[str, Any]] = []
        for mode, count in report.failure_mode_distribution.items():
            proportion = count / denominator
            if proportion < 0.10:
                continue
            priority_score = round(proportion * 100.0, 2)
            target_modes.append(
                {
                    "mode": mode,
                    "frequency": count,
                    "proportion": round(proportion, 4),
                    "priority_score": priority_score,
                    "num_demonstrations_needed": max(
                        10,
                        int(math.ceil(proportion * 100.0)),
                    ),
                    "demonstration_description": _DEMONSTRATION_DESCRIPTIONS.get(
                        mode,
                        _DEMONSTRATION_DESCRIPTIONS["unknown"],
                    ),
                    "key_scenarios": _SCENARIO_HINTS.get(
                        mode,
                        _SCENARIO_HINTS["unknown"],
                    ),
                    "failure_mode_description": FAILURE_MODES.get(
                        mode,
                        FAILURE_MODES["unknown"],
                    ),
                }
            )
        target_modes.sort(
            key=lambda item: (-item["priority_score"], item["mode"])
        )
        return {
            "task_name": report.task_name,
            "current_success_rate": report.success_rate,
            "diagnosed_failures": report.failed_episodes,
            "target_failure_modes": target_modes,
        }

    def _parse_and_validate_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse and validate the model's exact diagnosis JSON contract."""
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError("AI response is empty")
        candidate = raw_response.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("AI response is not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("AI response must be a JSON object")

        required = (
            "failure_mode",
            "confidence",
            "failure_step",
            "description",
            "root_cause",
            "suggested_data",
        )
        missing = [key for key in required if key not in parsed]
        if missing:
            raise ValueError(f"AI response missing fields: {', '.join(missing)}")
        failure_mode = str(parsed["failure_mode"])
        if failure_mode not in FAILURE_MODES:
            raise ValueError(f"Unknown failure mode: {failure_mode}")
        confidence = self._coerce_float(parsed["confidence"], default=-1.0)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        failure_step = self._coerce_int(parsed["failure_step"], default=-1)
        return {
            "failure_mode": failure_mode,
            "confidence": confidence,
            "failure_step": failure_step,
            "description": str(parsed["description"]),
            "root_cause": str(parsed["root_cause"]),
            "suggested_data": str(parsed["suggested_data"]),
        }

    def _extract_events(self, steps: Sequence[Mapping[str, Any]]) -> List[str]:
        """Identify reward, gripper, collision, and direction-change events."""
        events: List[Tuple[int, str]] = []
        rewards = [
            self._coerce_float(step.get("reward"), default=0.0)
            for step in steps
        ]
        deltas = [
            rewards[index] - rewards[index - 1]
            for index in range(1, len(rewards))
        ]
        if deltas:
            mean_abs = statistics.mean(abs(delta) for delta in deltas)
            spread = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            change_threshold = max(0.5, mean_abs + 2.0 * spread)
            for index, delta in enumerate(deltas, start=1):
                if abs(delta) >= change_threshold:
                    direction = "increase" if delta > 0 else "drop"
                    events.append(
                        (
                            self._step_number(steps[index], index),
                            f"large reward {direction} ({delta:+.3f})",
                        )
                    )

        previous_gripper: Optional[str] = None
        previous_position: Optional[List[float]] = None
        previous_direction: Optional[List[float]] = None
        for index, step in enumerate(steps):
            step_number = self._step_number(step, index)
            gripper = self._gripper_label(step)
            if gripper is not None and previous_gripper is not None:
                if gripper != previous_gripper:
                    events.append(
                        (
                            step_number,
                            f"gripper changed {previous_gripper} -> {gripper}",
                        )
                    )
            if gripper is not None:
                previous_gripper = gripper

            position = self._position_from_step(step)
            if position is not None:
                if previous_position is not None:
                    direction = [
                        position[axis] - previous_position[axis]
                        for axis in range(min(3, len(position)))
                    ]
                    if previous_direction is not None and self._is_direction_change(
                        previous_direction,
                        direction,
                    ):
                        events.append((step_number, "end-effector direction change"))
                    if any(abs(value) > 1e-9 for value in direction):
                        previous_direction = direction
                previous_position = position

            if self._step_has_collision(step):
                events.append((step_number, "collision or contact detected"))

        return [
            f"Step {step_number}: {description}"
            for step_number, description in sorted(events, key=lambda item: item[0])
        ]

    def _extract_terminal_state(
        self,
        root: Mapping[str, Any],
        steps: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Collect final robot, gripper, base, and object state."""
        last = steps[-1] if steps else {}
        last_info = last.get("info", {})
        if not isinstance(last_info, Mapping):
            last_info = {}
        terminal = root.get("terminal_state", {})
        if not isinstance(terminal, Mapping):
            terminal = {}

        def value(keys: Iterable[str]) -> Any:
            for source in (terminal, last, last_info, root):
                found = self._first_value(source, keys)
                if found is not None:
                    return found
            return None

        object_positions = value(
            ("object_positions", "objects", "object_states")
        )
        if object_positions is None:
            object_positions = {}
        return {
            "ee_position": value(
                ("ee_position", "end_effector_position", "eef_position")
            )
            or self._position_from_step(last),
            "gripper_state": value(
                ("gripper_state", "gripper", "gripper_position")
            ),
            "base_position": value(
                ("base_position", "robot_base_position", "base_pose")
            ),
            "object_positions": object_positions,
        }

    def _classify_termination(
        self,
        root: Mapping[str, Any],
        steps: Sequence[Mapping[str, Any]],
    ) -> str:
        """Classify termination as timeout, collision, done, success, or unknown."""
        explicit = self._first_value(
            root,
            ("termination_reason", "terminated_by", "termination"),
        )
        if explicit is not None:
            return str(explicit)
        if any(self._step_has_collision(step) for step in steps):
            return "collision"
        if any(self._step_has_timeout(step) for step in steps):
            return "timeout"
        if steps and bool(steps[-1].get("done", False)):
            if bool(steps[-1].get("success", False)) or bool(
                root.get("success", False)
            ):
                return "success_done_signal"
            return "done_signal"
        return "unknown"

    def _summarize_rewards(self, rewards: Sequence[float]) -> str:
        """Create a compact, model-readable reward curve summary."""
        if not rewards:
            return "No per-step rewards were recorded."
        deltas = [
            rewards[index] - rewards[index - 1]
            for index in range(1, len(rewards))
        ]
        drops = [
            f"step {index}: {delta:+.3f}"
            for index, delta in enumerate(deltas, start=1)
            if delta < 0
        ]
        non_decreasing = all(delta >= -1e-9 for delta in deltas)
        non_increasing = all(delta <= 1e-9 for delta in deltas)
        if non_decreasing:
            trend = "monotonic_increasing"
        elif non_increasing:
            trend = "monotonic_decreasing"
        else:
            trend = "non_monotonic"
        drop_text = ", ".join(drops[:8]) if drops else "none"
        return (
            f"initial={rewards[0]:.3f}; final={rewards[-1]:.3f}; "
            f"minimum={min(rewards):.3f}; maximum={max(rewards):.3f}; "
            f"monotonic={trend}; drops={drop_text}"
        )

    def _read_summary_counts(self, directory: Path) -> Tuple[int, int]:
        """Read total and failed counts from an optional summary CSV."""
        summary_path = directory / "summary.csv"
        if not summary_path.exists():
            return 0, 0
        total = 0
        failed = 0
        try:
            with summary_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    total += 1
                    success = str(row.get("success", "")).strip().lower()
                    if success in {"false", "0", "no"}:
                        failed += 1
        except (OSError, csv.Error):
            logger.exception("Unable to read summary file: %s", summary_path)
            return 0, 0
        return total, failed

    def _build_priorities(
        self,
        distribution: Mapping[str, int],
        failed_episodes: int,
    ) -> List[Dict[str, Any]]:
        """Build sorted report priorities from a failure distribution."""
        denominator = max(failed_episodes, 1)
        return [
            {
                "failure_mode": mode,
                "count": count,
                "proportion": count / denominator,
                "description": FAILURE_MODES.get(
                    mode,
                    FAILURE_MODES["unknown"],
                ),
            }
            for mode, count in distribution.items()
        ]

    @staticmethod
    def _get_steps(trajectory: Any) -> List[Mapping[str, Any]]:
        if isinstance(trajectory, list):
            candidate = trajectory
        elif isinstance(trajectory, Mapping):
            candidate = trajectory.get("steps")
            if candidate is None:
                candidate = trajectory.get("trajectory")
            if candidate is None:
                candidate = trajectory.get("records")
            if isinstance(candidate, Mapping):
                candidate = candidate.get("steps", [])
        else:
            candidate = []
        if not isinstance(candidate, list):
            return []
        return [step for step in candidate if isinstance(step, Mapping)]

    @staticmethod
    def _first_value(
        source: Mapping[str, Any],
        keys: Iterable[str],
    ) -> Any:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
        return None

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _step_number(step: Mapping[str, Any], fallback: int) -> int:
        return FailureAnalyzer._coerce_int(step.get("step"), default=fallback)

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "unavailable"
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)

    @staticmethod
    def _format_failure_categories() -> str:
        return "\n".join(
            f"- {name}: {description}"
            for name, description in FAILURE_MODES.items()
        )

    @staticmethod
    def _format_events(events: Sequence[str]) -> str:
        if not events:
            return "No key events detected."
        return "\n".join(f"- {event}" for event in events)

    @staticmethod
    def _gripper_label(step: Mapping[str, Any]) -> Optional[str]:
        value = step.get("gripper_state")
        if value is None:
            info = step.get("info")
            if isinstance(info, Mapping):
                value = info.get("gripper_state", info.get("gripper"))
        if value is None:
            return None
        if isinstance(value, str):
            return value.lower()
        numeric = FailureAnalyzer._coerce_float(value, default=float("nan"))
        if math.isnan(numeric):
            return str(value)
        return "open" if numeric > 0.5 else "closed"

    @staticmethod
    def _position_from_step(step: Mapping[str, Any]) -> Optional[List[float]]:
        candidates: List[Any] = [
            step.get("ee_position"),
            step.get("end_effector_position"),
            step.get("eef_position"),
        ]
        robot_state = step.get("robot_state")
        if isinstance(robot_state, Mapping):
            candidates.extend(
                [
                    robot_state.get("ee_position"),
                    robot_state.get("end_effector_position"),
                    robot_state.get("position"),
                ]
            )
        elif isinstance(robot_state, Sequence) and not isinstance(
            robot_state, (str, bytes)
        ):
            candidates.append(robot_state[:3])
        for candidate in candidates:
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes)
            ):
                values = [
                    FailureAnalyzer._coerce_float(value, default=float("nan"))
                    for value in candidate[:3]
                ]
                if len(values) == 3 and all(not math.isnan(value) for value in values):
                    return values
        return None

    @staticmethod
    def _step_has_collision(step: Mapping[str, Any]) -> bool:
        if bool(step.get("collision", False)):
            return True
        info = step.get("info")
        if isinstance(info, Mapping):
            return bool(
                info.get("collision", False)
                or info.get("collided", False)
                or info.get("contact", False)
            )
        return False

    @staticmethod
    def _step_has_timeout(step: Mapping[str, Any]) -> bool:
        if bool(step.get("timeout", False)):
            return True
        info = step.get("info")
        if isinstance(info, Mapping):
            return bool(info.get("timeout", False))
        return False

    @staticmethod
    def _is_direction_change(
        previous: Sequence[float],
        current: Sequence[float],
    ) -> bool:
        if not previous or not current:
            return False
        length = min(len(previous), len(current), 3)
        dot = sum(previous[index] * current[index] for index in range(length))
        previous_norm = math.sqrt(
            sum(previous[index] ** 2 for index in range(length))
        )
        current_norm = math.sqrt(
            sum(current[index] ** 2 for index in range(length))
        )
        return previous_norm > 1e-9 and current_norm > 1e-9 and dot < 0.0


__all__ = [
    "FAILURE_ANALYSIS_PROMPT",
    "FAILURE_MODES",
    "FailureAnalyzer",
    "FailureDiagnosis",
    "TaskDiagnosisReport",
]
