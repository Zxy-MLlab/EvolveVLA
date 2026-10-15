"""Opt-in closed-loop trace wrappers for the RoboTwin policy boundary.

This module is intentionally independent of the pinned evaluation harness.
Nothing is wrapped, opened, or written unless ROBOTWIN_TRACE_JSONL is set.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable


ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
_SESSION_ATTR = "_robotwin_loop_trace_session"


def trace_enabled() -> bool:
    value = os.environ.get("ROBOTWIN_TRACE_JSONL")
    return bool(value and os.path.isabs(value))


def _as_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        value = value.reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = list(value)
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("trace action contains a non-finite value")
    return result


def _arm_norm(action: list[float]) -> float:
    if len(action) <= max(ARM_INDICES):
        raise ValueError(f"trace action must have at least 13 values, got {len(action)}")
    return math.sqrt(sum(action[index] ** 2 for index in ARM_INDICES))


class _TraceSession:
    def __init__(self, model_path: str) -> None:
        trace_path = os.environ["ROBOTWIN_TRACE_JSONL"]
        if not os.path.isabs(trace_path):
            raise RuntimeError("ROBOTWIN_TRACE_JSONL must be an absolute path")
        self.model_path = str(model_path)
        self.path = Path(trace_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")
        self.episode = -1
        self.step = 0
        self.policy_call_index = 0
        self.chunk_index = -1
        self.current: dict[str, Any] | None = None
        self.raw_action: list[float] | None = None
        self.postprocessed_action: list[float] | None = None
        self.queue_chunk_position = 0
        self.queue_length_before = 0
        self.queue_length_after = 0
        self.n_action_steps = 1

    def close(self) -> None:
        self.stream.close()

    def wrap_policy(self, policy: Any) -> None:
        original_reset = policy.reset
        original_select_action = policy.select_action
        session = self

        def reset(*args: Any, **kwargs: Any) -> Any:
            result = original_reset(*args, **kwargs)
            session.episode += 1
            session.step = 0
            session.current = None
            session.raw_action = None
            session.postprocessed_action = None
            return result

        def select_action(batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            queue = getattr(policy, "_action_queue", ())
            before = len(queue)
            session.n_action_steps = int(
                getattr(getattr(policy, "config", None), "n_action_steps", 1)
            )
            result = original_select_action(batch, *args, **kwargs)
            after = len(getattr(policy, "_action_queue", ()))
            session.queue_length_before = before
            session.queue_length_after = after
            if before == 0:
                session.chunk_index += 1
                session.queue_chunk_position = 0
            else:
                session.queue_chunk_position = max(
                    0, session.n_action_steps - 1 - after
                )
            session.raw_action = _as_list(result)
            return result

        policy.reset = reset
        policy.select_action = select_action

    def begin_policy_call(self, observation: dict[str, Any]) -> None:
        if self.episode < 0:
            self.episode = 0
        self.current = {
            "policy_call_index": self.policy_call_index,
            "episode": self.episode,
            "step": self.step,
            "frame_source": "sim",
            "frame_index": self.step,
            "frame_index_scope": "episode_sim_get_obs_ordinal",
            "model_path": self.model_path,
        }
        self.policy_call_index += 1
        self.raw_action = None
        self.postprocessed_action = None

    def finish_action(self, executed: Any) -> None:
        if (
            self.current is None
            or self.raw_action is None
            or self.postprocessed_action is None
        ):
            raise RuntimeError("trace action was finalized without a policy call")
        raw = self.raw_action
        post = self.postprocessed_action
        done = _as_list(executed)
        record = dict(self.current)
        record.update(
            {
                "queue_chunk_position": self.queue_chunk_position,
                "chunk_index": self.chunk_index,
                "queue_length_before": self.queue_length_before,
                "queue_length_after": self.queue_length_after,
                "n_action_steps": self.n_action_steps,
                "raw_action": raw,
                "postprocessed_action": post,
                "executed_action": done,
                "raw_action_arm_norm": _arm_norm(raw),
                "postprocessed_action_arm_norm": _arm_norm(post),
                "executed_action_arm_norm": _arm_norm(done),
                "arm_indices": list(ARM_INDICES),
            }
        )
        self.stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.stream.flush()
        self.step += 1
        self.current = None


def install(harness_module: Any) -> bool:
    """Install opt-in wrappers around the existing harness functions."""

    if not trace_enabled():
        return False
    if getattr(harness_module, _SESSION_ATTR, None) is not None:
        return True

    original_load_policy = harness_module.load_policy
    original_make_policy_batch = harness_module.make_policy_batch
    session_holder: dict[str, _TraceSession | None] = {"session": None}

    def load_policy(args: Any) -> Any:
        result = original_load_policy(args)
        policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = result
        session = _TraceSession(str(args.model_path))
        session_holder["session"] = session
        session.wrap_policy(policy)

        def traced_make_policy_batch(*batch_args: Any, **batch_kwargs: Any) -> Any:
            batch = original_make_policy_batch(*batch_args, **batch_kwargs)
            session.begin_policy_call(batch_args[0])
            return batch

        def traced_postprocessor(action: Any) -> Any:
            processed = postprocessor(action)
            session.postprocessed_action = _as_list(processed)
            return processed

        def traced_env_postprocessor(batch: dict[str, Any]) -> dict[str, Any]:
            processed = env_postprocessor(batch)
            action_key = getattr(harness_module, "ACTION", "action")
            if action_key not in processed:
                action_key = "action"
            session.finish_action(processed[action_key])
            return processed

        harness_module.make_policy_batch = traced_make_policy_batch
        harness_module.postprocessor = traced_postprocessor
        harness_module.env_postprocessor = traced_env_postprocessor
        return (
            policy,
            preprocessor,
            traced_postprocessor,
            env_preprocessor,
            traced_env_postprocessor,
        )

    # run_parent uses the postprocessors returned by load_policy, so the
    # wrappers above are sufficient; make_policy_batch is installed globally
    # after load_policy returns and before the first episode starts.
    harness_module.load_policy = load_policy
    setattr(harness_module, _SESSION_ATTR, session_holder)
    return True
