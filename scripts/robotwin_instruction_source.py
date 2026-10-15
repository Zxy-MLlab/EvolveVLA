#!/usr/bin/env python3
"""Instruction source helpers for RoboTwin evaluation."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Dict


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ROBOTWIN_ROOT = _PROJECT_ROOT / "robotwin_repo"
if _ROBOTWIN_ROOT.is_dir():
    sys.path.insert(0, str(_ROBOTWIN_ROOT))

from description.utils.generate_episode_instructions import (  # noqa: E402
    load_task_instructions,
    replace_placeholders,
)


_DEFAULT_ARM_TAG = "left"
_TASK_EPISODE_INFO: Dict[str, Dict[str, str]] = {
    "beat_block_hammer": {
        "{A}": "020_hammer/base0",
    },
}
_UNRESOLVED_PLACEHOLDER = re.compile(r"{[^{}]+}")


def fallback_instruction(task_name: str) -> str:
    """Last-resort back-compat fallback; never return full_description."""
    return task_name


def _episode_params(task_name: str, arm_tag: str | None) -> Dict[str, str]:
    params = dict(_TASK_EPISODE_INFO.get(task_name, {}))
    params["{a}"] = str(arm_tag or _DEFAULT_ARM_TAG)
    return params


def seen_instruction(
    task_name: str,
    arm_tag: str | None = None,
    seed: int = 0,
) -> tuple[str, str]:
    """Return a deterministic instantiated seen instruction for a RoboTwin task."""
    try:
        task_data = load_task_instructions(task_name)
        seen = task_data.get("seen", [])
        if not seen:
            return fallback_instruction(task_name), "task_name_fallback"

        # Deterministic in-distribution template choice: seed modulo seen count.
        template = seen[int(seed) % len(seen)]
        instruction = replace_placeholders(
            template,
            _episode_params(task_name, arm_tag),
            random.Random(seed),
        ).strip()
    except Exception:
        return fallback_instruction(task_name), "task_name_fallback"

    if not instruction or _UNRESOLVED_PLACEHOLDER.search(instruction):
        return fallback_instruction(task_name), "task_name_fallback"
    return instruction, "seen"


if __name__ == "__main__":
    for test_seed in range(5):
        instruction, source = seen_instruction("beat_block_hammer", seed=test_seed)
        print(f"INSTRUCTION={instruction} SOURCE={source}")
