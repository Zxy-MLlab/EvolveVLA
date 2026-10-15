#!/usr/bin/env python3
"""Official RoboTwin instruction-selection adapter.

Purpose: provide the same ``seen_instruction(task_name, arm_tag, seed)``
interface as ``robotwin_instruction_source.py`` while mirroring the official
RoboTwin prompt construction and selection path.

IO: input is a RoboTwin task name, optional arm tag, and integer episode seed;
output is ``(instruction_text, source_name)``.

Exact official algorithm mirrored:
``candidates = generate_episode_descriptions(task_name, [episode_info],
test_num, seed=seed)[0]["seen"]`` followed by
``random.Random(f"{seed}:select").choice(candidates)``.
"""

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
    generate_episode_descriptions,
)


_DEFAULT_ARM_TAG = "left"
_DEFAULT_TEST_NUM = 5
_TASK_EPISODE_INFO: Dict[str, Dict[str, str]] = {
    "beat_block_hammer": {
        "{A}": "020_hammer/base0",
    },
}
_UNRESOLVED_PLACEHOLDER = re.compile(r"{[^{}]+}")


def fallback_instruction(task_name: str) -> str:
    """Return the task name when official prompt construction has no result."""
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
    """Return an instruction using the official RoboTwin selection algorithm."""
    try:
        episode_info = _episode_params(task_name, arm_tag)
        results = generate_episode_descriptions(
            task_name,
            [episode_info],
            _DEFAULT_TEST_NUM,
            seed=seed,
        )
        candidates = results[0].get("seen", []) if results else []
        if candidates:
            instruction = random.Random(f"{seed}:select").choice(candidates).strip()
            if instruction and not _UNRESOLVED_PLACEHOLDER.search(instruction):
                return instruction, "official"
    except Exception:
        pass
    return fallback_instruction(task_name), "task_name_fallback"


if __name__ == "__main__":
    for test_seed in (42, 43, 44):
        instruction, source = seen_instruction(
            "beat_block_hammer",
            seed=test_seed,
        )
        print(f"INSTRUCTION={instruction} SOURCE={source}")
