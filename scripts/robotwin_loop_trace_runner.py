#!/usr/bin/env python3
"""Run the unchanged RoboTwin harness with an opt-in closed-loop trace."""

from __future__ import annotations

import os
import sys


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import robotwin_eval_harness as harness  # noqa: E402
from robotwin_loop_trace import install  # noqa: E402


def main() -> int:
    source = os.environ.get("ROBOTWIN_INSTRUCTION_SOURCE", "ours")
    if source == "ours":
        from robotwin_instruction_source import seen_instruction
    elif source == "official":
        from robotwin_instruction_source_official import seen_instruction
    else:
        raise SystemExit(
            f"invalid ROBOTWIN_INSTRUCTION_SOURCE={source!r}; expected ours or official"
        )
    harness.seen_instruction = seen_instruction
    harness.__file__ = __file__
    install(harness)
    return int(harness.main())


if __name__ == "__main__":
    raise SystemExit(main())
