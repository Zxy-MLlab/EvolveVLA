#!/usr/bin/env python3
"""Select the D127 instruction source, then run the unchanged harness.

The wrapper is used for both harness roles. It replaces the imported selector
before dispatching to ``robotwin_eval_harness.main()``; no simulator or policy
is started by this module itself.
"""

from __future__ import annotations

import os
import sys

import robotwin_eval_harness as harness


def main() -> int:
    source = os.environ.get("ROBOTWIN_INSTRUCTION_SOURCE", "ours")
    if source == "ours":
        from robotwin_instruction_source import seen_instruction
    elif source == "official":
        from robotwin_instruction_source_official import seen_instruction
    else:
        raise SystemExit(
            f"invalid ROBOTWIN_INSTRUCTION_SOURCE={source!r}; "
            "expected ours or official"
        )
    harness.seen_instruction = seen_instruction
    # The unchanged parent computes the simulator script from its own
    # ``__file__``; point that lookup back to this dispatch wrapper so the
    # child receives the same selector choice.
    harness.__file__ = __file__
    return int(harness.main())


if __name__ == "__main__":
    raise SystemExit(main())
