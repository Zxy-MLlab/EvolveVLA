#!/usr/bin/env python3
"""Decode length-prefixed pickle messages captured from the sim child."""

from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path
from typing import Any


HEADER_SIZE = 8


def try_frame(data: bytes, offset: int) -> tuple[int, Any] | None:
    if offset < 0 or offset + HEADER_SIZE > len(data):
        return None
    size = struct.unpack("!Q", data[offset : offset + HEADER_SIZE])[0]
    end = offset + HEADER_SIZE + size
    if size > len(data) - offset - HEADER_SIZE:
        return None
    payload = data[offset + HEADER_SIZE : end]
    try:
        value = pickle.loads(payload)
    except Exception:
        return None
    return size, value


def locate_start(data: bytes) -> tuple[int, int, Any] | None:
    for offset in range(0, max(0, len(data) - HEADER_SIZE + 1)):
        frame = try_frame(data, offset)
        if frame is not None:
            size, value = frame
            return offset, size, value
    return None


def decode_frames(data: bytes, start: int) -> tuple[list[dict[str, Any]], int | None]:
    frames: list[dict[str, Any]] = []
    offset = start
    while offset < len(data):
        if offset + HEADER_SIZE > len(data):
            return frames, offset
        size = struct.unpack("!Q", data[offset : offset + HEADER_SIZE])[0]
        end = offset + HEADER_SIZE + size
        if size > len(data) - offset - HEADER_SIZE:
            return frames, offset
        payload = data[offset + HEADER_SIZE : end]
        try:
            value = pickle.loads(payload)
        except Exception:
            return frames, offset
        frames.append(
            {
                "offset": offset,
                "header_hex": data[offset : offset + HEADER_SIZE].hex(),
                "size": size,
                "payload_len": len(payload),
                "payload_head_hex": payload[:32].hex(),
                "value": value,
            }
        )
        offset = end
    return frames, None


def format_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", "<missing>"))
    return type(value).__name__


def classify(frames: list[dict[str, Any]], trailing_offset: int | None) -> str:
    for frame in frames:
        value = frame["value"]
        if isinstance(value, dict) and value.get("type") == "error":
            message = str(value.get("message", ""))
            if "Traceback (most recent call last):" in message:
                return "VERIFIED: real Python exception traceback is present in the decoded error message."
            return "VERIFIED: decoded error message is present, but it does not contain a Python traceback."
    if trailing_offset is not None:
        return (
            "VERIFIED: no decoded error traceback; the capture ends with an invalid or "
            "truncated frame at the reported offset."
        )
    return "UNVERIFIED: no decoded error message or traceback was found."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/EvolveVLA/results/sim_child_stdout.log",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/EvolveVLA/results/sim_child_error_decoded.md",
    )
    parser.add_argument(
        "--stderr",
        default="/root/autodl-tmp/EvolveVLA/results/sim_child_stderr.log",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = input_path.read_bytes()
    first64 = data[:64].hex()
    located = locate_start(data)

    lines: list[str] = [
        "# Sim-child IPC Decode",
        "",
        f"- **Status:** {'VERIFIED' if located is not None else 'UNVERIFIED'}",
        f"- **Input:** `{input_path}`",
        f"- **Capture size:** `{len(data)}` bytes",
        f"- **Raw first 64 bytes (hex):** `{first64}`",
        "",
        "## Framing",
        "",
        "The harness uses `struct.pack(\"!Q\", len(encoded))` followed by "
        "`pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)`.",
        "",
    ]

    if located is None:
        lines.extend(
            [
                "No valid length-prefixed pickle frame was found at offset 0 or by scanning the file.",
                "",
                "## Classification",
                "",
                "UNVERIFIED: the capture could not be decoded as a length-prefixed pickle stream.",
                "",
            ]
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print("NO_VALID_FRAME")
        print(f"RAW_FIRST64_HEX {first64}")
        return 1

    start, first_size, first_value = located
    frames, trailing_offset = decode_frames(data, start)
    lines.extend(
        [
            f"- **First valid frame offset:** `{start}`",
            f"- **First frame header bytes:** `{data[start:start + HEADER_SIZE].hex()}`",
            f"- **First frame declared payload size:** `{first_size}` bytes",
            f"- **First frame decoded object type:** `{format_value(first_value)}`",
            f"- **Decoded frame count:** `{len(frames)}`",
            f"- **Trailing invalid/truncated offset:** `{trailing_offset}`"
            if trailing_offset is not None
            else "- **Trailing invalid/truncated offset:** none",
            "",
            "## Decoded Messages",
            "",
        ]
    )

    for idx, frame in enumerate(frames):
        value = frame["value"]
        message_type = format_value(value)
        print(f"MSG {idx} type={message_type}")
        lines.extend(
            [
                f"### Message {idx}",
                "",
                f"- **Type:** `{message_type}`",
                f"- **Frame offset:** `{frame['offset']}`",
                f"- **Header bytes:** `{frame['header_hex']}`",
                f"- **Declared size:** `{frame['size']}` bytes",
                f"- **Decoded payload length:** `{frame['payload_len']}` bytes",
                f"- **Payload prefix (hex):** `{frame['payload_head_hex']}`",
                "",
            ]
        )
        if isinstance(value, dict) and value.get("type") == "error":
            message = str(value.get("message", ""))
            print(message, end="" if message.endswith("\n") else "\n")
            lines.extend(["#### Full `message` field", "", "```text", message, "```", ""])
        elif isinstance(value, dict):
            summary = {
                key: value[key]
                for key in ("type", "episode", "seed", "instruction")
                if key in value
            }
            if "observation" in value and isinstance(value["observation"], dict):
                summary["observation_keys"] = sorted(value["observation"].keys())
            lines.extend(["```text", repr(summary), "```", ""])
        else:
            lines.extend(["```text", repr(value), "```", ""])

    lines.extend(["## Classification", "", classify(frames, trailing_offset), ""])
    stderr_path = Path(args.stderr)
    if stderr_path.exists():
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        matching = [
            (idx + 1, line)
            for idx, line in enumerate(stderr.splitlines())
            if any(term in line for term in ("Traceback", "Error", "Exception"))
        ]
        lines.extend(["## Stderr grep", ""])
        lines.append("Command-equivalent match for `Traceback`, `Error`, or `Exception`:")
        lines.append("")
        if matching:
            lines.extend(f"{line_no}:{line}" for line_no, line in matching)
        else:
            lines.append("(no matching lines)")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
