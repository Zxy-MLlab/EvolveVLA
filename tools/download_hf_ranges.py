#!/usr/bin/env python3
"""Resume a large Hugging Face blob with independent HTTP range requests."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("output", type=Path)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    part_dir = args.output.with_name(args.output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    width = (args.size + args.chunks - 1) // args.chunks

    def fetch(i: int) -> Path:
        start = i * width
        end = min(args.size, (i + 1) * width) - 1
        part = part_dir / f"{i:04d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        tmp = part.with_suffix(".part.incomplete")
        cmd = [
            "curl", "-sS", "-L", "--fail", "--retry", "12", "--retry-all-errors",
            "--connect-timeout", "30", "--max-time", "300",
            "--speed-time", "60", "--speed-limit", "1000",
        ]
        if tmp.exists() and tmp.stat().st_size > 0:
            cmd.extend(["-C", "-"])
        else:
            cmd.extend(["-r", f"{start}-{end}"])
        cmd.extend(["-o", str(tmp), args.url])
        subprocess.run(cmd, check=True)
        got = tmp.stat().st_size
        expected = end - start + 1
        if got != expected:
            raise RuntimeError(f"chunk {i} size {got}, expected {expected}")
        os.replace(tmp, part)
        return part

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(fetch, range(args.chunks)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        for i in range(args.chunks):
            out.write((part_dir / f"{i:04d}.part").read_bytes())
    if args.output.stat().st_size != args.size:
        raise RuntimeError("assembled output has unexpected size")
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
