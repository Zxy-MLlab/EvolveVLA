#!/usr/bin/env python3
"""Resilient RoboCasa LeRobot demo fetcher.

Fetch order is intentionally:
1. Hugging Face Hub anonymous resolve URLs
2. hf-mirror.com resolve URLs
3. Official RoboCasa Box links

The script never treats a successful process exit as dataset success. It checks
for downloaded files and LeRobot metadata before printing INTEGRITY_OK.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


DEFAULT_OUTPUT = Path("/root/autodl-tmp/EvolveVLA/data/robocasa_datasets")
DEFAULT_REPOS = (
    "robocasa/robocasa365",
    "robocasa/robocasa365_datasets",
)
BACKOFF_SECONDS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch RoboCasa LeRobot demo archives with bounded retries and integrity checks."
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="RoboCasa task names, e.g. CloseBlenderLid")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Dataset base dir (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--source", choices=("human", "mimicgen"), default="human")
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60, help="Per-attempt network timeout in seconds")
    parser.add_argument("--overall-timeout", type=int, default=900, help="Overall wall-clock bound in seconds")
    parser.add_argument("--no-proxy", action="store_true", help="Do not source /etc/network_turbo")
    parser.add_argument("--dry-run", action="store_true", help="Print planned URLs and destinations without downloading")
    return parser.parse_args()


def apply_network_turbo(no_proxy: bool) -> None:
    if no_proxy:
        return
    turbo = Path("/etc/network_turbo")
    if not turbo.exists():
        print("NETWORK_TURBO_MISSING /etc/network_turbo", file=sys.stderr)
        return
    command = "set -a; source /etc/network_turbo >/dev/null 2>&1; env"
    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print("NETWORK_TURBO_FAIL", result.stderr.strip(), file=sys.stderr)
        return
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.lower() in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        }:
            os.environ[key] = value


def load_robocasa_registry(task: str, source: str, split: str) -> tuple[Path, str]:
    from robocasa.utils.dataset_registry_utils import get_ds_path

    ds_path = Path(get_ds_path(task=task, ds_type=source, split=split))
    parts = ds_path.parts
    try:
        idx = parts.index("v1.0")
        tar_key = "/".join(parts[idx:-1] + (f"{parts[-1]}.tar",))
    except ValueError as exc:
        raise RuntimeError(f"Cannot derive RoboCasa tar key from path: {ds_path}") from exc
    return ds_path, tar_key


def official_box_url(tar_key: str) -> str | None:
    import robocasa

    box_json = Path(robocasa.__path__[0]) / "models/assets/box_links/box_links_ds.json"
    with box_json.open("r", encoding="utf-8") as f:
        links = json.load(f)
    return links.get(tar_key)


def destination_for(output: Path, registered_path: Path) -> Path:
    parts = registered_path.parts
    try:
        idx = parts.index("v1.0")
    except ValueError as exc:
        raise RuntimeError(f"Cannot map RoboCasa dataset path to output root: {registered_path}") from exc
    return output.joinpath(*parts[idx:])


def candidate_urls(tar_key: str, box_url: str | None) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    for repo in DEFAULT_REPOS:
        urls.append(("hf", f"https://huggingface.co/datasets/{repo}/resolve/main/{tar_key}"))
    for repo in DEFAULT_REPOS:
        urls.append(("hf-mirror", f"https://hf-mirror.com/datasets/{repo}/resolve/main/{tar_key}"))
    if box_url:
        urls.append(("box", box_url))
    return urls


def download_url(url: str, archive_path: Path, timeout: int, attempts: int, deadline: float) -> bool:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        if time.monotonic() > deadline:
            print(f"ATTEMPT_SKIP overall_timeout url={url}", file=sys.stderr)
            return False
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EvolveVLA-RoboCasa-fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
            if archive_path.stat().st_size > 0:
                print(f"DOWNLOAD_OK bytes={archive_path.stat().st_size} url={url}")
                return True
            print(f"DOWNLOAD_EMPTY attempt={attempt} url={url}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - log and retry all network failures.
            print(f"DOWNLOAD_FAIL attempt={attempt} url={url} error={type(exc).__name__}: {exc}", file=sys.stderr)
            archive_path.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
    return False


def extract_archive(archive_path: Path, dest: Path) -> None:
    extract_dir = dest.parent
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=extract_dir)


def integrity_ok(destinations: list[Path], output: Path) -> bool:
    file_count = sum(1 for p in output.rglob("*") if p.is_file())
    lerobot_roots = [p for p in destinations if (p / "meta").exists() or (p / "data").exists()]
    meta_ok = all((p / "meta/info.json").is_file() for p in lerobot_roots)
    ok = file_count > 0 and (not lerobot_roots or meta_ok)
    print(f"INTEGRITY_FILE_COUNT {file_count}")
    for dest in destinations:
        print(f"INTEGRITY_DEST {dest} meta_info={(dest / 'meta/info.json').is_file()}")
    print("INTEGRITY_OK" if ok else "INTEGRITY_FAIL")
    return ok


def main() -> int:
    args = parse_args()
    if str(args.output) != str(DEFAULT_OUTPUT) and not str(args.output).startswith("/root/autodl-tmp/"):
        print(f"INTEGRITY_FAIL output must be on data volume: {args.output}", file=sys.stderr)
        return 2

    apply_network_turbo(args.no_proxy)
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.overall_timeout
    destinations: list[Path] = []

    for task in args.tasks:
        registered_path, tar_key = load_robocasa_registry(task, args.source, args.split)
        dest = destination_for(args.output, registered_path)
        destinations.append(dest)
        box_url = official_box_url(tar_key)
        archive_path = dest.with_suffix(".tar")
        urls = candidate_urls(tar_key, box_url)
        print(f"TASK {task}")
        print(f"DEST {dest}")
        for label, url in urls:
            print(f"PLAN_SOURCE {label} {url}")
        if args.dry_run:
            continue
        if (dest / "meta/info.json").is_file():
            print(f"SKIP_EXISTING {dest}")
            continue
        fetched = False
        for label, url in urls:
            print(f"SOURCE_BEGIN {label} {url}")
            if download_url(url, archive_path, args.timeout, args.max_attempts, deadline):
                extract_archive(archive_path, dest)
                archive_path.unlink(missing_ok=True)
                fetched = True
                break
        if not fetched:
            print(f"TASK_FETCH_FAIL {task}", file=sys.stderr)

    if args.dry_run:
        return 0
    return 0 if integrity_ok(destinations, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
