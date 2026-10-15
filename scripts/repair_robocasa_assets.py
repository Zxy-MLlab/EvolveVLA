#!/usr/bin/env python
"""Repair RoboCasa object assets with an idempotent symlink.

The source tree is /root/autodl-tmp/robocasa_assets_package/objects, extracted
from /root/autodl-tmp/EvolveVLA/robocasa_assets/objects_lightwheel.zip.
The official RoboCasa acquisition path is:
  python robocasa/scripts/download_kitchen_assets.py --type objs_lw

This script links the installed robocasa/models/assets/objects path to that
source tree and refuses to replace a non-empty unmanaged directory.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


SOURCE_ROOT = Path("/root/autodl-tmp/robocasa_assets_package/objects")
SOURCE_ARCHIVE = Path(
    "/root/autodl-tmp/EvolveVLA/robocasa_assets/objects_lightwheel.zip"
)


def installed_robocasa_root() -> Path:
    spec = importlib.util.find_spec("robocasa")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("robocasa is not importable in this environment")
    return Path(next(iter(spec.submodule_search_locations)))


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            "missing object source tree: %s (archive: %s)"
            % (SOURCE_ROOT, SOURCE_ARCHIVE)
        )

    target = installed_robocasa_root() / "models" / "assets" / "objects"
    target.parent.mkdir(parents=True, exist_ok=True)
    wanted = SOURCE_ROOT.resolve()

    if target.exists() and target.resolve() == wanted:
        print("OBJECTS LINK OK:", target, "->", wanted)
        return

    if target.is_symlink():
        current = Path(os.path.realpath(target))
        if current == wanted:
            print("OBJECTS LINK OK:", target, "->", wanted)
            return
        target.unlink()
    elif target.exists():
        if target.is_dir() and any(target.iterdir()):
            raise RuntimeError(
                "refusing to replace non-empty directory: %s; "
                "move it aside before rerunning" % target
            )
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()

    target.symlink_to(wanted, target_is_directory=True)
    print("OBJECTS LINK CREATED:", target, "->", wanted)
    print("SOURCE ARCHIVE:", SOURCE_ARCHIVE)
    print("OFFICIAL COMMAND: python robocasa/scripts/download_kitchen_assets.py --type objs_lw")


if __name__ == "__main__":
    main()
