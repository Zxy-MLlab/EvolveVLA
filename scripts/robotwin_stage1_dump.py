#!/usr/bin/env python
"""Restore the RobotWin Orbax params tree on CPU and dump a neutral snapshot."""

import hashlib
import os
import pickle
import sys
import traceback
import zipfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import numpy as np
import orbax.checkpoint as ocp
from etils import epath


ROOT = Path(
    "/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin/"
    "RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999"
)
PARAMS = ROOT / "params"
OUTPUT_DIR = Path("/root/autodl-tmp/EvolveVLA/checkpoints/convert_stageA")
NPZ_OUTPUT = OUTPUT_DIR / "orbax_params.npz"
PICKLE_OUTPUT = OUTPUT_DIR / "orbax_params.pkl"
MANIFEST_PATH = Path(
    "/root/autodl-tmp/EvolveVLA/results/stage1_leaf_manifest.txt"
)


class KeyPathNamingError(ValueError):
    """Raised when a pytree key path cannot be represented as an npz name."""


def restore_verified_tree():
    handler = ocp.PyTreeCheckpointHandler()
    metadata = handler.metadata(epath.Path(str(PARAMS)))
    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def make_restore_args(node):
        if isinstance(node, dict):
            return {key: make_restore_args(value) for key, value in node.items()}
        return ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=cpu_sharding,
            global_shape=tuple(node.shape),
        )

    restore_args = make_restore_args(metadata.tree)
    return handler.restore(
        epath.Path(str(PARAMS)),
        args=ocp.args.PyTreeRestore(restore_args=restore_args),
    )


def key_path_to_name(key_path):
    parts = []
    for key in key_path:
        if isinstance(key, jax.tree_util.DictKey):
            parts.append(f"dict[{key.key!r}]")
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(f"seq[{key.idx}]")
        elif isinstance(key, jax.tree_util.GetAttrKey):
            parts.append(f"attr[{key.name}]")
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            parts.append(f"flat[{key.key}]")
        else:
            raise KeyPathNamingError(
                f"unsupported pytree key type: {type(key).__name__}"
            )

    name = "/".join(parts) or "root"
    if not name or "\x00" in name:
        raise KeyPathNamingError(f"invalid pytree key path name: {name!r}")
    return name


def flatten_named_leaves(tree):
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    named = []
    seen = set()
    for key_path, leaf in path_leaves:
        name = key_path_to_name(key_path)
        if name in seen:
            raise KeyPathNamingError(f"duplicate pytree key path name: {name}")
        seen.add(name)
        named.append((name, leaf))
    return sorted(named, key=lambda item: item[0])


def fallback_named_leaves(tree):
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    named = []
    for index, (_key_path, leaf) in enumerate(path_leaves):
        named.append((f"leaf_{index:04d}", leaf))
    return sorted(named, key=lambda item: item[0])


def leaf_metadata(named_leaves):
    rows = []
    total_elements = 0
    for name, leaf in named_leaves:
        shape = tuple(int(dim) for dim in leaf.shape)
        dtype = str(np.dtype(leaf.dtype))
        total_elements += int(leaf.size)
        rows.append((name, shape, dtype))
    return rows, total_elements


def manifest_text(rows):
    return "\n".join(
        f"{name} {shape} {dtype}" for name, shape, dtype in rows
    ) + "\n"


def dump_npz(named_leaves, output_path):
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for index, (name, leaf) in enumerate(named_leaves, start=1):
                print(
                    f"STAGE1_PROGRESS={index}/{len(named_leaves)} "
                    f"name={name}",
                    flush=True,
                )
                with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(
                        member,
                        np.asarray(leaf),
                        allow_pickle=False,
                    )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def dump_pickle(named_leaves, output_path):
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        arrays = {}
        for index, (name, leaf) in enumerate(named_leaves, start=1):
            print(
                f"STAGE1_PROGRESS={index}/{len(named_leaves)} name={name}",
                flush=True,
            )
            arrays[name] = np.asarray(leaf)
        with temporary_path.open("wb") as handle:
            pickle.dump(arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("STAGE1_RESTORE_START", flush=True)
    restored = restore_verified_tree()
    print("STAGE1_RESTORE_DONE", flush=True)

    try:
        named_leaves = flatten_named_leaves(restored)
        output_path = NPZ_OUTPUT
        output_format = "npz"
    except KeyPathNamingError as exc:
        print(f"STAGE1_FORMAT_FALLBACK={exc}", flush=True)
        named_leaves = fallback_named_leaves(restored)
        output_path = PICKLE_OUTPUT
        output_format = "pickle"

    rows, total_elements = leaf_metadata(named_leaves)
    print(f"STAGE1_FORMAT={output_format}", flush=True)
    print(
        f"STAGE1_EXPECTED_LEAVES={len(named_leaves)} "
        f"STAGE1_EXPECTED_TOTAL_ELEMENTS={total_elements}",
        flush=True,
    )

    if output_format == "npz":
        dump_npz(named_leaves, output_path)
    else:
        dump_pickle(named_leaves, output_path)

    manifest = manifest_text(rows)
    MANIFEST_PATH.write_text(manifest, encoding="utf-8")
    output_bytes, output_sha256 = sha256_and_size(output_path)

    print(f"STAGE1_OUTPUT_PATH={output_path}", flush=True)
    print(f"STAGE1_OUTPUT_BYTES={output_bytes}", flush=True)
    print(f"STAGE1_OUTPUT_SHA256={output_sha256}", flush=True)
    print(f"STAGE1_LEAF_COUNT={len(named_leaves)}", flush=True)
    print(f"STAGE1_TOTAL_ELEMENTS={total_elements}", flush=True)
    print("STAGE1_LEAF_MANIFEST_BEGIN", flush=True)
    print(manifest, end="", flush=True)
    print("STAGE1_LEAF_MANIFEST_END", flush=True)
    print("STAGE1_EXIT=0", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("STAGE1_EXIT=1", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
