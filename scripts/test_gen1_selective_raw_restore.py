#!/usr/bin/env python3
"""Tiny synthetic Orbax 0.11.13 selective restore regressions; no real weights.

Run using gen1_train_ready/bin/python -I -B. All generated files are retained in
one newly created workspace tmp directory. No project modules are imported.
Validation here is deliberately independent of Orbax's restore arguments: item
placeholders do not enforce shapes/dtypes, and requested dtype can cast values.
"""
import os

# Set before importing numerical libraries, including when imported by a runner.
os.environ.update(
    CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
    JAX_ENABLE_COMPILATION_CACHE="false", JAX_COMPILATION_CACHE_DIR="",
    HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled",
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
)
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import traceback
from collections import Counter
from contextlib import contextmanager
from unittest import mock

ROOT = Path("/root/autodl-tmp/EvolveVLA")
# These constants are an independent reference, NOT derived from saved metadata.
REFERENCE = {
    ("dense", "kernel"): ((3, 2), "float32"),
    ("dense", "bias"): ((2,), "float32"),
    ("value",): ((1,), "float32"),
    ("value_projection",): ((2, 1), "float32"),
}


def extract_params(flat):
    """Strip exactly the params root and final value component, never substrings."""
    extracted = {}
    for key, value in flat.items():
        if not isinstance(key, tuple) or not all(isinstance(k, str) for k in key):
            raise ValueError("keys must be full string tuples")
        if not key or key[0] != "params":
            continue
        if len(key) < 3 or key[-1] != "value":
            raise ValueError(f"not an NNX final value leaf: {key!r}")
        target = key[1:-1]
        if target in extracted:
            raise ValueError(f"duplicate parameter: {target!r}")
        extracted[target] = value
    return extracted


def validate_reference(values, reference=REFERENCE):
    """Check independent keys, shape, dtype on metadata OR restored arrays."""
    if set(values) != set(reference):
        raise ValueError("reference keys mismatch")
    for key, (shape, dtype) in reference.items():
        value = values[key]
        if tuple(value.shape) != shape:
            raise ValueError(f"reference shape mismatch: {key!r}")
        if str(value.dtype) != dtype:
            raise ValueError(f"reference dtype mismatch: {key!r}")


def expect_rejection(fn, reason):
    try:
        fn()
    except ValueError as error:
        assert reason in str(error), (reason, str(error))
        return str(error)
    raise AssertionError(f"expected rejection: {reason}")


@contextmanager
def trace_numpy(checkpoint, allowed, records):
    """Fail before I/O for any array outside the exact requested synthetic leaves.

    The class patch is confined to this isolated process and restored on exit.
    Records capture requested reads and completed deserializations separately.
    This witnesses logical array reads, not physical OCDBT block read ranges.
    """
    from orbax.checkpoint import type_handlers
    original = type_handlers.NumpyHandler.deserialize
    expected_names = {".".join(k) for k in allowed}

    async def traced(handler, infos, args=None):
        names = [info.name for info in infos]
        for info in infos:
            path = Path(str(info.path)).resolve()
            assert path.is_relative_to(checkpoint.resolve()), path
            assert info.name in expected_names, (info.name, expected_names)
        records["requested"].extend(names)
        result = await original(handler, infos, args)
        records["completed"].extend(names)
        return result

    with mock.patch.object(type_handlers.NumpyHandler, "deserialize", traced):
        yield


def restore_selected(reader, checkpoint, selected, records):
    import orbax.checkpoint as ocp
    # Keep complete tuple paths all the way through; selecting a params parent
    # with a single placeholder does NOT describe the desired array leaves.
    item = ocp.tree.from_flat_dict({key: 0 for key in selected})
    args = ocp.tree.from_flat_dict({
        key: ocp.RestoreArgs(restore_type=np.ndarray) for key in selected
    })
    with trace_numpy(checkpoint, selected, records):
        restored = reader.restore(checkpoint, args=ocp.args.PyTreeRestore(
            item=item, restore_args=args, transforms={}))
    actual = ocp.tree.to_flat_dict(restored)
    assert set(actual) == set(selected), (set(actual), set(selected))
    assert all(type(value) is np.ndarray for value in actual.values())
    expected = Counter(".".join(key) for key in selected)
    assert Counter(records["requested"]) == expected, records
    assert Counter(records["completed"]) == expected, records
    return actual


def empty_records():
    return {"requested": [], "completed": []}


def extraction_regressions():
    values = {
        ("params", "value", "value"): 1,
        ("params", "value_projection", "value"): 2,
        ("params", "value", "kernel", "value"): 3,
        ("params_extra", "bad", "value"): 4,
        ("optimizer", "params", "bad", "value"): 5,
        ("step",): 6,
    }
    assert extract_params(values) == {
        ("value",): 1, ("value_projection",): 2, ("value", "kernel"): 3}
    for ending in ("value_extra", "myvalue", "kernel"):
        expect_rejection(lambda: extract_params({("params", "x", ending): 1}),
                         "final value")
    expect_rejection(lambda: extract_params({("params", "value"): 1}), "final value")
    expect_rejection(lambda: extract_params({"params/x/value": 1}), "full string tuples")


def rejection_regressions(values):
    """Mutate references, not source-derived expectations; no restore is invoked."""
    shape = dict(REFERENCE)
    shape[("dense", "kernel")] = ((2, 3), "float32")
    dtype = dict(REFERENCE)
    dtype[("dense", "kernel")] = ((3, 2), "float16")
    missing = dict(REFERENCE)
    del missing[("dense", "bias")]
    extra = dict(REFERENCE)
    extra[("absent",)] = ((1,), "float32")
    return {
        "shape": expect_rejection(lambda: validate_reference(values, shape), "shape mismatch"),
        "dtype": expect_rejection(lambda: validate_reference(values, dtype), "dtype mismatch"),
        "missing_key": expect_rejection(lambda: validate_reference(values, missing), "keys mismatch"),
        "extra_key": expect_rejection(lambda: validate_reference(values, extra), "keys mismatch"),
    }


def run_layout(evidence, use_ocdbt):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import nnx

    class Tiny(nnx.Module):
        def __init__(self):
            self.dense = nnx.Dict(
                kernel=nnx.Param(jnp.asarray(np.arange(6, dtype=np.float32).reshape(3, 2))),
                bias=nnx.Param(np.array([-2, 3], dtype=np.float32)),
            )
            self.value = nnx.Param(np.array([7], dtype=np.float32))
            self.value_projection = nnx.Param(np.array([[11], [13]], dtype=np.float32))

    params = nnx.state(Tiny())
    assert isinstance(params, nnx.State)
    item = {
        "params": params,
        "optimizer": {"params": jax.tree.map(lambda x: x + np.float32(100), params),
                      "momentum": np.full((2, 2), 19, dtype=np.float32)},
        "params_extra": {"value": np.array([23], dtype=np.float32)},
        "step": np.asarray(17, dtype=np.int32),
    }
    original = ocp.tree.to_flat_dict(item)
    logical_bytes = sum(np.asarray(x).nbytes for x in original.values())
    assert logical_bytes < 1024
    expected = {("params",) + key + ("value",) for key in REFERENCE}
    assert {key for key in original if key[0] == "params"} == expected
    source_values = {k: np.asarray(v).copy() for k, v in original.items() if k in expected}
    checkpoint = evidence / ("ocdbt" if use_ocdbt else "non_ocdbt")
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt)) as reader:
        # Pass actual nnx.State, not a pre-flattened imitation, to the serializer.
        reader.save(checkpoint, args=ocp.args.PyTreeSave(item=item))
        metadata = ocp.tree.to_flat_dict(reader.metadata(checkpoint))
        assert set(metadata) == set(original)
        validate_reference(extract_params(metadata))
        metadata_rejections = rejection_regressions(extract_params(metadata))
        requested = sorted(key for key in metadata if key[0] == "params")
        assert set(requested) == expected
        reads = empty_records()
        restored = restore_selected(reader, checkpoint, requested, reads)
        validate_reference(extract_params(restored))
        for key, value in restored.items():
            np.testing.assert_array_equal(value, source_values[key])
        value_rejections = rejection_regressions(extract_params(restored))

        # Also select just one full tuple to exclude sibling params, not just
        # optimizer/step. Every leaf is independently selectable.
        per_leaf_reads = {}
        for key in requested:
            records = empty_records()
            single = restore_selected(reader, checkpoint, [key], records)
            np.testing.assert_array_equal(single[key], source_values[key])
            per_leaf_reads["/".join(key)] = records

        # The independent checks above run both before and after restore;
        # no shape or dtype coercion is requested from Orbax.
        np.testing.assert_array_equal(restored[("params", "value", "value")], [7])
        damaged = None
        if not use_ocdbt:
            optimizer_key = ("optimizer", "momentum")
            leaf_dir = checkpoint / ".".join(optimizer_key)
            # Inspect the exact synthetic leaf before renaming only that folder.
            assert checkpoint.parent == evidence
            assert leaf_dir.is_dir() and not leaf_dir.is_symlink()
            assert (leaf_dir / ".zarray").is_file()
            zarray = json.loads((leaf_dir / ".zarray").read_text())
            assert tuple(zarray["shape"]) == (2, 2)
            assert np.dtype(zarray["dtype"]) == np.dtype("float32")
            files = sorted(str(p.relative_to(leaf_dir)) for p in leaf_dir.rglob("*") if p.is_file())
            renamed = evidence / "own_optimizer_momentum_renamed"
            assert not renamed.exists()
            leaf_dir.rename(renamed)
            try:
                after_reads = empty_records()
                after = restore_selected(reader, checkpoint, requested, after_reads)
                for key in requested:
                    np.testing.assert_array_equal(after[key], source_values[key])
                optimizer_reads = empty_records()
                try:
                    restore_selected(reader, checkpoint, [optimizer_key], optimizer_reads)
                except (FileNotFoundError, ValueError) as error:
                    message = str(error)
                    assert "optimizer.momentum" in message, message
                    assert "not found" in message.lower() or "exist" in message.lower(), message
                    assert optimizer_reads["requested"] == ["optimizer.momentum"]
                    assert optimizer_reads["completed"] == []
                    damaged = {"inspected_leaf": str(leaf_dir), "files": files,
                               "metadata": zarray, "params_reads_after_rename": after_reads,
                               "optimizer_reads": optimizer_reads,
                               "expected_optimizer_failure": f"{type(error).__name__}: {message}"}
                else:
                    raise AssertionError("missing requested optimizer unexpectedly restored")
            finally:
                # Restore this test's own fixture; delete nothing.
                renamed.rename(leaf_dir)
    return {
        "use_ocdbt": use_ocdbt, "checkpoint": str(checkpoint),
        "actual_nnx_state_serialized": True, "params_leaf_count": len(requested),
        "saved_total_leaf_count": len(original), "saved_logical_bytes": logical_bytes,
        "largest_synthetic_array_bytes": max(np.asarray(x).nbytes for x in original.values()),
        "params_restore_reads": reads, "per_leaf_restore_reads": per_leaf_reads,
        "metadata_reference_rejections": metadata_rejections,
        "restored_reference_rejections": value_rejections,
        "missing_optimizer_control": damaged,
    }


def main():
    assert sys.flags.isolated and sys.flags.dont_write_bytecode, "Run with -I -B"
    assert Path(sys.prefix).name == "gen1_train_ready", sys.prefix
    assert Path(__file__).resolve().parent == ROOT / "scripts"
    assert (ROOT / "tmp").is_dir()
    evidence = Path(tempfile.mkdtemp(prefix="gen1_selective_raw_restore_", dir=ROOT / "tmp"))
    # Any library fallback temporary/cache path stays inside this fixture too.
    os.environ.update(TMPDIR=str(evidence), XDG_CACHE_HOME=str(evidence / "unused_cache"))
    tempfile.tempdir = str(evidence)
    print(f"SYNTHETIC_FIXTURE={evidence}", flush=True)
    report = {"status": "failed", "evidence_dir": str(evidence), "python": sys.executable,
              "isolated": bool(sys.flags.isolated), "bytecode_disabled": bool(sys.flags.dont_write_bytecode),
              "real_checkpoint_read": False, "results": [],
              "limitations": ["Tiny synthetic compatibility only; no large-weight memory or latency claim.",
                              "Deserialize tracing proves logical leaves, not OCDBT physical read ranges.",
                              "This standalone test does not modify or validate the frozen production loader."]}
    try:
        global np
        import numpy as np
        import jax
        jax.config.update("jax_enable_compilation_cache", False)
        report["orbax_version"] = importlib.metadata.version("orbax-checkpoint")
        assert report["orbax_version"] == "0.11.13"
        report["jax_devices"] = [str(device) for device in jax.devices()]
        assert all(device.platform == "cpu" for device in jax.devices())
        assert not jax.config.jax_enable_compilation_cache
        extraction_regressions()
        report["exact_final_value_suffix_regressions"] = "passed"
        for use_ocdbt in (False, True):
            report["results"].append(run_layout(evidence, use_ocdbt))
            print(f"PASS use_ocdbt={use_ocdbt}: params-only, 4 single-leaf selections, reference rejection", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["fixture_file_bytes"] = sum(p.stat().st_size for p in evidence.rglob("*") if p.is_file())
    with (evidence / "result.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
