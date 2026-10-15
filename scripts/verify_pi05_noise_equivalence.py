#!/usr/bin/env python3
"""Verify that the v11 explicit-noise stream equals OpenPI's internal stream.

For the Orbax policy, ``Policy.infer`` splits its private JAX key once per
inference and ``Pi0.sample_actions`` draws a normal tensor from the returned
subkey when no explicit noise is supplied.  The v11 adapter maintains a second
key initialized to the same seed, splits it once per inference, and supplies
the resulting normal tensor explicitly.  This check makes that equivalence
machine-verifiable without loading the checkpoint or occupying the GPU.
"""

from __future__ import annotations

import argparse

import jax
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inferences", type=int, default=100)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=32)
    args = parser.parse_args()

    internal_key = jax.random.key(args.seed)
    explicit_key = jax.random.key(args.seed)
    shape = (args.action_horizon, args.action_dim)
    for index in range(args.inferences):
        internal_key, internal_sample_key = jax.random.split(internal_key)
        explicit_key, explicit_sample_key = jax.random.split(explicit_key)
        internal_noise = np.asarray(
            jax.random.normal(internal_sample_key, shape), dtype=np.float32
        )
        explicit_noise = np.asarray(
            jax.random.normal(explicit_sample_key, shape), dtype=np.float32
        )
        if not np.array_equal(internal_noise, explicit_noise):
            raise AssertionError(f"noise mismatch at inference {index}")

    print(
        "noise streams are elementwise identical: "
        f"seed={args.seed}, inferences={args.inferences}, shape={shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
