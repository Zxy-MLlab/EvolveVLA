#!/usr/bin/env bash
set -euo pipefail

# The Python entrypoint accepts --instruction-source and --cam-profile in
# addition to the existing launcher arguments; this wrapper forwards all args.
exec /root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python \
  /root/autodl-tmp/EvolveVLA/scripts/gate3_full_ab_rollout.py "$@"
