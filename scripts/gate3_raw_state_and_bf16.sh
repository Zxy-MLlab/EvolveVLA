#!/usr/bin/env bash
set -u

cd /root/autodl-tmp/EvolveVLA
PYTHON=/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python
export PYTHONUNBUFFERED=1
export HEARTBEAT_LOG=/root/runs/zhouxueyangapexinai/gate3_raw_state_and_bf16.log

"$PYTHON" -m py_compile scripts/gate3_raw_state_and_bf16.py
compile_exit=$?
printf 'SMOKE_PY_COMPILE_EXIT=%s\n' "$compile_exit"
if [ "$compile_exit" -ne 0 ]; then
    exit "$compile_exit"
fi

"$PYTHON" -c 'import gate3_teacher_forcing as tf; import gate3_two_checkpoint_discriminator as d; import gate3_weight_swap_fixed_config as w; checks=[(d,"load_harness"),(d,"load_sample"),(tf,"make_observation"),(w,"run_oracle")]; h=d.load_harness(); checks += [(h,"load_policy"),(h,"make_policy_batch"),(h,"simulator_observation")]; missing=[f"{m.__name__}.{a}" for m,a in checks if not hasattr(m,a)]; print("HASATTR_CHECKS=" + ("PASS" if not missing else "FAIL:" + ",".join(missing))); raise SystemExit(0 if not missing else 1)'
hasattr_exit=$?
printf 'SMOKE_HASATTR_EXIT=%s\n' "$hasattr_exit"
if [ "$hasattr_exit" -ne 0 ]; then
    exit "$hasattr_exit"
fi

exec "$PYTHON" scripts/gate3_raw_state_and_bf16.py
