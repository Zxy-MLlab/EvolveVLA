#!/usr/bin/env bash
set -u
ROOT=/root/autodl-tmp/EvolveVLA
LOG=$ROOT/results/pi05_public_baseline/stage0_supervisor.log
exec >>"$LOG" 2>&1
while [ ! -f "$ROOT/artifacts/gen0/baseline_report.md" ]; do
  if ! pgrep -f "bash $ROOT/scripts/run_stage0_pipeline.sh 393247" >/dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) pipeline absent; restarting with resume"
    setsid -f sh -c "export SKIP_COVERAGE=1; exec $ROOT/scripts/run_stage0_pipeline.sh 393247" >/tmp/evolvevla_stage0_pipeline.nohup 2>&1
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) Generation-0 exists; supervisor exiting"
