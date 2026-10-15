#!/usr/bin/env python3
"""Run the bounded D013 pi05 training smoke and capture raw evidence."""

from __future__ import annotations

import re
import shutil
import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/root/autodl-tmp/EvolveVLA")
LOG_PATH = ROOT / "results/d013_smoke_training_20261001.log"
EXIT_PATH = ROOT / "results/d013_smoke_training_20261001.exit"
VRAM_PATH = ROOT / "results/d013_smoke_training_20261001.vram"
OUTPUT_DIR = ROOT / "checkpoints/d013_smoketest"
TRAIN = Path("/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/lerobot-train")
BASE = Path("/root/autodl-tmp/.cache/huggingface/hub/models--lerobot--pi05_base")
DATASET = ROOT / "data/smoke_lerobot"

TARGET_MODULES = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
)

COMMAND = [
    str(TRAIN),
    "--dataset.repo_id=robocasa/d013_smoke_only",
    f"--dataset.root={DATASET}",
    f"--policy.path={BASE}",
    "--policy.device=cuda:0",
    "--policy.push_to_hub=false",
    "--rename_map="
    + json.dumps(
        {
            "observation.images.robot0_agentview_left": "observation.images.base_0_rgb",
            "observation.images.robot0_agentview_right": "observation.images.right_wrist_0_rgb",
            "observation.images.robot0_eye_in_hand": "observation.images.left_wrist_0_rgb",
        },
        separators=(",", ":"),
    ),
    "--steps=5",
    "--batch_size=1",
    "--num_workers=0",
    "--persistent_workers=false",
    f"--output_dir={OUTPUT_DIR}",
    "--job_name=d013_pi05_smoke_only",
    "--save_freq=5",
    "--wandb.enable=false",
    "--save_checkpoint_to_hub=false",
    f"--peft.target_modules={TARGET_MODULES}",
    "--peft.full_training_modules=[]",
    "--peft.method_type=LORA",
    "--peft.r=16",
    "--peft.lora_alpha=32",
]


def gpu_memory_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    values: list[int] = []
    for line in result.stdout.splitlines():
        match = re.match(r"\s*0\s*,\s*(\d+)\s*$", line)
        if match:
            values.append(int(match.group(1)))
    return max(values) if values else None


def main() -> int:
    ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_PATH.unlink(missing_ok=True)
    EXIT_PATH.unlink(missing_ok=True)
    VRAM_PATH.unlink(missing_ok=True)

    print("SMOKE_COMMAND " + " ".join(COMMAND), flush=True)
    with LOG_PATH.open("w", encoding="utf-8") as log, VRAM_PATH.open("w", encoding="utf-8") as vram:
        log.write("SMOKE_COMMAND " + " ".join(COMMAND) + "\n")
        log.flush()
        process = subprocess.Popen(
            COMMAND,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        peak = 0
        while True:
            current = gpu_memory_mib()
            if current is not None:
                peak = max(peak, current)
                vram.write(f"{time.time():.3f} memory_used_mib={current}\n")
                vram.flush()
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.25)
        for line in process.stdout or []:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
        vram.write(f"peak_memory_used_mib={peak}\n")
        vram.flush()
        log.write(f"SMOKE_RETURN_CODE {return_code}\n")
        log.write(f"SMOKE_PEAK_VRAM_MIB {peak}\n")
    EXIT_PATH.write_text(f"{return_code}\n", encoding="utf-8")
    print(f"SMOKE_RETURN_CODE {return_code}")
    print(f"SMOKE_PEAK_VRAM_MIB {peak}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
