# Environment Specification — EvolveVLA
# Reproduction-grade environment details (verified 2026-09-30)

## Hardware
- GPU: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (102GB usable VRAM)
- RAM: 1.0 TiB, CPU: 208 cores
- Server: RTX_PRO_6000 (Ubuntu 22.04.5 LTS, kernel 5.15.0-78-generic)
- Driver: 590.44.01 | CUDA toolkit: 12.8 | GPU compute capability: sm_120

## Conda Environment
- Name: `evolvevla` | Python: 3.11.16
- Activate: `source /root/miniconda3/bin/activate && conda activate evolvevla`

## Verified Package Versions (WORKING)
| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.11.0+cu128 | sm_120 Blackwell kernels verified |
| torchvision | 0.26.0+cu128 | |
| numpy | 2.2.5 | RoboCasa-compatible pin |
| scipy | 1.15.3 | RoboCasa-compatible pin (_promote API) |
| numba | 0.61.2 | RoboCasa-compatible pin |
| mujoco | 3.3.1 | |
| robosuite | 1.5.2 | installed from robosuite master commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| robocasa | 1.0.1 | installed from GitHub source |
| lerobot | 0.3.3 | |
| peft | 0.21.1 | |
| transformers | 4.53.2 | pinned per community guidance |
| accelerate | 1.15.0 | |
| safetensors | 0.8.0 | |

## RoboSuite Source Revision
The installed package metadata reports version `1.5.2`, while its source was installed from
`/root/autodl-tmp/robosuite-master-filtered` at commit
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.

## Environment Variables (required)
```bash
source /etc/network_turbo 2>/dev/null          # AutoDL proxy (for downloads)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

## Key Paths
- Project: /root/autodl-tmp/EvolveVLA/
- π0.5 base checkpoint: /root/autodl-tmp/.cache/huggingface/hub/models--lerobot--pi05_base/ (14GB)
- RoboCasa assets: /root/autodl-tmp/EvolveVLA/robocasa_assets/ (21GB)
- Code: /root/autodl-tmp/EvolveVLA/evolvevla/ + scripts/

## RoboCasa Configuration Notes
- Task naming: use `robocasa/PickPlaceCounterToCabinet` (gymnasium API)
- `obj_registries=('lightwheel',)` required — full objaverse/aigen_objs registries not fully installed
- `split='pretrain'` for leaderboard-comparable evaluation

## Disk Budget
- Root overlay (/): 30GB — packages only. ~18GB used (60%)
- /root/autodl-tmp/: 550GB — project data. ~44GB used (8%)
