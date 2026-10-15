#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path("/root/autodl-tmp/EvolveVLA")
DEFAULT_CKPT = ROOT / "checkpoints/pi05_robotwin_lerobot"
CKPT = DEFAULT_CKPT
DATA = ROOT / "data/gate2_robotwin_cache/clean"
OPENPI = Path("/root/autodl-tmp/openpi_feasibility_src/src")
TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer/tokenizer.model")
EPISODE_INDEX = 50
POSITION = 31
VIDEO_INDEX = 7219
NUM_STEPS = 10
MODEL_DTYPE = "float32"
EXPECTED_IMAGE_HASHES = {
    "cam_high": "6813d9707236c51f5f65e2ecacd028c5c5ca21c863dc71586e74905071f66f04",
    "cam_left_wrist": "f7f91a608a5506567258e9b342b4555a2ec4b91ab306bf3407e5ec7c23bab707",
    "cam_right_wrist": "6b580e8fd7f49b1bf4a588cbede26aaa76b9bca27717e6ed09d1525460fbc705",
}
STATE = np.array(
    [-0.78727591, 2.11586475, 1.90931702, -1.21535110, -0.14554431,
     -0.76654691, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    dtype=np.float32,
)
TASK = "Utilize the left arm to hold the grippy handle hammer and strike."
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
NORM_MODE = "quantiles"


def resolve_checkpoint(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def frame(path, index):
    import av

    with av.open(str(path)) as container:
        for i, decoded_frame in enumerate(container.decode(video=0)):
            if i == index:
                image = decoded_frame.to_ndarray(format="rgb24")
                expected = 480 * 640 * 3
                if image.nbytes != expected:
                    raise RuntimeError(
                        f"unexpected frame bytes={image.nbytes} expected={expected}"
                    )
                return image
    raise RuntimeError(f"missing frame {index} in {path}")


def _tensor(path, key):
    payload = load_file(str(path), device="cpu")
    if key not in payload:
        raise RuntimeError(f"normalizer state missing key={key} file={path}")
    return payload[key].detach().cpu().numpy().astype(np.float32)


def _stats_from_lerobot_normalizers(checkpoint):
    pre = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not pre.exists() or not post.exists():
        raise RuntimeError(
            "MEAN_STD requested but LeRobot normalizer safetensors are absent: "
            f"pre={pre.exists()} post={post.exists()}"
        )
    return {
        "state": {
            "mean": _tensor(pre, "observation.state.mean"),
            "std": _tensor(pre, "observation.state.std"),
        },
        "actions": {
            "mean": _tensor(post, "action.mean"),
            "std": _tensor(post, "action.std"),
        },
        "_source": "lerobot_processor_safetensors",
    }


def stats(checkpoint=None, norm_mode=None):
    checkpoint = resolve_checkpoint(checkpoint or CKPT)
    norm_mode = norm_mode or NORM_MODE
    norm_json = checkpoint / "norm_stats.json"
    if norm_json.exists():
        payload = json.loads(norm_json.read_text())
        if "norm_stats" not in payload:
            raise RuntimeError("norm_stats.json missing top-level norm_stats object")
        result = payload["norm_stats"]
        if "state" not in result or "actions" not in result:
            raise RuntimeError("norm_stats.json norm_stats object missing state/actions")
        result["_source"] = "norm_stats.json"
    else:
        result = _stats_from_lerobot_normalizers(checkpoint)

    result["_checkpoint"] = str(checkpoint)
    result["_norm_mode"] = norm_mode
    return result


def _require_vector(st, section, key, shape):
    if section not in st or key not in st[section]:
        raise RuntimeError(f"normalization stats missing {section}.{key}")
    value = np.asarray(st[section][key], dtype=np.float32)
    if value.shape != shape:
        raise RuntimeError(
            f"{section}.{key} shape mismatch actual={value.shape} expected={shape}"
        )
    return value


def normalize_state(state, st):
    mode = st.get("_norm_mode", NORM_MODE)
    if mode == "quantiles":
        q01 = _require_vector(st, "state", "q01", state.shape)
        q99 = _require_vector(st, "state", "q99", state.shape)
        if np.any(q99 <= q01):
            raise RuntimeError("state q99 must be greater than q01")
        return (2.0 * (state - q01) / (q99 - q01) - 1.0).astype(np.float32)
    if mode == "mean_std":
        mean = _require_vector(st, "state", "mean", state.shape)
        std = _require_vector(st, "state", "std", state.shape)
        if np.any(std <= 0):
            raise RuntimeError("state std must be positive")
        return ((state - mean) / std).astype(np.float32)
    raise RuntimeError(f"unsupported norm mode: {mode}")


def unnormalize_action(action, st):
    mode = st.get("_norm_mode", NORM_MODE)
    if mode == "quantiles":
        q01 = _require_vector(st, "actions", "q01", action.shape)
        q99 = _require_vector(st, "actions", "q99", action.shape)
        if np.any(q99 <= q01):
            raise RuntimeError("action q99 must be greater than q01")
        return ((action + 1.0) * (q99 - q01) / 2.0 + q01).astype(np.float32)
    if mode == "mean_std":
        mean = _require_vector(st, "actions", "mean", action.shape)
        std = _require_vector(st, "actions", "std", action.shape)
        if np.any(std <= 0):
            raise RuntimeError("action std must be positive")
        return (action * std + mean).astype(np.float32)
    raise RuntimeError(f"unsupported norm mode: {mode}")


def load_weights(model, checkpoint=None):
    checkpoint = resolve_checkpoint(checkpoint or CKPT)
    path = checkpoint / "model.safetensors"
    try:
        sd = load_file(str(path), device="cpu")
        report = model.load_state_dict(sd, strict=False)
    except Exception as exc:
        raise RuntimeError(f"key_mismatch: {type(exc).__name__}: {exc}") from exc

    missing = sorted(report.missing_keys)
    unexpected = sorted(report.unexpected_keys)
    allowed_missing = {
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        "paligemma_with_expert.paligemma.lm_head.weight",
    }
    if not set(missing).issubset(allowed_missing) or unexpected:
        raise RuntimeError(f"key_mismatch: missing={missing} unexpected={unexpected}")

    loaded_keys = len(sd) - len(unexpected)
    return loaded_keys


def resolve_matched_observation(st):
    image_paths = {
        "cam_high": DATA / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
        "cam_left_wrist": DATA / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
        "cam_right_wrist": DATA / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
    }
    images = {}
    image_hashes = {}
    for camera, path in image_paths.items():
        image = frame(path, VIDEO_INDEX)
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        print(f"RAW_FRAME_HASH video_frame={VIDEO_INDEX} camera={camera} sha256={image_hash}")
        expected = EXPECTED_IMAGE_HASHES[camera]
        if image_hash != expected:
            raise AssertionError(
                f"raw decoded frame hash mismatch camera={camera} actual={image_hash} expected={expected}"
            )
        image_hashes[camera] = image_hash
        images[camera] = image

    normalized_state = normalize_state(STATE, st)
    return images, image_hashes, normalized_state


def one_obs_verbose(st):
    checkpoint = st.get("_checkpoint", str(CKPT))
    norm_mode = st.get("_norm_mode", NORM_MODE)
    label = "PEER" if "sidneyxie_pi05_robotwin" in checkpoint else "REFERENCE"
    print(
        f"REFERENCE_ONE_OBS_CONFIG dtype={MODEL_DTYPE} num_steps={NUM_STEPS} "
        f"device=cpu model_loaded=0 forward=0 checkpoint={checkpoint}"
    )
    print(
        f"RESOLVED_OBS episode_index={EPISODE_INDEX} position={POSITION} "
        f"video_index={VIDEO_INDEX}"
    )
    images, image_hashes, normalized_state = resolve_matched_observation(st)
    print(f"NORMALIZATION_SOURCE={st.get('_source', 'unknown')}")
    print(f"NORMALIZATION_APPLIED={norm_mode.upper()}")
    print(f"{label}_NORMALIZATION_APPLIED={norm_mode.upper()}")
    print(f"STATE_SHAPE={normalized_state.shape}")
    print(f"BATCH_STATE_SHAPE={(1, normalized_state.shape[0])}")
    print(f"RAW_FRAME_HASH_ASSERT video_frame={VIDEO_INDEX} hashes={image_hashes} expected={EXPECTED_IMAGE_HASHES}")
    print("REFERENCE_INPUT=RAW decoded 7219 frame; preprocessing=OpenPI-owned (not OUR bridge)")
    print(f"{label}_REFERENCE_ONE_OBS_READY=PASS")


def openpi_forward(device, st):
    if str(OPENPI) not in sys.path:
        sys.path.insert(0, str(OPENPI))

    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.shared.image_tools import resize_with_pad_torch

    checkpoint = resolve_checkpoint(st.get("_checkpoint", CKPT))
    normalized_state = normalize_state(STATE, st)
    model = PI0Pytorch(Pi0Config(pi05=True, dtype=MODEL_DTYPE))
    loaded_keys = load_weights(model, checkpoint)
    model = model.to(device).eval()

    import sentencepiece

    tok = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tok._max_len = 200
    with TOKENIZER.open("rb") as f:
        tok._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    tokens, token_mask = tok.tokenize(TASK, normalized_state)

    raw_images, _, _ = resolve_matched_observation(st)
    image_key_map = {
        "cam_high": "base_0_rgb",
        "cam_left_wrist": "left_wrist_0_rgb",
        "cam_right_wrist": "right_wrist_0_rgb",
    }
    images = {}
    for camera, image_key in image_key_map.items():
        img = torch.from_numpy(raw_images[camera]).float() / 255.0 * 2.0 - 1.0
        img = img.unsqueeze(0)
        resized = resize_with_pad_torch(img, 224, 224)
        resized = resized.unsqueeze(0).permute(0, 3, 1, 2)
        images[image_key] = resized.to(device=device, dtype=torch.float32)

    obs = Observation.from_dict({
        "image": images,
        "image_mask": {
            key: torch.ones(1, dtype=torch.bool, device=device) for key in images
        },
        "state": torch.from_numpy(normalized_state).view(1, -1).to(device=device, dtype=torch.float32),
        "tokenized_prompt": torch.from_numpy(tokens).view(1, -1).to(device=device, dtype=torch.long),
        "tokenized_prompt_mask": torch.from_numpy(token_mask).view(1, -1).to(
            device=device, dtype=torch.bool
        ),
    })

    for key in images:
        print(
            f"IMAGE_CHECK key={key} shape={tuple(images[key].shape)} "
            f"dtype={images[key].dtype} min={float(images[key].min()):.4f} "
            f"max={float(images[key].max()):.4f}"
        )

    for key in images:
        print(
            f"IMAGE_CHECK after_observation key={key} shape={tuple(obs.images[key].shape)} "
            f"dtype={obs.images[key].dtype} min={float(obs.images[key].min()):.4f} "
            f"max={float(obs.images[key].max()):.4f}"
        )

    preprocessed_images, _, _, _, _ = model._preprocess_observation(obs, train=False)
    for key, image in zip(images, preprocessed_images, strict=True):
        print(
            f"IMAGE_CHECK after_preprocess key={key} shape={tuple(image.shape)} "
            f"dtype={image.dtype} min={float(image.min()):.4f} "
            f"max={float(image.max()):.4f}"
        )

    torch.manual_seed(0)
    actions = model.sample_actions(device, obs, noise=None, num_steps=NUM_STEPS)
    first_action = actions[0, 0, :14].detach().float().cpu().numpy()
    first_action = unnormalize_action(first_action, st)
    arm_l2 = float(np.linalg.norm(first_action[ARM_DIMS]))
    return arm_l2, loaded_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-obs-verbose", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_CKPT))
    parser.add_argument("--norm-mode", choices=("quantiles", "mean_std"), default="quantiles")
    args = parser.parse_args()
    try:
        global CKPT, NORM_MODE
        CKPT = resolve_checkpoint(args.model_path)
        NORM_MODE = args.norm_mode
        if str(OPENPI) not in sys.path:
            sys.path.insert(0, str(OPENPI))
        st = stats(CKPT, NORM_MODE)
        if args.one_obs_verbose:
            one_obs_verbose(st)
            return 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arm_l2, loaded_keys = openpi_forward(device, st)
        print(
            "OPENPI_PATH_OK "
            f"model=PI0Pytorch checkpoint={CKPT} norm_mode={NORM_MODE} "
            f"loaded_keys={loaded_keys} "
            "preprocessing=openpi.resize_with_pad_torch,openpi.PaligemmaTokenizer,openpi.Observation.from_dict "
            f"dtype={MODEL_DTYPE} num_steps={NUM_STEPS}"
        )
        print(f"ARM_L2={arm_l2:.8f}")
        print(f"REFERENCE_ARM_L2={arm_l2:.8f}")
        if "sidneyxie_pi05_robotwin" in str(CKPT):
            print(f"PEER_REFERENCE_ARM_L2={arm_l2:.8f}")
        return 0
    except Exception as exc:
        print(traceback.format_exc(), end="")
        print(f"REFERENCE_BLOCKER={type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
