#!/usr/bin/env python3
"""D171 bounded arm scale-versus-trajectory diagnostic.

Real mode uses the pinned serving path on cached RoboTwin observations only.
It does not create a simulator or run a rollout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
OFFICIAL_CHECKPOINT = ROOT / "checkpoints/pi05_robotwin_lerobot"
PEER_CHECKPOINT = ROOT / "checkpoints/sidneyxie_pi05_robotwin"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
TASK = "beat_block_hammer"
EPISODE = 50
START_POSITION = 31
CHUNK_SIZE = 50
ACTION_DIM = 14
ARM_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64
)
SCALE_COV_THRESHOLD = 0.30
TRAJECTORY_COS_THRESHOLD = 0.20
WATCHDOG_STALE_SECONDS = 300.0
HEARTBEAT_EVERY_STEPS = 10
OFFICIAL_HALF_PATH = ROOT / "results/d171_official_half.json"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def heartbeat(label: str) -> None:
    path_text = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not path_text:
        return
    path = Path(path_text)
    if not path.is_absolute():
        raise RuntimeError("ROBOTWIN_HEARTBEAT_LOG must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    print(f"HEARTBEAT={label}", flush=True)


def start_watchdog() -> tuple[threading.Event, threading.Thread]:
    path_text = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not path_text or not Path(path_text).is_absolute():
        raise RuntimeError("ROBOTWIN_HEARTBEAT_LOG must be an absolute path")
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(30.0):
            try:
                age = time.time() - path.stat().st_mtime
            except OSError as exc:
                print(f"WATCHDOG_ERROR={type(exc).__name__}: {exc}", flush=True)
                os._exit(124)
            if age > WATCHDOG_STALE_SECONDS:
                print(f"WATCHDOG_STALE_LOG age={age:.1f}s path={path}", flush=True)
                os._exit(124)

    thread = threading.Thread(target=monitor, name="d171-log-watchdog", daemon=True)
    thread.start()
    return stop, thread


def cosine_rows(predicted: np.ndarray, recorded: np.ndarray) -> np.ndarray:
    pred = predicted[:, ARM_INDICES].astype(np.float64)
    target = recorded[:, ARM_INDICES].astype(np.float64)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    dots = np.sum(pred * target, axis=1)
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 1e-12)


def scale_probe(official: np.ndarray, peer: np.ndarray) -> tuple[str, float, np.ndarray]:
    official_mag = np.mean(np.abs(official[:, ARM_INDICES]), axis=0)
    peer_mag = np.mean(np.abs(peer[:, ARM_INDICES]), axis=0)
    ratio = np.divide(
        official_mag,
        peer_mag,
        out=np.full_like(official_mag, np.nan, dtype=np.float64),
        where=peer_mag > 1e-12,
    )
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        raise RuntimeError("official/peer arm magnitude ratios are all non-finite")
    cov = float(np.std(finite) / max(abs(float(np.mean(finite))), 1e-12))
    verdict = "uniform" if cov < SCALE_COV_THRESHOLD else "concentrated"
    print("SCALE_TABLE_SHAPE=(12, 4)")
    print("dim official_mean_abs peer_mean_abs official_over_peer")
    for dim, off, peer_value, ratio_value in zip(
        ARM_INDICES, official_mag, peer_mag, ratio
    ):
        print(f"{int(dim)} {off:.8f} {peer_value:.8f} {ratio_value:.8f}")
    cosmetic_gain = float(np.mean(peer_mag) / max(np.mean(official_mag), 1e-12))
    print(f"SCALE_COV={cov:.8f} threshold={SCALE_COV_THRESHOLD:.2f}")
    print(f"COSMETIC_GAIN_PEER_OVER_OFFICIAL={cosmetic_gain:.8f}")
    print("SCALE_STATS=official_quantiles_peer_mean_std_untouched")
    return verdict, cov, ratio


def trajectory_probe(
    official_chunks: list[np.ndarray],
    peer_chunks: list[np.ndarray],
    recorded_chunks: list[np.ndarray],
) -> str:
    official_means = []
    peer_means = []
    for index, (official, peer, recorded) in enumerate(
        zip(official_chunks, peer_chunks, recorded_chunks)
    ):
        official_cos = cosine_rows(official, recorded)
        peer_cos = cosine_rows(peer, recorded)
        off_mean = float(np.mean(official_cos))
        peer_mean = float(np.mean(peer_cos))
        official_means.append(off_mean)
        peer_means.append(peer_mean)
        print(
            f"TRAJ_OBS[{index}] cos_official={off_mean:.8f} "
            f"cos_peer={peer_mean:.8f}"
        )
    for label, values in (("official", official_means), ("peer", peer_means)):
        array = np.asarray(values, dtype=np.float64)
        print(
            f"TRAJ_SUMMARY_{label}=mean:{np.mean(array):.8f} "
            f"median:{np.median(array):.8f} spread:{np.ptp(array):.8f}"
        )
    official_mean = float(np.mean(official_means))
    return "tracks" if official_mean > TRAJECTORY_COS_THRESHOLD else "decorrelated"


def read_cached_rows(probe: Any) -> dict[int, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        probe.DATA_PATH,
        columns=[
            "observation.state",
            "action",
            "episode_index",
            "index",
            "task_index",
            "frame_index",
            "timestamp",
        ],
    ).to_pydict()
    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for index in range(len(table["episode_index"])):
        episode = int(table["episode_index"][index])
        rows_by_episode.setdefault(episode, []).append(
            {key: table[key][index] for key in table}
        )
    for rows in rows_by_episode.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return rows_by_episode


def choose_chunk_positions(
    rows_by_episode: dict[int, list[dict[str, Any]]], count: int
) -> tuple[str, list[tuple[int, int, int]]]:
    target = count
    episode_rows = rows_by_episode.get(EPISODE, [])
    max_position = len(episode_rows) - CHUNK_SIZE
    if max_position >= START_POSITION:
        available = max_position - START_POSITION + 1
        if available >= 4:
            selected_count = min(target, available)
            positions = np.linspace(
                START_POSITION,
                max_position,
                num=selected_count,
                dtype=np.int64,
            ).tolist()
            return (
                "a_overlapping_ep50",
                [
                    (
                        EPISODE,
                        int(position),
                        len(episode_rows[int(position) : int(position) + CHUNK_SIZE]),
                    )
                    for position in positions
                ],
            )

    candidates = [
        (episode, len(rows) - CHUNK_SIZE)
        for episode, rows in sorted(rows_by_episode.items())
        if len(rows) >= CHUNK_SIZE
    ]
    selections: list[tuple[int, int, int]] = []
    for index in range(min(target, len(candidates))):
        episode, max_start = candidates[index]
        position = min(START_POSITION, max_start)
        selections.append(
            (episode, position, len(rows_by_episode[episode][position : position + CHUNK_SIZE]))
        )
    return "b_multi_episode", selections


def print_selection_proof(
    strategy: str, selections: list[tuple[int, int, int]]
) -> None:
    print(f"TRAJECTORY_SELECTION_STRATEGY={strategy}")
    print(f"TRAJECTORY_SELECTION={selections}")
    print(f"TRAJECTORY_SELECTION_N={len(selections)}")
    if len(selections) < 4:
        print(f"TRAJECTORY_LOW_POWER_N={len(selections)}")


def synthetic_dry_run(args: argparse.Namespace) -> int:
    probe = load_module(
        SCRIPTS / "gate3_camera_permutation_repro.py",
        "d171_camera_permutation_dry_run_loader",
    )
    rows_by_episode = read_cached_rows(probe)
    strategy, selections = choose_chunk_positions(rows_by_episode, args.obs_count)
    print_selection_proof(strategy, selections)

    rng = np.random.default_rng(171)
    peer = rng.normal(0.0, 1.0, size=(CHUNK_SIZE, ACTION_DIM))
    official = 0.25 * peer
    recorded = peer.copy()
    scale_verdict, _, _ = scale_probe(official, peer)
    trajectory_verdict = trajectory_probe(
        [official.copy() for _ in selections],
        [peer.copy() for _ in selections],
        [recorded.copy() for _ in selections],
    )
    print("DRY_RUN=synthetic_action_arrays_no_model_load_dataset_selector_read")
    print(f"DRY_RUN_PER_DIM_TABLE_SHAPE=(12, 4)")
    print(f"DRY_RUN_TRAJECTORY_OBS_LINES={len(selections)}")
    print(f"SCALE={scale_verdict}")
    print(f"TRAJECTORY={trajectory_verdict}")
    return 0


def load_cached_observations(probe: Any, count: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows_by_episode = read_cached_rows(probe)
    strategy, selections = choose_chunk_positions(rows_by_episode, count)
    print_selection_proof(strategy, selections)
    tasks = pq.read_table(probe.TASKS_PATH).to_pydict()
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    observations = []
    for episode, position, expected_count in selections:
        rows = rows_by_episode[episode]
        chunk_rows = rows[position : position + expected_count]
        if len(chunk_rows) != expected_count:
            raise RuntimeError(
                f"missing {expected_count}-row chunk at episode "
                f"{episode} position {position}"
            )
        first = chunk_rows[0]
        state = np.asarray(first["observation.state"], dtype=np.float32).reshape(-1)
        recorded = np.asarray(
            [np.asarray(row["action"], dtype=np.float32).reshape(-1) for row in chunk_rows]
        )
        if state.shape != (ACTION_DIM,) or recorded.shape != (expected_count, ACTION_DIM):
            raise RuntimeError(
                f"invalid cached observation at episode {episode} "
                f"position {position}"
            )
        video_index = int(first["index"])
        frames = {
            camera: probe.decode_frame(path, video_index)
            for camera, path in probe.CAMERA_FILES.items()
        }
        observations.append(
            {
                "episode": episode,
                "position": position,
                "video_index": video_index,
                "instruction": str(task_text[int(first["task_index"])]),
                "state": state,
                "recorded": recorded,
                "frames": frames,
            }
        )
    print(
        f"TRAJECTORY_OBS_SOURCE=cached_dataset_varied_scene_states "
        f"count={len(observations)} simulator_rollout=SKIPPED"
    )
    return observations


def predict_action_chunk_with_heartbeat(
    policy: Any,
    batch: dict[str, Any],
    postprocessor: Any,
    env_postprocessor: Any,
    heartbeat_label: str,
) -> np.ndarray:
    from lerobot.utils.constants import ACTION
    import torch

    policy.reset()
    n_steps = int(
        getattr(
            policy.config,
            "n_action_steps",
            getattr(policy.config, "chunk_size", CHUNK_SIZE),
        )
    )
    if n_steps <= 0:
        raise RuntimeError(f"invalid action horizon: {n_steps}")
    predicted: list[np.ndarray] = []
    with torch.inference_mode():
        for step in range(n_steps):
            action = policy.select_action(batch)
            action = postprocessor(action)
            action = env_postprocessor({ACTION: action})[ACTION]
            array = action.detach().float().cpu().numpy().reshape(-1)
            if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
                raise RuntimeError(f"invalid predicted action: {array.shape}")
            predicted.append(array)
            if (step + 1) % HEARTBEAT_EVERY_STEPS == 0 or step + 1 == n_steps:
                heartbeat(f"{heartbeat_label}_step_{step + 1}_of_{n_steps}")
    return np.stack(predicted, axis=0)


def predict_for_observation(
    official_module: Any,
    harness: Any,
    probe: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    sample: dict[str, Any],
    heartbeat_label: str,
) -> np.ndarray:
    batch = official_module.make_observation_and_batch(
        harness,
        probe,
        sample,
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    predicted = predict_action_chunk_with_heartbeat(
        policy,
        batch,
        postprocessor,
        env_postprocessor,
        heartbeat_label,
    )
    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.shape != (CHUNK_SIZE, ACTION_DIM) or not np.isfinite(predicted).all():
        raise RuntimeError(f"invalid predicted action chunk: {predicted.shape}")
    return predicted


def persist_official_half(
    official_chunks: list[np.ndarray],
    recorded_chunks: list[np.ndarray],
) -> None:
    if len(official_chunks) != len(recorded_chunks):
        raise RuntimeError("official/recorded chunk counts differ")
    per_dim = [
        np.mean(np.abs(chunk[:, ARM_INDICES]), axis=0).tolist()
        for chunk in official_chunks
    ]
    trajectory_cosines = [
        cosine_rows(chunk, recorded).tolist()
        for chunk, recorded in zip(official_chunks, recorded_chunks)
    ]
    all_official = np.concatenate(official_chunks, axis=0)
    payload = {
        "schema": "d173_official_half_v1",
        "obs_count": len(official_chunks),
        "chunk_shape": list(official_chunks[0].shape) if official_chunks else [],
        "arm_indices": ARM_INDICES.tolist(),
        "official_chunks": [chunk.tolist() for chunk in official_chunks],
        "official_arm_mean_abs_by_dim": per_dim,
        "official_arm_mean_abs_by_dim_aggregate": np.mean(
            np.abs(all_official[:, ARM_INDICES]), axis=0
        ).tolist(),
        "official_trajectory_cosines": trajectory_cosines,
        "official_trajectory_cosine_means": [
            float(np.mean(values)) for values in trajectory_cosines
        ],
        "caveats": {
            "overlap": "selection strategy may use overlapping cached ep50 chunks",
            "static_only": "cached action chunks do not measure closed-loop grasp timing",
        },
    }
    OFFICIAL_HALF_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFICIAL_HALF_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"OFFICIAL_HALF_PATH={OFFICIAL_HALF_PATH}")


def run_real(args: argparse.Namespace) -> int:
    stop, thread = start_watchdog()
    try:
        heartbeat("before_loader_import")
        official_module = load_module(
            SCRIPTS / "gate3_official_serving_adjudicate.py",
            "d171_official_serving_loader",
        )
        probe = load_module(
            SCRIPTS / "gate3_camera_permutation_repro.py",
            "d171_camera_permutation_loader",
        )
        harness = official_module.load_harness()
        base = probe.load_episode()
        if base["episode"] != EPISODE or base["position"] != START_POSITION:
            raise RuntimeError("pinned base observation coordinates changed")
        observations = load_cached_observations(probe, args.obs_count)
        heartbeat("before_policy_load_official")
        official_policy, official_pre, official_post, official_env_pre, official_env_post = (
            harness.load_policy(
                official_module.harness_args(Path(args.official), TOKENIZER_PATH)
            )
        )
        official_policy.eval()
        heartbeat("after_policy_load_official")
        official_chunks = []
        peer_chunks = []
        recorded_chunks = [sample["recorded"] for sample in observations]
        for index, sample in enumerate(observations):
            official_chunks.append(
                predict_for_observation(
                    official_module,
                    harness,
                    probe,
                    official_policy,
                    official_pre,
                    official_post,
                    official_env_pre,
                    official_env_post,
                    sample,
                    heartbeat_label=f"official_obs_{index + 1}",
                )
            )
            heartbeat(f"official_obs_{index + 1}")
        persist_official_half(official_chunks, recorded_chunks)
        del official_policy, official_pre, official_post, official_env_pre, official_env_post
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        heartbeat("before_policy_load_peer")
        peer_policy, peer_pre, peer_post, peer_env_pre, peer_env_post = harness.load_policy(
            official_module.harness_args(Path(args.peer), TOKENIZER_PATH)
        )
        peer_policy.eval()
        heartbeat("after_policy_load_peer")
        for index, sample in enumerate(observations):
            peer_chunks.append(
                predict_for_observation(
                    official_module,
                    harness,
                    probe,
                    peer_policy,
                    peer_pre,
                    peer_post,
                    peer_env_pre,
                    peer_env_post,
                    sample,
                    heartbeat_label=f"peer_obs_{index + 1}",
                )
            )
            heartbeat(f"peer_obs_{index + 1}")
        scale_verdict, _, _ = scale_probe(official_chunks[0], peer_chunks[0])
        trajectory_verdict = trajectory_probe(
            official_chunks, peer_chunks, recorded_chunks
        )
        print(f"SCALE={scale_verdict}")
        print(f"TRAJECTORY={trajectory_verdict}")
        return 0
    finally:
        stop.set()
        thread.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--official", default=str(OFFICIAL_CHECKPOINT))
    parser.add_argument("--peer", default=str(PEER_CHECKPOINT))
    parser.add_argument("--obs-count", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.obs_count <= 0:
        raise RuntimeError("--obs-count must be positive")
    if args.dry_run:
        return synthetic_dry_run(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
