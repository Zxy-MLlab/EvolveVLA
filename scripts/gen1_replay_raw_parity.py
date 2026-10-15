"""Bind raw Clean trajectories to the official export and probe RGB alignment."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import av
import cv2
import h5py
import numpy as np

from gen1_official_replay_audit import ROOT, digest, require


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'New output required')
    audit = json.loads((ROOT / 'outputs/gen1_official_replay_structural_audit_20261013.json').read_text())
    lookup = {r['trajectory_sha256']: r for r in audit['records']}
    require(len(lookup) == 2500, 'Duplicate official trajectories')
    local = json.loads((ROOT / 'data/gen1_multitask_coverage_20261012/split_manifest.json').read_text())
    local_seeds = {(r['task'], r['seed']) for r in local['episodes']}
    fields = ('left_arm_joint_states', 'left_ee_joint_states',
              'right_arm_joint_states', 'right_ee_joint_states')
    bindings, probes = [], []
    for task in ('place_mouse_pad', 'hanging_mug'):
        root = ROOT / 'data/gen1_official_raw_probe_20261013'
        source = json.loads((root / f'{task}.json').read_text())
        archive = root / f'{task}.zip'
        require(digest(archive) == source['sha256'], 'Raw archive digest changed')
        with zipfile.ZipFile(archive) as z:
            seeds = [int(x) for x in z.read(f'{task}/aloha_agilex/seed.txt').decode().split()]
            names = sorted(n for n in z.namelist() if n.endswith('.hdf5'))
            require(len(seeds) == len(names) == 50, 'Unexpected raw episode count')
            sampled = set()
            for name, seed in zip(names, seeds, strict=True):
                content = z.read(name)
                with h5py.File(io.BytesIO(content)) as f:
                    h = hashlib.sha256()
                    for group in ('state', 'action'):
                        a = np.concatenate([f[f'{group}/{k}'][:] for k in fields], axis=1)
                        h.update(np.asarray(a, dtype='<f4').tobytes())
                    row = lookup.get(h.hexdigest())
                    require(row is not None, 'Raw episode absent from export')
                    prompts = json.loads(f['instructions'][()])
                    require(row['prompt'] in prompts, 'Export prompt absent from source')
                    require((task, seed) not in local_seeds, 'Official/local collection seed overlap')
                    binding = dict(task=task, source=name, raw_sha256=hashlib.sha256(content).hexdigest(),
                                   seed=seed, export_episode=row['episode'], frames=row['frames'],
                                   state_action_exact=True, prompt_in_source=True,
                                   structural_candidate=row['structural_candidate'])
                    bindings.append(binding)
                    # One complete and one shortened episode per task. This is
                    # deliberately labeled a probe, never a full RGB audit.
                    category = row['structural_candidate']
                    if category in sampled:
                        continue
                    sampled.add(category)
                    for camera, raw_camera in (('cam_high', 'cam_head'),
                                               ('cam_left_wrist', 'cam_left_wrist'),
                                               ('cam_right_wrist', 'cam_right_wrist')):
                        meta = row['videos']['observation.images.' + camera]
                        raw = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                               for b in f[f'vision/{raw_camera}/colors'][:]]
                        require(all(x is not None and x.shape == (240,320,3) for x in raw),
                                'Raw RGB decode failed')
                        decoded = []
                        with av.open(meta['path']) as video:
                            stream = video.streams.video[0]
                            stream.thread_count = 2
                            stream.thread_type = 'SLICE'
                            video.seek(max(0, int((meta['start']/15)/stream.time_base)-1),
                                       stream=stream, backward=True)
                            for frame in video.decode(stream):
                                index = round(float(frame.pts*frame.time_base)*15)
                                if index < meta['start']:
                                    continue
                                if index >= meta['stop']:
                                    break
                                require(index == meta['start'] + len(decoded), 'Video decode gap')
                                decoded.append(frame.to_ndarray(format='rgb24'))
                        require(len(decoded) == meta['frames'], 'Decoded count differs from container')
                        # Use full-resolution mean error for native alignment,
                        # sampled spatial grid for nearby-frame diagnostic.
                        raw_array = np.stack(raw).astype(np.float32)
                        actual = np.stack(decoded).astype(np.float32)
                        mae = np.abs(actual-raw_array[:len(actual)]).mean(axis=(1,2,3))
                        offsets, margins = [], []
                        for i, rgb in enumerate(actual):
                            candidates = range(max(0,i-3), min(len(raw),i+4))
                            errors = [(float(np.abs(rgb[::4,::4]-raw_array[j,::4,::4]).mean()), j)
                                      for j in candidates]
                            best, j = min(errors)
                            offsets.append(j-i)
                            native = next(v for v,jj in errors if jj == i)
                            margins.append(native-best)
                        from collections import Counter
                        probes.append(dict(task=task, episode=row['episode'], camera=camera,
                            raw_frames=len(raw), export_frames=len(actual),
                            same_index_mae_mean=float(mae.mean()), same_index_mae_max=float(mae.max()),
                            nearby_best_offset_counts=dict(Counter(offsets)),
                            shifted_frames_with_mae_gain_over_1=sum(m>1 for m in margins),
                            caveat='Nearby matching is diagnostic; visually static frames can tie'))
                        print('PROBE', json.dumps(probes[-1]), flush=True)
    report = dict(status='raw_identity_verified_rgb_probed', ready_for_training=False,
                  raw_episode_bindings=bindings, rgb_probes=probes,
                  source_script_sha256=digest(Path(__file__)),
                  conclusion='Raw sources provide exact task, seed, action and prompt identity. '
                  'Use audited frame-preserving conversion before mixing; probes are not a full RGB audit.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print('RAW_BINDINGS', len(bindings), 'RGB_PROBES',len(probes), flush=True)


if __name__ == '__main__':
    main()
