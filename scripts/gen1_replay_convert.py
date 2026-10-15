"""Convert verified raw replay directly from ZIP, then audit every output frame.

Keeps archives immutable and avoids a second raw-data copy. Uses the existing
converter's schema and frame-preserving streaming encoder. No GPU imports.
"""
import argparse
import dataclasses
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import zipfile

from gen1_official_replay_audit import ROOT, digest, require
from gen1_replay_prepare import arrays, signature, CAMERAS

CONVERTER = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/scripts/transform_lerobot_v30_format.py'


def raw_episode(row):
    import h5py
    import numpy as np
    import cv2
    with zipfile.ZipFile(row['archive']) as archive:
        content = archive.read(row['member'])
    import hashlib
    require(hashlib.sha256(content).hexdigest() == row['sha256'], 'Raw bytes changed')
    with h5py.File(io.BytesIO(content)) as source:
        state, action = arrays(source)
        require(signature(state,action) == row['trajectory_sha256'], 'Raw action identity changed')
        images = {}
        for camera, source_camera in CAMERAS.items():
            frames = [cv2.imdecode(np.frombuffer(value,np.uint8),cv2.IMREAD_COLOR)
                      for value in source[f'vision/{source_camera}/colors'][:]]
            require(all(x is not None and x.shape == (240,320,3) for x in frames), 'RGB decode failed')
            images[camera] = np.stack(frames)
        require(all(len(x) == row['frames'] for x in images.values()), 'Raw RGB frame mismatch')
    return state,action,images


def audit(target, manifest, manifest_path):
    import numpy as np
    import pandas as pd
    from gen1_multitask_audit import SequentialVideo
    rows = manifest['episodes']
    info = json.loads((target/'meta/info.json').read_text())
    require((info['fps'],info['total_episodes'],info['total_frames']) ==
            (15,len(rows),manifest['frames']), 'Converted metadata mismatch')
    data = pd.read_parquet(target/'data').sort_values('index')
    require(np.array_equal(data['index'],np.arange(manifest['frames'])), 'Global indices mismatch')
    episodes = pd.read_parquet(target/'meta/episodes').set_index('episode_index').sort_index()
    require(list(episodes.index) == list(range(len(rows))), 'Episode index mismatch')
    tasks = pd.read_parquet(target/'meta/tasks.parquet')
    prompts = {int(r.task_index):str(p) for p,r in tasks.iterrows()}
    readers = {camera:SequentialVideo() for camera in CAMERAS}
    maximum = 0.
    for row in rows:
        state,action,images = raw_episode(row)
        frame = data.iloc[row['start']:row['stop']]
        meta = episodes.loc[row['episode']]
        n = row['frames']
        require((int(meta.dataset_from_index),int(meta.dataset_to_index),int(meta.length)) ==
                (row['start'],row['stop'],n), 'Episode boundary mismatch')
        require((frame.episode_index == row['episode']).all(), 'Episode identity mismatch')
        require(np.array_equal(frame.frame_index,np.arange(n)), 'Frame identity mismatch')
        require(np.allclose(frame.timestamp,np.arange(n)/15,atol=1e-5), 'Frame timestamp mismatch')
        np.testing.assert_array_equal(np.stack(frame['observation.state']),state)
        np.testing.assert_array_equal(np.stack(frame.action),action)
        require(frame.task_index.nunique() == 1 and
                prompts[int(frame.iloc[0].task_index)] in row['prompts'], 'Prompt binding mismatch')
        for camera in CAMERAS:
            key = 'observation.images.'+camera
            start = float(meta[f'videos/{key}/from_timestamp'])
            stop = float(meta[f'videos/{key}/to_timestamp'])
            require(abs(stop-start-n/15)<1e-5,'Shortened episode video')
            path = target/info['video_path'].format(video_key=key,
                chunk_index=int(meta[f'videos/{key}/chunk_index']),
                file_index=int(meta[f'videos/{key}/file_index']))
            for begin in range(0,n,32):
                count = min(32,n-begin)
                actual = readers[camera].read(path,start+begin/15,count).astype(np.float32)
                expected = images[camera][begin:begin+count].astype(np.float32)
                error = np.abs(actual-expected).mean(axis=(1,2,3))
                swapped = np.abs(actual-expected[...,::-1]).mean(axis=(1,2,3))
                require(bool((error<12).all()),f'Video/source alignment error: {row["episode"]}:{camera}')
                require(bool((error<=swapped+.5).all()),'Suspected channel swap')
                maximum = max(maximum,float(error.max()))
        if (row['episode']+1)%10 == 0:
            print('REPLAY_RGB_AUDIT',row['episode']+1,'/',len(rows),'max_mae',maximum,flush=True)
    for reader in readers.values():
        reader.finish()
        require(sum(r['frames'] for r in reader.report)==manifest['frames'],'Final video frame mismatch')
    report = dict(status='passed',all_frames_decoded=True,all_timestamps_checked=True,
        all_state_action_cells_exact=True,prompt_binding_checked=True,episodes=len(rows),
        frames=manifest['frames'],fps=15,maximum_video_mae=maximum,
        manifest=str(manifest_path),manifest_sha256=digest(manifest_path),
        converter_sha256=digest(CONVERTER),script_sha256=digest(Path(__file__)),
        norm_stats=manifest['norm_stats'],videos={c:r.report for c,r in readers.items()})
    with (target/'conversion_audit.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print('REPLAY_CONVERSION_AUDIT_PASSED',len(rows),flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--audit-only',action='store_true')
    args = parser.parse_args()
    require(args.repo_id.isidentifier(),'Invalid repository name')
    target = ROOT/'data'/args.repo_id
    manifest = json.loads(args.manifest.read_text())
    require(manifest['status']=='raw_manifest_passed','Raw preflight missing')
    require(digest(Path(manifest['norm_stats']['path'])) == manifest['norm_stats']['sha256'],
            'Initializer normalization changed')
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',HF_LEROBOT_HOME=str(ROOT/'data'),WANDB_MODE='disabled',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if args.audit_only:
        require((target/'encoding_completed.json').is_file(),'Encoding incomplete')
        audit(target,manifest,args.manifest)
        return
    require(not target.exists(),'Refusing to overwrite dataset')
    # Archives remain compressed. Allow up to 8 GiB output while preserving
    # 24 GiB: remaining 6.25 GiB checkpoint + eval outputs + emergency reserve.
    require(shutil.disk_usage(ROOT).free >= 32*2**30,'Need 8 GiB conversion budget plus 24 GiB reserve')
    spec = importlib.util.spec_from_file_location('replay_pinned_converter',CONVERTER)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=converter
    spec.loader.exec_module(converter)
    def reject_delete(*args,**kwargs):
        raise RuntimeError('Refusing converter deletion')
    converter.shutil=type('NoDelete',(),{'rmtree':staticmethod(reject_delete)})
    cfg = dataclasses.replace(converter.DEFAULT_DATASET_CONFIG,image_writer_processes=0,
                              image_writer_threads=2,video_backend='pyav')
    ds=converter.create_empty_dataset(args.repo_id,'unified_robot',
        converter._build_motor_names_from_dims([7,7]),15,image_height=240,image_width=320,dataset_config=cfg)
    if not hasattr(ds,'_streaming_encoder'):
        ds._streaming_encoder=ds.writer._streaming_encoder
    converter.configure_video_encoding(ds,cfg)
    from gen1_convert_ready import install_lossless_encoder
    encoding=install_lossless_encoder(ds._streaming_encoder)
    rng=random.Random(0)
    try:
        for row in manifest['episodes']:
            require(shutil.disk_usage(ROOT).free >= 24*2**30,'Conversion reached reserve')
            state,action,images=raw_episode(row)
            prompt=rng.choice(row['prompts'])
            for i in range(row['frames']):
                frame={'observation.state':state[i],'action':action[i],'task':prompt}
                frame.update({f'observation.images.{c}':img[i] for c,img in images.items()})
                ds.add_frame(frame)
            ds.save_episode()
            require(len(encoding)==row['episode']+1,'Encoder audit missing')
            require(all(v['frames']==row['frames'] for v in encoding[-1].values()),'Encoder dropped frames')
            if (row['episode']+1)%10==0:
                print('REPLAY_ENCODED',row['episode']+1,'/',len(manifest['episodes']),flush=True)
    finally:
        ds.finalize()
        converter.finalize_dataset(ds)
    with (target/'encoding_completed.json').open('x') as stream:
        json.dump(dict(episodes=len(encoding),manifest_sha256=digest(args.manifest),
                       encoders=encoding),stream)
    audit(target,manifest,args.manifest)


if __name__=='__main__':
    main()
