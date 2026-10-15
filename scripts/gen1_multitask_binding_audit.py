#!/usr/bin/env python3
"""Audit scene-aware official seen-prompt binding for every coverage episode."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT / 'scripts'))
from gen1_multitask_binding import bind

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--description-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    official_path = a.description_root.resolve() / 'utils/generate_episode_instructions.py'
    spec = importlib.util.spec_from_file_location('official_binding_audit', official_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    records = []
    for row in m['episodes']:
        info_path = Path(row['episode_directory']) / 'scene_info.json'
        info = json.loads(info_path.read_text())
        result = bind(row['task'], int(row['seed']), info, a.description_root)
        pool = json.loads(Path(result['source']).read_text())['seen']
        with patch.object(official.random, 'shuffle', lambda values: None):
            eligible = official.filter_instructions(pool.copy(), info['info'])
        if eligible != [pool[i] for i in result['eligible_indices']]:
            raise ValueError('Official template eligibility differs')
        choices = []
        for key, value in info['info'].items():
            obj = result['objects'].get(key[1:-1])
            if obj is not None:
                choices.append(json.loads(Path(obj['source']).read_text())['seen'][obj['index']])
        with patch.object(official.random, 'choice', side_effect=choices):
            expected = official.replace_placeholders(result['template'], info['info'])
        if expected != result['prompt']:
            raise ValueError('Official replacement differs')
        result.update(split=row['split'], trajectory_source=row['path'], trajectory_sha256=row['sha256'],
                      scene_info=str(info_path), scene_info_sha256=hashlib.sha256(info_path.read_bytes()).hexdigest())
        records.append(result)
    report = dict(status='passed', episodes=len(records), tasks=sorted({r['task'] for r in records}),
                  split_counts={s: sum(r['split'] == s for r in records) for s in ('train','val')},
                  records=records, manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  official_eligibility_parity=True, official_replacement_parity=True,
                  code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__), Path(__file__).with_name('gen1_multitask_binding.py'), official_path)},
                  rule='Official field eligibility; deterministic SHA256 selection; source fields only; no simulator RNG')
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('status','episodes','tasks','split_counts')}))

if __name__ == '__main__':
    main()
