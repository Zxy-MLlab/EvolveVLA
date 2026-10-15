#!/usr/bin/env python3
"""Export every valid paired evaluation as a labelled sparse-frame MP4."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/root/autodl-tmp/EvolveVLA')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def read(path):
    return json.loads(path.read_text())


def assets(source, task, seed, label):
    paths = list(source.glob(f'round_*_{label}/{task}_{seed}/result.json'))
    if len(paths) != 1:
        raise ValueError('Expected one recorded episode')
    directory = paths[0].parent
    row = read(paths[0])
    frames = sorted(directory.glob('frame_*.jpg'))
    steps = [int(p.stem.split('_')[1]) for p in frames]
    if not frames or steps != list(range(0, row['steps'], 50)):
        raise ValueError('Missing recorded inference-boundary frames')
    return directory, row, frames, steps


def compose(task, seed, data, timeline):
    canvas = Image.new('RGB', (1280, 656), '#101923')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 17)
    title = ImageFont.truetype(FONT, 28)
    draw.text((20, 14), f'{task} | Clean | seed {seed}', font=title, fill='white')
    for index, (directory, row, frames, steps) in enumerate(data):
        x = index*640
        step = steps[max(0, bisect.bisect_right(steps, timeline)-1)]
        path = frames[steps.index(step)]
        with Image.open(path) as frame:
            canvas.paste(frame.convert('RGB').resize((640, 480)), (x, 110))
        label = 'BEFORE: original public model' if index == 0 else 'AFTER: fine-tuned coverage / step 2000'
        draw.text((x+14, 54), label, font=font, fill='white')
        status = 'SUCCESS' if row['success'] else 'FAILURE'
        color = '#70dfa5' if row['success'] else '#ffa69e'
        draw.text((x+14, 83), f'Recorded result: {status} | episode ends at step {row["steps"]}',
                  font=small, fill=color)
        draw.text((x+14, 595), f'Source image: step {step} | comparison step {timeline}',
                  font=small, fill='white')
    draw.text((14, 628), 'Recorded image every 50 actions; images are held between samples. Final outcome is from result.json.',
              font=small, fill='#b5c6d9')
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT/'outputs/gen1_recovery_confirmation_20261012')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'artifacts/gen1_eval_videos_20261012')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False)
    result = read(source/'comparison.json')
    videos = []
    for task, task_result in result['tasks'].items():
        for pair in task_result['pairs']:
            seed = pair['seed']
            data = [assets(source, task, seed, label) for label in ('original', 'candidate')]
            if data[0][1]['hashes'] != data[1][1]['hashes'] or data[0][1]['binding'] != data[1][1]['binding']:
                raise ValueError('Paired initial inputs differ')
            if any(item[1]['status'] != 'completed' or item[1]['success'] != pair[label]
                   for item, label in zip(data, ('original', 'candidate'))):
                raise ValueError('Outcome differs from paired comparison')
            path = output/f'{task}_{seed}_before_after.mp4'
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-n',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x656',
                '-framerate', '8', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
                '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            timeline = range(0, max(item[1]['steps'] for item in data), 50)
            try:
                for step in timeline:
                    frame = compose(task, seed, data, step).tobytes()
                    for _ in range(4):
                        process.stdin.write(frame)
                process.stdin.close()
                if process.wait():
                    raise RuntimeError('Video encoding failed')
            except BaseException:
                process.kill()
                process.wait()
                raise
            videos.append(dict(task=task, seed=seed, video=str(path),
                original_success=pair['original'], candidate_success=pair['candidate'],
                frames=4*len(timeline), fps=8,
                sources=[dict(directory=str(item[0]), steps=item[3],
                    outcome=item[1]['success'], terminal_step=item[1]['steps'],
                    images=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in item[2]]) for item in data]))
            print(path.name, 'before', pair['original'], 'after', pair['candidate'], flush=True)
    manifest = dict(source=str(source), candidate_checkpoint=result['checkpoint'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison_sha256=hashlib.sha256((source/'comparison.json').read_bytes()).hexdigest(),
        sampling='One real source image every 50 policy actions, held for 0.5 playback seconds; no interpolation.',
        limitation='Sparse-frame replay, not full-motion recording or real-time playback. Last sampled image can precede terminal success.',
        coverage='All twenty valid paired cases; includes gains, regressions and joint failures.', videos=videos)
    with (output/'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    html = ['<!doctype html><meta charset="utf-8"><title>EvolveVLA evaluation videos</title>',
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;background:#f4f6f8}video{width:100%}section{margin:25px 0;padding:20px;background:white}</style>',
        '<h1>Before / after fine-tuning: Clean paired evaluation</h1>',
        '<p>Left: original model. Right: coverage fine-tuned checkpoint 2000. All 20 valid pairs are included.</p>',
        '<p>Sparse replay: source images every 50 actions, held between samples. This is not full-motion recording. Final outcomes come from saved results.</p>']
    for row in videos:
        html.append(f'<section><h2>{row["task"]} / seed {row["seed"]}</h2><p>Before: {row["original_success"]}; after: {row["candidate_success"]}</p><video controls preload="none" src="{Path(row["video"]).name}"></video></section>')
    (output/'index.html').write_text('\n'.join(html))
    print('EXPORTED', len(videos), 'paired videos', flush=True)


if __name__ == '__main__':
    main()
