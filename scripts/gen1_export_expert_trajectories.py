#!/usr/bin/env python3
"""Export the accepted Gen1 training/validation expert trajectories as videos."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

import cv2
import h5py
import imageio_ffmpeg
import numpy as np


ROOT = Path('/root/autodl-tmp/EvolveVLA')
SPLIT = ROOT/'data/gen1_pilot_two_task_20261010/split_manifest.json'
CAMERAS = (
    ('cam_head', 'Front'),
    ('cam_left_wrist', 'Left wrist'),
    ('cam_right_wrist', 'Right wrist'),
    ('cam_third_view', 'External'),
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def validate_manifest(manifest):
    if manifest['status'] != 'passed' or manifest['counts'] != {'train': 90, 'val': 10}:
        raise ValueError('Unaccepted trajectory split')
    rows = manifest['episodes']
    if len(rows) != 100 or len({(r['task'], r['seed']) for r in rows}) != 100:
        raise ValueError('Expected 100 distinct accepted trajectories')
    for task in ('place_mouse_pad', 'hanging_mug'):
        for split, count in (('train', 45), ('val', 5)):
            if sum(r['task'] == task and r['split'] == split for r in rows) != count:
                raise ValueError('Task/split trajectory count changed')
    if any(not (r['success'] and r['plan_success'] and r['replay_success']) for r in rows):
        raise ValueError('Unsuccessful expert trajectory in split')
    return sorted(rows, key=lambda r: (r['task'], r['split'] != 'train', r['seed']))


def encode(row, output):
    source = Path(row['path']).resolve()
    if not source.is_relative_to(ROOT/'data') or digest(source) != row['sha256']:
        raise ValueError(f'Source path/hash mismatch: {source}')
    name = f"{row['task']}_{row['split']}_seed_{row['seed']:06d}.mp4"
    video = output/name
    temporary = output/(name + '.part.mp4')
    if video.exists() or temporary.exists():
        raise FileExistsError(f'Existing export: {name}')
    with h5py.File(source) as record:
        fps = int(record['additional_info/frequency'][()])
        frames = int(row['frames'])
        arrays = [record[f'vision/{camera}/colors'] for camera, _ in CAMERAS]
        if fps != 15 or any(len(array) != frames for array in arrays):
            raise ValueError(f'Camera length or frame rate mismatch: {source}')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-n',
            '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '640x520',
            '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
            '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(temporary)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for index in range(frames):
                canvas = np.full((520, 640, 3), (31, 35, 39), dtype=np.uint8)
                title = f"{row['task']} | {row['split']} | seed {row['seed']}"
                cv2.putText(canvas, title, (9, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.60, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f'{index+1}/{frames}', (556, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.49, (210, 210, 210), 1, cv2.LINE_AA)
                for number, (camera, label) in enumerate(CAMERAS):
                    frame = cv2.imdecode(np.frombuffer(arrays[number][index], dtype=np.uint8),
                                         cv2.IMREAD_COLOR)
                    if frame is None or frame.shape != (240, 320, 3):
                        raise ValueError(f'Invalid {camera} frame {index}: {source}')
                    x, y = (number % 2)*320, 40+(number // 2)*240
                    canvas[y:y+240, x:x+320] = frame
                    cv2.rectangle(canvas, (x, y), (x+110, y+25), (24, 27, 30), -1)
                    cv2.putText(canvas, label, (x+6, y+17), cv2.FONT_HERSHEY_SIMPLEX,
                                0.48, (255, 255, 255), 1, cv2.LINE_AA)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'Encoder failed for {name}')
            os.replace(temporary, video)
        except BaseException:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            raise
    return {'task': row['task'], 'split': row['split'], 'seed': row['seed'],
            'frames': frames, 'fps': fps, 'duration_seconds': frames/fps,
            'source': str(source), 'source_sha256': row['sha256'],
            'video': video.name, 'video_sha256': digest(video)}


def write_index(output, videos):
    lines = [
        '<!doctype html><html lang="zh"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Gen1 expert trajectories</title>',
        '<style>body{font:15px system-ui,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;'
        'color:#20262b;background:#f7f8f8}h1{font-size:24px}h2{font-size:19px;margin-top:32px}'
        'video{width:100%;max-width:800px;background:#111}table{border-collapse:collapse;width:100%}'
        'td,th{padding:8px;border-bottom:1px solid #d7dfe0;text-align:left}'
        'a{color:#006b77}small{color:#546268}</style>',
        '<h1>Gen1 仿真专家轨迹</h1>',
        '<p>鼠标垫与挂杯任务，共 90 条训练、10 条验证轨迹。四视角逐帧录制，15 fps；'
        '这是专家演示，不是模型评估或真机录像。</p>',
    ]
    for task, label in (('place_mouse_pad', '鼠标垫'), ('hanging_mug', '挂杯')):
        lines.append(f'<h2>{label} / {task}</h2>')
        sample = next(r for r in videos if r['task'] == task and r['split'] == 'train')
        lines.append(f'<video controls preload="metadata" src="{html.escape(sample["video"])}"></video>')
        lines.append('<table><thead><tr><th>划分</th><th>Seed</th><th>长度</th><th>视频</th></tr></thead><tbody>')
        for row in (r for r in videos if r['task'] == task):
            name = html.escape(row['video'])
            lines.append(f'<tr><td>{row["split"]}</td><td>{row["seed"]}</td>'
                         f'<td>{row["duration_seconds"]:.1f} 秒</td>'
                         f'<td><a href="{name}">{name}</a></td></tr>')
        lines.append('</tbody></table>')
    lines.append('</html>')
    (output/'index.html').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'artifacts/gen1_expert_trajectory_videos_20261012')
    parser.add_argument('--max-per-task', type=int, default=0,
                        help='Optional sample limit per task; zero exports the entire split')
    args = parser.parse_args()
    if args.max_per_task < 0:
        raise ValueError('Sample limit must be nonnegative')
    manifest = json.loads(SPLIT.read_text())
    rows = validate_manifest(manifest)
    if args.max_per_task:
        rows = [row for task in ('place_mouse_pad', 'hanging_mug')
                for row in [r for r in rows if r['task'] == task][:args.max_per_task]]
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    videos = []
    for row in rows:
        video = encode(row, output)
        videos.append(video)
        print('EXPORTED', video['video'], f"{video['frames']} frames", flush=True)
    report = {'status': 'completed', 'source_split': str(SPLIT),
              'source_split_sha256': digest(SPLIT),
              'export_script_sha256': digest(Path(__file__)),
              'all_split_trajectories': not args.max_per_task,
              'layout': 'Top left: front; top right: left wrist; bottom left: right wrist; bottom right: external.',
              'videos': videos}
    with (output/'manifest.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    write_index(output, videos)
    print('COMPLETE', len(videos), output, flush=True)


if __name__ == '__main__':
    main()
