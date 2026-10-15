"""Process-local decoded-frame cache for small, audited LeRobot datasets.

Preserves native RGB pixels and timestamp tolerance. Uses no recompression,
resizing, augmentation or channel swaps. Sampled exact parity with the existing
PyAV loader is required before installation. Existing dataset files are read-only.
"""
from pathlib import Path


def install(roots, max_bytes=20*2**30):
    import av
    import numpy as np
    import torch
    import lerobot.datasets.lerobot_dataset as dataset
    original=dataset.decode_video_frames
    cache={};used=0;checks=[]
    for root in roots:
        for path in sorted((Path(root)/'videos').rglob('*.mp4')):
            path=path.resolve()
            with av.open(str(path)) as video:
                stream=video.streams.video[0]
                count=stream.frames
                if not count:raise ValueError('Missing frame count')
                shape=(count,stream.height,stream.width,3)
                size=int(np.prod(shape))
                if used+size>max_bytes:raise ValueError('Decoded cache exceeds byte budget')
                frames=np.empty(shape,dtype=np.uint8)
                times=np.empty(count,dtype=np.float32)
                index=-1
                for index,frame in enumerate(video.decode(stream)):
                    if index>=count:raise ValueError('More frames than metadata')
                    frames[index]=frame.to_ndarray(format='rgb24')
                    times[index]=float(frame.pts*frame.time_base)
                if index+1!=count or not np.all(np.diff(times)>0):
                    raise ValueError('Invalid decoded frame count/timing')
            cache[str(path)]=(frames,times)
            used+=size
            for index in sorted(set([0,1,count//4,count//2,3*count//4,count-2,count-1])):
                ts=float(times[index])
                expected=original(path,[ts],.001,'pyav')
                actual=torch.from_numpy(frames[index].copy()).permute(2,0,1).unsqueeze(0).float()/255
                if not torch.equal(expected,actual):raise ValueError(f'Pixel parity failed: {path}:{index}')
            checks.append(dict(path=str(path),frames=count,bytes=size,pixel_parity=True))
            print('VIDEO_CACHE',path.name,str(path.parent.parent),count,used,flush=True)

    def decode(video_path,timestamps,tolerance_s,backend=None,**kwargs):
        key=str(Path(video_path).resolve())
        if key not in cache:return original(video_path,timestamps,tolerance_s,backend,**kwargs)
        frames,times=cache[key]
        indices=[]
        for timestamp in timestamps:
            distances=np.abs(times-np.float32(timestamp))
            index=int(np.argmin(distances))
            if distances[index]>=tolerance_s:raise ValueError('Cached timestamp violates tolerance')
            indices.append(index)
        return torch.from_numpy(frames[indices].copy()).permute(0,3,1,2).float()/255
    dataset.decode_video_frames=decode
    return dict(bytes=used,videos=checks,scope='process-local; sampled exact PyAV pixel parity')
