"""Unactivated candidate: bounded weights for future closing AND opening.

Training targets are unchanged. This module is not imported by the active
trainer. Event presence is not evidence of grasp/release success.
"""
import numpy as np
from gen1_training_event_audit import event_windows


def dual_event_weights(actions, rows, horizon=50):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
        raise ValueError('Expected finite physical 14-D targets')
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Expected positive integer horizon')
    weights = np.ones(len(actions), dtype=np.float32)
    offset = 0
    for row in rows:
        n = row['frames']
        if (row['split'] != 'train' or n < 1 or row['start'] != offset
                or row['stop'] != offset + n or row['stop'] > len(actions)):
            raise ValueError('Invalid contiguous training episode')
        flags = event_windows(actions[offset:offset+n, [6, 13]], horizon=horizon)
        # Count the union once, so overlapping closing/opening windows do not
        # receive double weight. Only real within-episode transitions count.
        event = (flags['closing'] | flags['opening']).any(axis=1)
        late = np.arange(n) >= np.ceil(0.8 * max(n - 1, 1))
        weights[offset:offset+n] = 1.0 + event + 0.5 * late
        offset += n
    if not rows or offset != len(actions):
        raise ValueError('Episode coverage differs from action array')
    return weights
