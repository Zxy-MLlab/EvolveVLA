#!/usr/bin/env python3
"""Export loss curves from recorded recovery runs; no GPU or log mutation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/root/autodl-tmp/EvolveVLA')
RUNS = [
    ('gen1_recovery_action_expert_v3_20261012', 'Action expert: native loss', '#3675b5'),
    ('gen1_recovery_augmented_20261012', 'Full model + prompt augmentation: weighted physical loss', '#cb7634'),
    ('gen1_recovery_coverage_v2_20261012', 'Coverage continuation: weighted physical loss', '#278b70'),
]
WINDOW = 100


def main():
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    summaries = []
    for ax, (name, label, color) in zip(axes, RUNS):
        directory = ROOT/'outputs'/name
        rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()
                if line.strip()]
        steps = np.asarray([row['step'] for row in rows])
        losses = np.asarray([row['loss'] for row in rows], dtype=float)
        if not np.isfinite(losses).all() or np.any(losses <= 0):
            raise ValueError('Log scale requires finite positive losses')
        if not np.array_equal(steps, np.arange(1, len(rows)+1)):
            raise ValueError('Missing or duplicate recorded steps')
        smooth = np.convolve(losses, np.ones(WINDOW)/WINDOW, mode='valid')
        ax.plot(steps, losses, color=color, alpha=0.22, linewidth=0.55,
                label='Recorded per-update loss', rasterized=True)
        ax.plot(steps[WINDOW-1:], smooth, color=color, linewidth=2.2,
                label='100-update trailing mean')
        ax.set_yscale('log')
        ax.set_ylabel('Training loss (log scale)')
        ax.set_xlabel('Update within this run')
        ax.set_title(label, loc='left', fontsize=12, pad=10)
        ax.grid(True, which='major', alpha=0.20)
        first = float(losses[:WINDOW].mean())
        last = float(losses[-WINDOW:].mean())
        completed = (directory/'completed.json').is_file()
        if completed:
            if json.loads((directory/'completed.json').read_text())['status'] != 'completed':
                raise ValueError('Invalid completion record')
        text = f'First 100 mean: {first:.5f}    Last 100 mean: {last:.5f}'
        ax.text(0.02, 0.05, text, transform=ax.transAxes, fontsize=10,
                bbox={'facecolor': 'white', 'alpha': 0.88, 'edgecolor': 'none'})
        if name == 'gen1_recovery_coverage_v2_20261012':
            ax.axvline(2000, color='#495563', linestyle=':', linewidth=1.3)
            ax.axvline(steps[-1], color='#a23b3b', linestyle='--', linewidth=1.3)
            ax.annotate('Saved checkpoint: 2000', xy=(2000, 0.92),
                        xycoords=('data', 'axes fraction'), ha='center', fontsize=9,
                        bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
            ax.annotate(f'Interrupted at {steps[-1]} / 4000', xy=(steps[-1], 0.92),
                        xycoords=('data', 'axes fraction'), ha='right', fontsize=9,
                        color='#a23b3b', bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
        summaries.append(dict(run=name, source=str(directory/'metrics.jsonl'),
            recorded_updates=len(rows), completed=completed,
            first100_mean=first, last100_mean=last,
            change_percent=(last/first-1)*100, minimum_loss=float(losses.min()),
            maximum_loss=float(losses.max()), moving_average_window=WINDOW))
    axes[0].legend(loc='upper right', fontsize=9, frameon=False)
    fig.suptitle('EvolveVLA: recorded training loss', fontsize=17, x=0.08, ha='left', y=0.985)
    fig.text(0.08, 0.025,
        'Loss definitions differ across panels; compare trends within each run.\n'
        'Continuation updates start at 1. Training loss is not rollout success rate.',
        fontsize=10, color='#495563')
    fig.tight_layout(rect=[0.02, 0.065, 0.99, 0.955], h_pad=2.0)
    stem = ROOT/'artifacts/gen1_loss_curves_20261012'
    for extension in ('png', 'pdf', 'svg'):
        path = stem.with_suffix('.'+extension)
        fig.savefig(path, dpi=180, bbox_inches='tight')
        print(path)
    report = dict(runs=summaries,
        note='Training curves only. Native and weighted physical loss have different definitions.')
    with stem.with_suffix('.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    plt.close(fig)


if __name__ == '__main__':
    main()
