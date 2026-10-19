"""Plot labels / predictions against real time and against cycle position.

Outputs (in OUT_DIR):
  timeline_<task>_<video>.png  x = time (s); top: label vs prediction, bottom: cycle_r sawtooth
  cycle_vs_<task>.png          x = cycle_r, y = class; label vs prediction for all segments
  segments_<task>.csv          the table behind both plots

Edit the CONFIG block, then run from anywhere:  python plot_cycle_timeline.py
"""
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
ROOT = Path('/workspace/pyskl')
CSV_PATH = ROOT / 'data/swimming/metadata/cycles_phases.csv'
ANN_PATH = ROOT / 'data/swimming/datasets/full_video_poses/swimming_4class_phase.pkl'
RESULTS_PATH = ROOT / 'work_dirs/stgcn/stgcn_swimming_twohead/j/result.pkl'  # None -> labels only
OUT_DIR = ROOT / 'data/swimming/outputs/cycle_timeline'

TASK = 'phase'        # 'phase' or 'error'
SPLIT = 'xsub_test'   # split the results pkl was produced on
SWIMMERS = [7]        # None -> all swimmers
EXT = 'png'           # 'png' or 'pdf' for the paper
# Used only if the dataset pkl has no 'phase_num' key: pulls video / cycle / phase out of frame_dir.
FRAME_DIR_RE = r'(?P<video>s\d+_s\d+_[lr]side).*?(?P<cycle_num>\d+).*?(?P<phase>catch|pull|push|recovery)'
# ====================================================================

PHASES = ['catch', 'pull', 'push', 'recovery']
ERRORS = ['correct', 'dropped_elbow', 'straight_arm_pull', 'shortened_pull']
CLASSES = PHASES if TASK == 'phase' else ERRORS
PRIOR_MU = [0.0898, 0.2056, 0.3115, 0.6952]  # timing-prior centres, catch..recovery
LABEL_C, PRED_C, INK, MUTED = '#2a78d6', '#eb6834', '#2b2b2b', '#8a8a85'


def style(ax, classes=None):
    """Shared axis look; optional categorical y-axis with the first class on top."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis='x', color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)
    if classes:
        ax.set_yticks(range(len(classes)), classes)
        ax.set_ylim(len(classes) - 0.4, -0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)


# --- 1. Labels and cycle position from the CSV ----------------------------------
# Missing start/end are filled from the neighbouring segment in the same video
# (end <- next start, start <- previous end), for plotting only.
# cycle_r = segment midpoint's position inside its stroke cycle, where the cycle spans
# the first start to the last end of that (video, cycle_num). It is computed BEFORE
# dropping unusable rows so a cycle's bounds don't shrink when one of its phases is unusable.
df = pd.read_csv(CSV_PATH).sort_values('phase_num', ignore_index=True)
by_video = df.groupby('video')
df['start'] = df.start.fillna(by_video.end.shift(1))
df['end'] = df.end.fillna(by_video.start.shift(-1))
df = df.dropna(subset=['start', 'end'])

by_cycle = df.groupby(['video', 'cycle_num'])
cycle_start, cycle_end = by_cycle.start.transform('min'), by_cycle.end.transform('max')
df['cycle_r'] = ((df.start + df.end) / 2 - cycle_start) / (cycle_end - cycle_start)

# Class index per segment. Error label = whichever error flag is set (at most one), else 'correct'.
flags = df[ERRORS[1:]].fillna(0).to_numpy()
df['label'] = df.phase.map(PHASES.index) if TASK == 'phase' else np.where(flags.any(1), flags.argmax(1) + 1, 0)

df = df[df.usable == 1]
if SWIMMERS:
    df = df[df.swimmer.isin(SWIMMERS)]

# --- 2. Model predictions (optional) ----------------------------------------------
# PYSKL filters annotations by split while keeping their file order, and dist_test.sh writes
# results in that same order, so the i-th result belongs to the i-th split annotation.
# Two-head results are assumed to be (phase_scores, error_scores) per sample.
if RESULTS_PATH:
    data = pickle.load(open(ANN_PATH, 'rb'))
    results = pickle.load(open(RESULTS_PATH, 'rb'))
    in_split = set(data['split'][SPLIT])
    anns = [a for a in data['annotations'] if a['frame_dir'] in in_split]
    assert len(anns) == len(results), f'{len(anns)} annotations in {SPLIT} vs {len(results)} results'

    head_idx = 0 if TASK == 'phase' else 1
    preds = [int(np.argmax(r[head_idx] if isinstance(r, (tuple, list)) else r)) for r in results]

    # Join predictions to CSV rows: by phase_num if the pkl kept it, else by parsing frame_dir.
    if 'phase_num' in anns[0]:
        keys, pred_df = ['phase_num'], pd.DataFrame({'phase_num': [a['phase_num'] for a in anns]})
    else:
        keys = ['video', 'cycle_num', 'phase']
        matches = [re.search(FRAME_DIR_RE, a['frame_dir']) for a in anns]
        bad = next((a['frame_dir'] for a, m in zip(anns, matches) if m is None), None)
        assert bad is None, f'FRAME_DIR_RE does not match {bad!r}; annotation keys: {sorted(anns[0])}'
        pred_df = pd.DataFrame([m.groupdict() for m in matches]).astype({'cycle_num': int})
    pred_df['pred'] = preds
    df = df.merge(pred_df, on=keys, how='inner')
    print(f'{len(preds)} predictions, {len(df)} matched to CSV rows')

assert len(df), 'No rows left to plot - check SWIMMERS / SPLIT / FRAME_DIR_RE.'
has_pred = 'pred' in df
OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / f'segments_{TASK}.csv', index=False)

# --- 3. Timeline per video: x = real time ------------------------------------------
# Two stacked panels sharing the time axis instead of a dual y-axis, since class names
# and cycle_r have different scales. Label/prediction bars are offset so both stay visible.
off = 0.13 if has_pred else 0
for video, d in df.groupby('video', sort=False):
    d = d.sort_values('start')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 5.5),
                                   gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.12))
    ax1.hlines(d.label - off, d.start, d.end, color=LABEL_C, lw=5, label='Label')
    title = video
    if has_pred:
        wrong = d[d.label != d.pred]
        ax1.hlines(d.pred + off, d.start, d.end, color=PRED_C, lw=5, label='Prediction')
        ax1.scatter((wrong.start + wrong.end) / 2, wrong.pred + off, marker='x', s=40, color=INK,
                    zorder=3, label=f'Mismatch ({len(wrong)}/{len(d)})')
        title += f'  -  accuracy {1 - len(wrong) / len(d):.1%}'
    ax1.set_title(title, loc='left', fontsize=11, color=INK)
    style(ax1, CLASSES)

    # cycle_r: one rising line per stroke cycle (the sawtooth); faint bars show each segment's span.
    for _, c in d.groupby('cycle_num'):
        ax2.plot((c.start + c.end) / 2, c.cycle_r, color=LABEL_C, lw=2, marker='o', ms=6,
                 markeredgecolor='white', markeredgewidth=1.5)
    ax2.hlines(d.cycle_r, d.start, d.end, color=LABEL_C, lw=1, alpha=0.35)
    for mu, name in zip(PRIOR_MU, PHASES):
        ax2.axhline(mu, color=MUTED, lw=0.8, ls=':')
        ax2.text(1.005, mu, f'μ {name}', transform=ax2.get_yaxis_transform(),
                 va='center', fontsize=8, color=MUTED)
    ax2.set(ylim=(0, 1), ylabel='cycle_r', xlabel='Time (s)')
    style(ax2)
    fig.savefig(OUT_DIR / f'timeline_{TASK}_{video}.{EXT}', bbox_inches='tight', dpi=200)
    plt.close(fig)

# --- 4. Cycle position vs class: all segments in one plot ----------------------------
# Small vertical jitter separates overlapping points; a grey line joins each wrong
# prediction to its label so errors show where in the cycle they happen.
off = 0.15 if has_pred else 0
jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(df))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(df.cycle_r, df.label - off + jitter, s=28, color=LABEL_C,
           edgecolor='white', lw=0.8, label='Label', zorder=3)
if has_pred:
    wrong = df.label != df.pred
    ax.vlines(df.cycle_r[wrong], df.label[wrong] - off, df.pred[wrong] + off, color=MUTED, lw=0.6, alpha=0.6)
    ax.scatter(df.cycle_r, df.pred + off + jitter, s=32, marker='^', color=PRED_C, edgecolor='white',
               lw=0.8, label=f'Prediction ({wrong.sum()} wrong / {len(df)})', zorder=3)
for mu in PRIOR_MU:
    ax.axvline(mu, color=MUTED, lw=0.8, ls=':')
ax.set(xlim=(0, 1), xlabel='Cycle position cycle_r')
ax.set_title(f'{TASK}: label{" vs prediction" if has_pred else ""} by cycle position',
             loc='left', fontsize=11, color=INK)
style(ax, CLASSES)
fig.savefig(OUT_DIR / f'cycle_vs_{TASK}.{EXT}', bbox_inches='tight', dpi=200)
plt.close(fig)

print(f'{len(df)} segments, {df.video.nunique()} videos -> {OUT_DIR}')