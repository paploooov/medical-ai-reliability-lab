"""AP/PA stratification of the frozen-model external RSNA test.

Uses stored predictions in reports/external/predictions.npz and ViewPosition
from configs/external_rsna_sample.csv. No new inference and no new thresholds.

Run: python experiments/external_stratify.py
Writes reports/external/stratified_view.csv, stratified_view.json,
and stratified_view.png/.pdf.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/medical-ai-mpl')
from pathlib import Path
import csv, json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/external'
import sys
sys.path.insert(0, str(ROOT / 'experiments'))
from external_eval import summarize, fast, BOOT_KEYS, BOOT, BOOT_SEED, ARMS, SEEDS, CONDITIONS

METRIC_KEYS = ['accuracy', 'auroc', 'sensitivity', 'specificity', 'coverage', 'retained_error']
# fast() returns accuracy, auroc, ece, coverage, retained_error, sensitivity, specificity
FAST_INDEX = {k: i for i, k in enumerate(BOOT_KEYS)}
# Reorder to the reported metric list.
REPORT_INDEX = [FAST_INDEX[k] for k in METRIC_KEYS]
VIEWS = ('AP', 'PA')
ARMS_ORDER = ('control_28', 'augmented_28', 'control_64')


def load_primary():
    """Primary-arm labels, views and class names, aligned with stored predictions."""
    rows = [r for r in csv.DictReader((ROOT / 'configs/external_rsna_sample.csv').open()) if r['arm'] == 'primary']
    y = np.array([int(r['label']) for r in rows], dtype=int)
    view = np.array([r['view'] for r in rows])
    cls = np.array([r['rsna_class'] for r in rows])
    pred = np.load(OUT / 'predictions.npz')
    stored = pred['labels__external']
    if len(y) != len(stored) or not np.array_equal(y, stored):
        raise ValueError('primary sample labels do not match labels__external in predictions.npz')
    unexpected = set(view) - set(VIEWS)
    if unexpected:
        raise ValueError(f'unexpected ViewPosition values: {sorted(unexpected)}')
    cuts = {}
    for r in csv.DictReader((OUT / 'results.csv').open()):
        if r['dataset'] == 'external' and r['condition'] == 'clean':
            cuts[r['arm'], int(r['seed'])] = float(r['threshold'])
    if len(cuts) != 9:
        raise ValueError('expected 9 arm/seed thresholds in results.csv')
    return dict(y=y, view=view, cls=cls, pred=pred, cuts=cuts, rows=rows)


def class_counts(mask, y, cls):
    n = int(mask.sum())
    n_pos = int((y[mask] == 1).sum())
    n_neg = n - n_pos
    names = sorted({str(c) for c in cls[mask]})
    by_class = {c: int((cls[mask] == c).sum()) for c in names}
    return dict(n=n, n_positive=n_pos, n_negative=n_neg, prevalence=n_pos / n if n else float('nan'), class_counts=by_class)


def point_rows(data, condition='clean'):
    """Per-seed point estimates, same metrics as external_eval.summarize."""
    y, view, cls, pred, cuts = data['y'], data['view'], data['cls'], data['pred'], data['cuts']
    rows = []
    for arm in ARMS_ORDER:
        for seed in SEEDS:
            p = pred[f'external__{arm}__s{seed}__{condition}']
            cut = cuts[arm, seed]
            for v in VIEWS:
                mask = view == v
                counts = class_counts(mask, y, cls)
                s = summarize(y[mask], p[mask], cut)
                rows.append(dict(
                    dataset=f'external_{v}', view=v, arm=arm, seed=seed, condition=condition,
                    n=counts['n'], n_positive=counts['n_positive'], n_negative=counts['n_negative'],
                    prevalence=counts['prevalence'],
                    lung_opacity=counts['class_counts'].get('Lung Opacity', 0),
                    normal=counts['class_counts'].get('Normal', 0),
                    accuracy=s['accuracy'], auroc=s['auroc'],
                    sensitivity=s['sensitivity'], specificity=s['specificity'],
                    coverage=s['coverage'], retained_error=s['retained_error'],
                    threshold=cut,
                ))
    return rows


def bootstrap_view(data, condition='clean', replicates=BOOT, seed=BOOT_SEED):
    """Within-view image bootstrap, mean over seeds, percentile 95% CIs.

    AP and PA are resampled independently (disjoint cases) in each replicate,
    so AP−PA differences have well-defined intervals. Conditional on fixed
    models and cutoffs, as in external_eval.
    """
    y, view, pred, cuts = data['y'], data['view'], data['pred'], data['cuts']
    rng = np.random.default_rng(seed)
    out = {}
    for v in VIEWS:
        idx = np.flatnonzero(view == v)
        n = len(idx)
        draws = [idx[rng.integers(0, n, n)] for _ in range(replicates)]
        entry = {'n': n}
        boots = {arm: np.empty((replicates, len(METRIC_KEYS))) for arm in ARMS_ORDER}
        points = {}
        for arm in ARMS_ORDER:
            pts = np.mean([
                np.take(fast(y[idx], pred[f'external__{arm}__s{s}__{condition}'][idx], cuts[arm, s]), REPORT_INDEX)
                for s in SEEDS
            ], 0)
            points[arm] = pts
            for b, ix in enumerate(draws):
                boots[arm][b] = np.mean([
                    np.take(fast(y[ix], pred[f'external__{arm}__s{s}__{condition}'][ix], cuts[arm, s]), REPORT_INDEX)
                    for s in SEEDS
                ], 0)
            lo, hi = np.nanquantile(boots[arm], [.025, .975], axis=0)
            entry[arm] = {
                k: dict(point=float(pts[i]), ci=[float(lo[i]), float(hi[i])])
                for i, k in enumerate(METRIC_KEYS)
            }
        # AP minus PA is filled later; store boots for difference CIs.
        out[v] = dict(entry=entry, boots=boots, points=points)
    summary = {}
    for v in VIEWS:
        block = out[v]['entry']
        # attach class distribution (fixed, not bootstrapped)
        mask = view == v
        block['distribution'] = class_counts(mask, y, data['cls'])
        summary[v] = block
    # Differences AP minus PA, independent within-view resamples already drawn.
    summary['AP_minus_PA'] = {}
    for arm in ARMS_ORDER:
        diff = out['AP']['boots'][arm] - out['PA']['boots'][arm]
        lo, hi = np.nanquantile(diff, [.025, .975], axis=0)
        dp = out['AP']['points'][arm] - out['PA']['points'][arm]
        summary['AP_minus_PA'][arm] = {
            k: dict(point=float(dp[i]), ci=[float(lo[i]), float(hi[i])])
            for i, k in enumerate(METRIC_KEYS)
        }
    return summary


def write_figure(summary, path_png, path_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    colors = {'control_28': '#345995', 'augmented_28': '#bd5735', 'control_64': '#6a994e'}
    labels = {'control_28': 'Control 28px', 'augmented_28': 'Noise-augmented 28px', 'control_64': 'Control 64px'}
    metrics = [
        ('accuracy', 'Accuracy', (0.5, 1.0)),
        ('auroc', 'AUROC (within view)', (0.5, 1.0)),
        ('retained_error', 'Error among accepted', (0.0, 0.45)),
        ('coverage', 'Acceptance rate', (0.4, 1.0)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.2))
    x = np.arange(len(ARMS_ORDER))
    width = 0.36
    for ax, (key, title, ylim) in zip(axes, metrics):
        for j, v in enumerate(VIEWS):
            pts = np.array([summary[v][arm][key]['point'] for arm in ARMS_ORDER])
            lo = np.array([summary[v][arm][key]['ci'][0] for arm in ARMS_ORDER])
            hi = np.array([summary[v][arm][key]['ci'][1] for arm in ARMS_ORDER])
            xpos = x + (j - 0.5) * width
            color = '#3d405b' if v == 'AP' else '#e07a5f'
            ax.bar(xpos, pts, width=width, color=color, alpha=.85, label=v if ax is axes[0] else None)
            ax.errorbar(xpos, pts, yerr=[pts - lo, hi - pts], fmt='none', ecolor='black', capsize=3, elinewidth=1)
        ax.set(title=title, ylim=ylim, xticks=x, xticklabels=[labels[a] for a in ARMS_ORDER])
        ax.tick_params(axis='x', labelrotation=20)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(title='ViewPosition', fontsize=8)
    n_ap, n_pa = summary['AP']['n'], summary['PA']['n']
    fig.suptitle(
        f'External RSNA test, clean images, stratified by DICOM ViewPosition '
        f'(AP n={n_ap}, PA n={n_pa}); mean of three seeds, 95% bootstrap intervals'
    )
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def main():
    data = load_primary()
    rows = []
    for condition in CONDITIONS:
        rows.extend(point_rows(data, condition))
    # Bootstrap CIs for the pre-registered clean condition (exploratory stratification).
    clean = bootstrap_view(data, 'clean')
    fields = list(rows[0].keys())
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'stratified_view.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    payload = {
        'protocol': 'reports/external/PROTOCOL.md',
        'note': (
            'Exploratory stratification by DICOM ViewPosition on the frozen-model '
            'external test. Not a pre-registered comparison. AP/PA differences '
            'motivate investigation of possible dependence on view type; they do '
            'not prove a view-type shortcut. Within-view AUROC, sensitivity and '
            'specificity are less tied to class prevalence than accuracy.'
        ),
        'bootstrap': dict(
            replicates=BOOT, seed=BOOT_SEED,
            unit='image within view (one per RSNA patientId)',
            scope=(
                'conditional on fixed models and cutoffs; percentile 95% intervals; '
                'mean over seeds 17/42/83; AP and PA resampled independently'
            ),
        ),
        'condition': 'clean',
        'results': {k: ({kk: vv for kk, vv in v.items()} if k != 'AP_minus_PA' else v)
                    for k, v in clean.items()},
        'provenance': dict(
            sample_csv_sha256=hashlib.sha256((ROOT / 'configs/external_rsna_sample.csv').read_bytes()).hexdigest(),
            predictions_sha256=hashlib.sha256((OUT / 'predictions.npz').read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    }
    (OUT / 'stratified_view.json').write_text(json.dumps(payload, indent=2) + '\n')
    write_figure(clean, OUT / 'stratified_view.png', OUT / 'stratified_view.pdf')
    for v in VIEWS:
        dist = clean[v]['distribution']
        print(v, 'n', dist['n'], 'prev', round(dist['prevalence'], 3), 'classes', dist['class_counts'])
        for arm in ARMS_ORDER:
            r = clean[v][arm]
            print(' ', arm, {k: (round(r[k]['point'], 4), [round(x, 4) for x in r[k]['ci']]) for k in METRIC_KEYS})
    print('wrote', OUT / 'stratified_view.csv')


if __name__ == '__main__':
    main()
