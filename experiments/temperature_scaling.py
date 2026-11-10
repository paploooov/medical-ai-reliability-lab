"""Exploratory temperature scaling fitted only on the clean calibration split."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/medical-ai-mpl')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from external_eval import ece, reliability, perturb

ROOT = Path(__file__).resolve().parents[1]


def nll(y, z):
    """Stable binary log loss from logits, including extreme logits."""
    return float(np.mean(np.logaddexp(0, z) - np.asarray(y) * z))


def fit_temperature(y, z, bounds=(0.05, 20.0)):
    y, z = np.asarray(y), np.asarray(z, dtype=float)
    if y.shape != z.shape or y.ndim != 1 or not len(y):
        raise ValueError('Expected matching nonempty one-dimensional arrays')
    if not np.isfinite(z).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite logits and binary labels required')
    low, high = bounds
    if not 0 < low < high:
        raise ValueError('Invalid temperature bounds')
    # NLL is convex in inverse temperature. Include endpoints explicitly.
    result = minimize_scalar(lambda inv: nll(y, z * inv),
                             bounds=(1 / high, 1 / low), method='bounded',
                             options={'xatol': 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    candidates = [low, high, 1 / result.x]
    return min(candidates, key=lambda t: nll(y, z / t))


def scores(y, z):
    p = expit(z)
    return np.array([nll(y, z), np.mean((p-y)**2), ece(y, p)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resolution_study import ResolutionCNN, logits, tensor

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg = json.loads((ROOT / 'configs/temperature_scaling.json').read_text())
    manifest = json.loads((ROOT / 'configs/resolution_manifest.json').read_text())
    archive = ROOT / 'data/pneumoniamnist.npz'
    assert sha(archive) == manifest['archives']['28']
    d = np.load(archive)
    ci, ti = manifest['indices']['calibration'], manifest['indices']['test']
    cy, y = d['val_labels'][ci].ravel(), d['test_labels'][ti].ravel()
    cx = tensor(d['val_images'][ci])
    images = d['test_images'][ti].astype(np.float32) / 255
    previous = np.load(ROOT / 'reports/augmentation/comparison_predictions.npz')
    np.testing.assert_array_equal(y, previous['labels'])
    models, arrays, rows = [], dict(calibration_labels=cy, test_labels=y,
                                    calibration_indices=ci, test_indices=ti), []
    for arm in cfg['arms']:
        folder = 'budget' if arm == 'control' else 'augmentation'
        run = json.loads((ROOT / f'reports/{folder}/run.json').read_text())
        for seed in cfg['seeds']:
            choice = next(c for c in run['choices'] if c['resolution'] == 28 and c['seed'] == seed)
            weights = ROOT / f'models/{folder}/r28_s{seed}.pt'
            assert sha(weights) == choice['weights_sha256']
            model = ResolutionCNN()
            model.load_state_dict(torch.load(weights, weights_only=True))
            model.eval()
            zc = logits(model, cx).astype(float)
            t = fit_temperature(cy, zc, cfg['temperature_bounds'])
            prefix = f'{arm}_s{seed}'
            arrays[prefix + '_calibration_logits'] = zc
            models.append(dict(arm=arm, seed=seed, temperature=t,
                               boundary_fit=t in cfg['temperature_bounds'],
                               calibration_nll_before=nll(cy, zc),
                               calibration_nll_after=nll(cy, zc/t),
                               original_cutoff=choice['threshold'],
                               transformed_cutoff=float(expit(logit(choice['threshold'])/t)),
                               weights_sha256=sha(weights)))
            for condition in cfg['conditions']:
                z = logits(model, torch.tensor(perturb(images, condition, 28)[:, None],
                                               dtype=torch.float32)).astype(float)
                np.testing.assert_allclose(expit(z), previous[f'{prefix}_{condition}'], atol=1e-6)
                arrays[f'{prefix}_{condition}_logits'] = z
                # Compare in logit space: same selected cases, no rounding ties.
                cut = logit(choice['threshold'])
                np.testing.assert_array_equal(np.abs(z) >= cut, np.abs(z/t) >= cut/t)
                for stage, zz in [('before', z), ('after', z/t)]:
                    value = scores(y, zz)
                    rows.append(dict(arm=arm, seed=seed, condition=condition, stage=stage,
                                     temperature=t, nll=value[0], brier=value[1], ece=value[2]))
    with (output / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / 'logits.npz', **arrays)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = rng.integers(0, len(y), (cfg['bootstrap_replicates'], len(y)))
    summary = dict(config=cfg, models=models, results={},
                   provenance=dict(archive_sha256=sha(archive),
                                   manifest_sha256=sha(ROOT/'configs/resolution_manifest.json'),
                                   script_sha256=sha(__file__),
                                   config_sha256=sha(ROOT/'configs/temperature_scaling.json'),
                                   commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()))
    for arm in cfg['arms']:
        for condition in cfg['conditions']:
            selected = [m for m in models if m['arm'] == arm]
            zs = [arrays[f'{arm}_s{m["seed"]}_{condition}_logits'] for m in selected]
            ts = [m['temperature'] for m in selected]
            before = np.mean([scores(y, z) for z in zs], axis=0)
            after = np.mean([scores(y, z/t) for z,t in zip(zs, ts)], axis=0)
            boot = np.array([np.mean([scores(y[ix], z[ix]/t)-scores(y[ix], z[ix])
                                     for z,t in zip(zs,ts)], axis=0) for ix in draws])
            lo, hi = np.quantile(boot, [.025,.975], axis=0)
            summary['results'][f'{arm}|{condition}'] = {
                name: dict(before=float(before[i]), after=float(after[i]),
                           difference=float(after[i]-before[i]), ci=[float(lo[i]), float(hi[i])])
                for i,name in enumerate(['nll','brier','ece'])}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True)
    for a,arm in enumerate(cfg['arms']):
        for c,condition in enumerate(cfg['conditions']):
            ax=axes[a,c]; ax.plot([.5,1],[.5,1],':',color='gray')
            for stage,style in [('before','o--'),('after','s-')]:
                probs=[]
                for m in models:
                    if m['arm'] != arm: continue
                    z=arrays[f'{arm}_s{m["seed"]}_{condition}_logits']
                    probs.append(expit(z if stage=='before' else z/m['temperature']))
                counts,acc,conf=reliability(np.tile(y,len(probs)),np.concatenate(probs))
                mask=counts>=10
                ax.plot(conf[mask],acc[mask],style,label=stage)
            ax.set(title=f'{arm}: {condition}',xlim=(.48,1.02),ylim=(0,1.02))
            if a==1: ax.set_xlabel('Top-label confidence')
            if c==0: ax.set_ylabel('Bin accuracy')
            ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Internal test: temperature fitted on clean calibration only\n15 equal-width bins; seeds pooled for display; bins with at least 10 model–case pairs')
    fig.tight_layout();fig.savefig(output/'reliability.png',dpi=180);fig.savefig(output/'reliability.pdf');plt.close(fig)
    print(json.dumps(summary['results'], indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'reports/temperature_scaling')
    main(parser.parse_args().output)
