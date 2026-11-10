"""Frozen high-resolution probes under controlled corruption and noise placement."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/medical-ai-mpl')
import argparse
import csv
import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from high_resolution_probe import low_information, evaluate, bootstrap_metrics
from external_eval import risk_at, risk_coverage, COVERAGES

VIEWS = ['native_224', 'downsample28_upsample224']
METRICS = ['auroc', 'accuracy', 'nll', 'brier', 'ece', 'coverage', 'retained_error', 'risk_at_80']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantize(x):
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def transform(image, index, condition, view):
    if view not in VIEWS:
        raise ValueError(view)
    low = view == VIEWS[1]
    if condition == 'clean':
        return low_information(image) if low else image.copy()
    x = image.astype(np.float64) / 255
    if condition in ['noise_input', 'noise_source']:
        noise = np.random.default_rng(np.random.SeedSequence([42, int(index)])).normal(0, .1, image.shape)
        if condition == 'noise_input':
            base = low_information(image) if low else image
            return quantize(base.astype(np.float64) / 255 + noise)
        source = quantize(x + noise)
    elif condition == 'relative_blur':
        source = quantize(gaussian_filter(x, 8, mode='reflect'))
    elif condition == 'contrast':
        source = quantize(.5 + .5 * (x - .5))
    else:
        raise ValueError(condition)
    return low_information(source) if low else source


def probe_logits(features, probes, view):
    # Match StandardScaler.transform's float32 arithmetic used in the clean run.
    x = np.asarray(features, dtype=np.float32).copy()
    x -= probes[view + '_scaler_mean']
    x /= probes[view + '_scaler_scale']
    return x @ probes[view + '_coefficient'].ravel() + probes[view + '_intercept'][0]


def scalar(value):
    return float(value) if np.isfinite(value) else None


def interval(point, values):
    finite = np.asarray(values)[np.isfinite(values)]
    return dict(point=scalar(point), ci=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                defined_replicates=len(finite), undefined_replicates=len(values) - len(finite))


def metric_vector(y, z, cutoff):
    return np.r_[bootstrap_metrics(y, z, cutoff), risk_at(y, expit(z), .8)]


def main(output):
    import torch
    from torchvision.models import resnet18
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    cfg_path = ROOT / 'configs/high_resolution_robustness.json'
    cfg = json.loads(cfg_path.read_text())
    baseline_path = ROOT / 'reports/high_resolution/summary.json'
    baseline = json.loads(baseline_path.read_text())
    probe_path = ROOT / 'reports/high_resolution/predictions_and_probes.npz'
    probes = np.load(probe_path)
    archive = ROOT / 'data/pneumoniamnist_224.npz'
    checkpoint = ROOT / 'data/torch/hub/checkpoints/resnet18-f37072fd.pth'
    assert sha(archive) == baseline['provenance']['archive_sha256']
    assert sha(checkpoint) == baseline['provenance']['encoder_sha256']
    manifest = ROOT / 'configs/resolution_manifest.json'
    assert sha(manifest) == baseline['provenance']['manifest_sha256']
    indices = np.array(json.loads(manifest.read_text())['indices']['test'])
    np.testing.assert_array_equal(indices, probes['test_indices'])
    data = np.load(archive)
    images = data['test_images'][indices]
    y = data['test_labels'][indices].ravel()
    np.testing.assert_array_equal(y, probes['test_labels'])
    torch.set_num_threads(2)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    provenance = dict(archive_sha256=sha(archive), checkpoint_sha256=sha(checkpoint),
                      baseline_summary_sha256=sha(baseline_path), probes_sha256=sha(probe_path),
                      manifest_sha256=sha(manifest), config_sha256=sha(cfg_path), script_sha256=sha(__file__))
    cache = ROOT / 'data/high_resolution_robustness'
    cache.mkdir(exist_ok=True)
    metadata = cache / 'provenance.json'
    if metadata.exists():
        assert json.loads(metadata.read_text()) == provenance, 'Stale cache'
    else:
        metadata.write_text(json.dumps(provenance, indent=2) + '\n')
    logits = {}
    for view in VIEWS:
        stored = np.load(ROOT / f'data/high_resolution_features/{view}_test.npz')
        np.testing.assert_allclose(probe_logits(stored['features'], probes, view),
                                   probes[view + '_test_logits'], atol=1e-10)
        logits[view, 'clean'] = probes[view + '_test_logits']
    for condition in cfg['conditions'][1:]:
        for view in VIEWS:
            path = cache / f'{view}_{condition}.npz'
            if path.exists():
                saved = np.load(path)
                np.testing.assert_array_equal(saved['indices'], indices)
                z = saved['logits']
            else:
                chunks = []
                for start in range(0, len(images), 32):
                    batch = np.stack([transform(image, index, condition, view)
                                      for image, index in zip(images[start:start+32], indices[start:start+32])])
                    x = torch.tensor(batch[:, None], dtype=torch.float32).repeat(1, 3, 1, 1) / 255
                    with torch.inference_mode():
                        features = model((x - mean) / std).numpy()
                    chunks.append(probe_logits(features, probes, view))
                z = np.concatenate(chunks)
                np.savez_compressed(path, logits=z, indices=indices)
            logits[view, condition] = z
            print(f'Finished {view}/{condition}: {len(z)} cases', flush=True)
    rows = []
    for condition in cfg['conditions']:
        for view in VIEWS:
            cutoff = baseline['results'][view]['threshold']
            rows.append(dict(view=view, condition=condition, threshold=cutoff,
                             **evaluate(y, logits[view, condition], cutoff)))
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    point = {(v, c): metric_vector(y, z, baseline['results'][v]['threshold'])
             for (v, c), z in logits.items()}
    boots = {key: [] for key in logits}
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    for _ in range(cfg['bootstrap_replicates']):
        ix = rng.integers(0, len(y), len(y))
        for (view, condition), z in logits.items():
            boots[view, condition].append(metric_vector(y[ix], z[ix], baseline['results'][view]['threshold']))
    boots = {key: np.array(values) for key, values in boots.items()}
    summary = dict(config=cfg, provenance=provenance, results={}, high_minus_low={})
    for condition in cfg['conditions']:
        for view in VIEWS:
            key = view, condition
            summary['results'][f'{view}|{condition}'] = {
                metric: interval(point[key][i], boots[key][:, i]) for i, metric in enumerate(METRICS)}
        high, low = (VIEWS[0], condition), (VIEWS[1], condition)
        summary['high_minus_low'][condition] = {
            metric: interval(point[high][i]-point[low][i], boots[high][:, i]-boots[low][:, i])
            for i, metric in enumerate(METRICS)}
    def contrast(values):
        return ((values[VIEWS[0], 'noise_input'][..., -1] - values[VIEWS[0], 'clean'][..., -1])
                - (values[VIEWS[1], 'noise_input'][..., -1] - values[VIEWS[1], 'clean'][..., -1]))
    summary['primary'] = interval(contrast(point), contrast(boots))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / 'predictions.npz', labels=y, test_indices=indices,
                        **{v+'__'+c: z for (v, c), z in logits.items()})
    np.savez_compressed(out / 'bootstrap.npz', primary=contrast(boots),
                        **{v+'__'+c: b for (v, c), b in boots.items()})
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharex=True, sharey=True)
    for ax, condition in zip(axes, cfg['conditions']):
        for view, color in zip(VIEWS, ['#345995', '#bd5735']):
            ax.plot(COVERAGES, risk_coverage(y, expit(logits[view, condition])), color=color, label=view)
        ax.set(title=condition, xlabel='Coverage', ylim=(0, 1))
        ax.axvline(.8, color='gray', linestyle=':')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Error among retained cases')
    axes[0].legend(fontsize=6)
    fig.suptitle('Frozen ResNet-18 probes: same cases, fixed heads, explicit noise placement')
    fig.tight_layout()
    fig.savefig(out / 'risk_coverage.png', dpi=180)
    fig.savefig(out / 'risk_coverage.pdf')
    plt.close(fig)
    print('Primary', summary['primary'], flush=True)
    for row in rows:
        print({k: row[k] for k in ['view', 'condition', 'accuracy', 'coverage', 'retained_error', 'risk_at_80']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'reports/high_resolution/robustness')
    main(parser.parse_args().output)
