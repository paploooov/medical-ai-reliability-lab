"""Weighted noise augmentation of a linear head on a fixed 224px encoder."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/medical-ai-mpl')
import argparse
import csv
import hashlib
import json
import warnings
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from high_resolution_probe import evaluate
from high_resolution_robustness import transform, quantize, metric_vector, interval, METRICS
from temperature_scaling import nll
from external_eval import risk_coverage, COVERAGES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def noisy_image(image, index, seed):
    noise = np.random.default_rng(np.random.SeedSequence([int(seed), int(index)])).normal(0, .1, image.shape)
    return quantize(image.astype(np.float64) / 255 + noise)


def standardize(features, probes):
    x = np.array(features, dtype=np.float32, copy=True)
    x -= probes['native_224_scaler_mean']
    x /= probes['native_224_scaler_scale']
    return x


def weighted_training(clean, noisy, labels):
    if clean.shape != noisy.shape or len(clean) != len(labels):
        raise ValueError('Paired clean/noisy features and labels required')
    return np.concatenate([clean, noisy]), np.tile(labels, 2), np.full(2 * len(labels), .5)


def select_head(clean, noisy, labels, selection, selection_labels):
    x, y, weights = weighted_training(clean, noisy, labels)
    candidates = []
    for c in [.01, .1, 1., 10.]:
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=3000)
        model.fit(x, y, sample_weight=weights)
        loss = nll(selection_labels, model.decision_function(selection))
        candidates.append((loss, c, model))
    return min(candidates, key=lambda item: item[0]), candidates


def main(output):
    import torch
    from torchvision.models import resnet18
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    cfg_path = ROOT / 'configs/high_resolution_augmentation.json'
    cfg = json.loads(cfg_path.read_text())
    baseline_path = ROOT / 'reports/high_resolution/summary.json'
    baseline = json.loads(baseline_path.read_text())
    probe_path = ROOT / 'reports/high_resolution/predictions_and_probes.npz'
    probes = np.load(probe_path)
    archive = ROOT / 'data/pneumoniamnist_224.npz'
    checkpoint = ROOT / 'data/torch/hub/checkpoints/resnet18-f37072fd.pth'
    manifest_path = ROOT / 'configs/resolution_manifest.json'
    assert sha(archive) == baseline['provenance']['archive_sha256']
    assert sha(checkpoint) == baseline['provenance']['encoder_sha256']
    assert sha(manifest_path) == baseline['provenance']['manifest_sha256']
    indices = json.loads(manifest_path.read_text())['indices']
    source = np.load(archive)
    clean = {}
    labels = {}
    for split in ['train', 'selection', 'calibration', 'test']:
        path = ROOT / f'data/high_resolution_features/native_224_{split}.npz'
        assert sha(path) == baseline['provenance']['feature_sha256'][path.name]
        data = np.load(path)
        np.testing.assert_array_equal(data['indices'], indices[split])
        original = 'val' if split in ['selection', 'calibration'] else split
        np.testing.assert_array_equal(data['labels'], source[original + '_labels'][indices[split]].ravel())
        clean[split] = data['features']
        labels[split] = data['labels']
    provenance = dict(config_sha256=sha(cfg_path), script_sha256=sha(__file__),
                      archive_sha256=sha(archive), encoder_sha256=sha(checkpoint),
                      manifest_sha256=sha(manifest_path), baseline_sha256=sha(baseline_path),
                      original_probes_sha256=sha(probe_path),
                      helpers_sha256={name: sha(ROOT / 'experiments' / name) for name in
                                      ['high_resolution_probe.py', 'high_resolution_robustness.py',
                                       'external_eval.py', 'temperature_scaling.py']})
    cache = ROOT / 'data/high_resolution_augmentation'
    cache.mkdir(exist_ok=True)
    meta = cache / 'provenance.json'
    if meta.exists():
        assert json.loads(meta.read_text()) == provenance, 'Stale feature cache'
    else:
        meta.write_text(json.dumps(provenance, indent=2) + '\n')
    torch.set_num_threads(2)
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(checkpoint, weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]

    def extract(split, condition):
        path = cache / f'{split}_{condition}.npz'
        ix = indices[split]
        if path.exists():
            saved = np.load(path)
            np.testing.assert_array_equal(saved['indices'], ix)
            np.testing.assert_array_equal(saved['labels'], labels[split])
            return saved['features']
        images = source[split + '_images'][ix]
        chunks = []
        for start in range(0, len(images), 32):
            batch = []
            for image, index in zip(images[start:start + 32], ix[start:start + 32]):
                if condition.startswith('noise_'):
                    altered = noisy_image(image, index, int(condition.split('_')[1]))
                else:
                    altered = transform(image, index, condition, 'native_224')
                batch.append(altered)
            x = torch.tensor(np.stack(batch)[:, None], dtype=torch.float32).repeat(1, 3, 1, 1) / 255
            with torch.inference_mode():
                chunks.append(encoder((x - mean) / std).numpy())
            if start % 512 == 0:
                print(f'Features {split}/{condition}: {min(start+32,len(images))}/{len(images)}', flush=True)
        features = np.concatenate(chunks)
        np.savez_compressed(path, features=features, labels=labels[split], indices=ix)
        return features

    warnings.filterwarnings('error', category=ConvergenceWarning)
    noisy_train = extract('train', 'noise_20261116')
    chosen, candidates = select_head(standardize(clean['train'], probes), standardize(noisy_train, probes),
                                     labels['train'], standardize(clean['selection'], probes), labels['selection'])
    loss, c, augmented = chosen
    zcal = augmented.decision_function(standardize(clean['calibration'], probes))
    pcal = expit(zcal)
    cutoff = float(np.quantile(np.maximum(pcal, 1-pcal), .2, method='linear'))
    selection = dict(chosen_C=c, selection_nll=loss, augmented_cutoff=cutoff,
                     total_training_weight=len(labels['train']), actual_training_rows=2 * len(labels['train']),
                     candidates=[dict(C=v, selection_nll=l, iterations=int(m.n_iter_[0])) for l, v, m in candidates])
    (out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print('Selected head using clean selection only:', selection, flush=True)
    np.savez_compressed(out / 'augmented_head.npz', coefficient=augmented.coef_, intercept=augmented.intercept_,
                        scaler_mean=probes['native_224_scaler_mean'], scaler_scale=probes['native_224_scaler_scale'],
                        calibration_logits=zcal, calibration_labels=labels['calibration'])
    logits = {}
    thresholds = dict(control=baseline['results']['native_224']['threshold'], augmentation=cutoff)
    previous = np.load(ROOT / 'reports/high_resolution/robustness/predictions.npz')
    np.testing.assert_array_equal(labels['test'], previous['labels'])
    for condition in cfg['conditions']:
        features = clean['test'] if condition == 'clean' else extract('test', condition)
        x = standardize(features, probes)
        z = x @ probes['native_224_coefficient'].ravel() + probes['native_224_intercept'][0]
        old_condition = 'noise_input' if condition == 'noise_42' else condition
        if 'native_224__' + old_condition in previous.files:
            np.testing.assert_allclose(z, previous['native_224__' + old_condition], atol=1e-10)
        logits['control', condition] = z
        logits['augmentation', condition] = augmented.decision_function(x)
    y = labels['test']
    rows = [dict(arm=arm, condition=condition, threshold=thresholds[arm], **evaluate(y, z, thresholds[arm]))
            for (arm, condition), z in logits.items()]
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    point = {key: metric_vector(y, z, thresholds[key[0]]) for key, z in logits.items()}
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    boots = {key: [] for key in logits}
    for _ in range(cfg['bootstrap_replicates']):
        ix = rng.integers(0, len(y), len(y))
        for key, z in logits.items():
            boots[key].append(metric_vector(y[ix], z[ix], thresholds[key[0]]))
    boots = {key: np.array(v) for key, v in boots.items()}
    summary = dict(config=cfg, provenance=provenance, selection=selection, results={}, augmented_minus_control={})
    for condition in cfg['conditions']:
        for arm in thresholds:
            key = arm, condition
            summary['results'][f'{arm}|{condition}'] = {
                metric: interval(point[key][i], boots[key][:, i]) for i, metric in enumerate(METRICS)}
        a, b = ('augmentation', condition), ('control', condition)
        summary['augmented_minus_control'][condition] = {
            metric: interval(point[a][i] - point[b][i], boots[a][:, i] - boots[b][:, i])
            for i, metric in enumerate(METRICS)}
    summary['primary'] = summary['augmented_minus_control']['noise_42']['risk_at_80']
    summary['feature_sha256'] = {p.name: sha(p) for p in cache.glob('*.npz')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / 'predictions.npz', labels=y, test_indices=indices['test'],
                        **{a + '__' + c: z for (a, c), z in logits.items()})
    np.savez_compressed(out / 'bootstrap.npz', **{a + '__' + c: v for (a, c), v in boots.items()})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, cfg['conditions']):
        for arm, color in [('control', '#345995'), ('augmentation', '#bd5735')]:
            ax.plot(COVERAGES, risk_coverage(y, expit(logits[arm, condition])), color=color, label=arm)
        ax.set(title=condition, xlabel='Coverage', ylabel='Error among retained cases', ylim=(0, .45))
        ax.axvline(.8, color='gray', linestyle=':'); ax.grid(alpha=.2)
    axes[0, 0].legend(); fig.suptitle('Frozen 224px encoder: clean head versus noise-augmented head')
    fig.tight_layout(); fig.savefig(out / 'risk_coverage.png', dpi=180); fig.savefig(out / 'risk_coverage.pdf'); plt.close(fig)
    print('Primary noise risk difference:', summary['primary'], flush=True)
    for row in rows:
        print({key: row[key] for key in ['arm', 'condition', 'accuracy', 'coverage', 'retained_error', 'risk_at_80']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/high_resolution/augmentation')
    main(parser.parse_args().output)
