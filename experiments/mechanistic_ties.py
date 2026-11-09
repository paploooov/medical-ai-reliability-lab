"""Versuch 09: eingefrorene Modelle, deterministische Annahmeregeln, keine Optimierung."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/medical-ai-mpl')
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import torch
from scipy.special import expit
from noise_diagnostics import inspect
from resolution_study import ResolutionCNN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mechanistic_ties'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys(split, indices):
    """Labelunabhängige, persistente Ordnung; keine Optimierung des Hash-Salts."""
    return np.array([hashlib.sha256(f'study09-v1:{split}:{int(i)}'.encode()).hexdigest()
                     for i in indices])


def fit_rule(confidence, identifiers):
    """floor(4*n/5) höchste Werte; SHA256 löst ausschließlich exakte Gleichstände."""
    n = len(confidence)
    if n < 2 or len(set(identifiers)) != n:
        raise ValueError('Mindestens zwei eindeutige Beobachtungsschlüssel erforderlich')
    k = 4 * n // 5
    order = np.lexsort((identifiers, -confidence))
    j = order[k - 1]
    return float(confidence[j]), str(identifiers[j])


def apply_rule(confidence, identifiers, rule):
    cutoff, boundary = rule
    return (confidence > cutoff) | ((confidence == cutoff) & (identifiers <= boundary))


def counts(keep, wrong):
    n = int(keep.sum())
    return dict(accepted=n, errors=int((keep & wrong).sum()), coverage=n / len(keep),
                risk=float(wrong[keep].mean()) if n else None)


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main(out):
    # Refuse to replace an existing result, including this study's own output.
    if (out / 'diagnostics.csv').exists():
        raise FileExistsError('Vorhandene Ergebnisse bleiben erhalten; anderes --output wählen')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    d = np.load(ROOT / 'data/pneumoniamnist.npz')
    saved = dict(np.load(ROOT / 'reports/calibration/predictions.npz'))
    bootstrap = dict(np.load(ROOT / 'reports/calibration/bootstrap.npz'))
    meta = json.loads((ROOT / 'reports/calibration/provenance.json').read_text())['models']
    ci, ti = saved['calibration_indices'], saved['test_indices']
    ck, tk = keys('val', ci), keys('test', ti)
    clean = d['test_images'][ti].astype(np.float32) / 255
    noisy = np.clip(clean + np.random.default_rng(42).normal(0, .1, clean.shape), 0, 1)
    inputs = [d['val_images'][ci].astype(np.float32) / 255, clean, noisy]
    diagnostics, outcomes, empty_rows, arrays, hashes = [], [], [], {}, {}
    # Re-use all 2,000 paired calibration/test draws from Study 08.
    # A bootstrap copy has its own occurrence key, but the same original case hash.
    for i, m in enumerate(meta):
        arm, seed = m['arm'], m['seed']
        name = f'{arm}_s{seed}'
        folder = 'budget' if arm == 'control' else 'augmentation'
        path = ROOT / f'models/{folder}/r28_s{seed}.pt'
        assert sha(path) == m['weights_sha256']
        hashes[str(path.relative_to(ROOT))] = sha(path)
        model = ResolutionCNN()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        bias = float(model.layers[-1].bias.detach()[0])
        probs, zeros = [], []
        for j, (split, images) in enumerate(zip(['calibration', 'clean', 'noise'], inputs)):
            z, zero, _ = inspect(model, torch.tensor(images[:, None], dtype=torch.float32))
            p = expit(z)
            reference = saved['calibration_probabilities'][i] if j == 0 else saved['test_probabilities'][i, j - 1]
            np.testing.assert_array_equal(p, reference)
            np.testing.assert_array_equal(z[zero], np.full(zero.sum(), bias, dtype=z.dtype))
            probs.append(p)
            zeros.append(zero)
            arrays[f'{name}_{split}_logits'] = z
            arrays[f'{name}_{split}_zero'] = zero
        cc = np.maximum(probs[0], 1 - probs[0])
        cutoff = float(np.quantile(cc, .2, method='linear'))
        tie = cc == cutoff
        diagnostics.append(dict(arm=arm, seed=seed, bias=bias,
            bias_probability=float(expit(np.float32(bias))), threshold=cutoff,
            calibration_ties=int(tie.sum()), ties_zero=int((tie & zeros[0]).sum()),
            ties_nonzero=int((tie & ~zeros[0]).sum()), calibration_zero=int(zeros[0].sum()),
            calibration_saturated=int((cc == 1).sum()),
            ties_saturated=int((tie & (cc == 1)).sum()),
            clean_saturated=int((np.maximum(probs[1], 1-probs[1]) == 1).sum()),
            noise_saturated=int((np.maximum(probs[2], 1-probs[2]) == 1).sum())))
        rule = fit_rule(cc, ck)
        assert apply_rule(cc, ck, rule).sum() == 4 * len(cc) // 5
        for split, p, zero, identifiers, y in zip(['calibration', 'clean', 'noise'], probs, zeros,
                [ck, tk, tk], [d['val_labels'][ci].ravel(), saved['test_labels'], saved['test_labels']]):
            conf = np.maximum(p, 1 - p)
            base = conf >= cutoff
            ranked = apply_rule(conf, identifiers, rule)
            for policy, keep in [('original', base), ('tie_break', ranked),
                    ('reject_zero', base & ~zero), ('tie_break_reject_zero', ranked & ~zero)]:
                outcomes.append(dict(arm=arm, seed=seed, split=split, policy=policy,
                    zero_accepted=int((keep & zero).sum()), **counts(keep, (p >= .5) != y)))
                arrays[f'{name}_{split}_{policy}'] = keep
        tc = np.maximum(np.array(probs[1:]), 1 - np.array(probs[1:]))
        tz = np.array(zeros[1:])
        accepted = np.zeros((len(bootstrap['test_indices']), 4, 2), dtype=np.int32)
        for b, (bc, bt) in enumerate(zip(bootstrap['calibration_indices'], bootstrap['test_indices'])):
            threshold = float(np.quantile(cc[bc], .2, method='linear'))
            # Occurrence suffix disambiguates repeated calibration cases. Test suffix
            # is fixed at zero; repeated test observations retain one decision.
            occurrence = np.array([f'{ck[v]}:{j:04d}' for j, v in enumerate(bc)])
            brule = fit_rule(cc[bc], occurrence)
            masks = [tc[:, bt] >= threshold,
                     apply_rule(tc[:, bt], np.char.add(tk[bt], ':0000'), brule)]
            masks += [masks[0] & ~tz[:, bt], masks[1] & ~tz[:, bt]]
            accepted[b] = np.array([mask.sum(1) for mask in masks])
        np.testing.assert_array_equal(accepted[:, 0], bootstrap['accepted_counts'][:, i])
        arrays[f'{name}_bootstrap_accepted'] = accepted
        if arm == 'control' and seed == 83:
            empty = accepted[:, 0, 1] == 0
            arrays['control83_empty_draws'] = np.flatnonzero(empty)
            assert empty.sum() == 54
            # In every empty draw the cutoff exceeds even the maximum noisy confidence.
            assert (bootstrap['thresholds'][empty, i] > tc[1].max()).all()
        for policy_idx, policy in enumerate(['original', 'tie_break', 'reject_zero', 'tie_break_reject_zero']):
            for j, condition in enumerate(['clean', 'noise']):
                empty_rows.append(dict(arm=arm, seed=seed, condition=condition, policy=policy,
                    empty_replicates=int((accepted[:, policy_idx, j] == 0).sum()), replicates=len(accepted)))
    write_csv(out / 'diagnostics.csv', diagnostics)
    write_csv(out / 'outcomes.csv', outcomes)
    write_csv(out / 'bootstrap_empty.csv', empty_rows)
    np.savez_compressed(out / 'mechanism.npz', **arrays)
    source_paths = ['reports/calibration/predictions.npz', 'reports/calibration/bootstrap.npz',
                    'reports/calibration/provenance.json', 'configs/resolution_manifest.json',
                    'data/pneumoniamnist.npz', 'experiments/mechanistic_ties.py',
                    'experiments/noise_diagnostics.py', 'experiments/resolution_study.py']
    hashes.update({p: sha(ROOT / p) for p in source_paths})
    (out / 'provenance.json').write_text(json.dumps(dict(inputs_sha256=hashes,
        rounding='floor(4*n/5)', hash_salt='study09-v1',
        conditions=['clean', 'noise_sd0.1_seed42'], models=meta,
        numpy=np.__version__, torch=torch.__version__, threads=2,
        deterministic_algorithms=True), indent=2) + '\n')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    main(parser.parse_args().output)
