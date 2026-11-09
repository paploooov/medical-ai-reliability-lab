"""Separate skalare Entscheidungsprüfung plus vollständiger deterministischer Zweitlauf."""
from pathlib import Path
import argparse
import csv
import json
import tempfile
import numpy as np
from mechanistic_ties import ROOT, OUT, main, sha, keys


def verify(out):
    provenance = json.loads((out / 'provenance.json').read_text())
    for path, digest in provenance['inputs_sha256'].items():
        assert sha(ROOT / path) == digest, path
    previous = json.loads((out / 'previous_artifacts.json').read_text())
    for path, digest in previous.items():
        assert sha(ROOT / path) == digest, path
    a = dict(np.load(out / 'mechanism.npz'))
    p = dict(np.load(ROOT / 'reports/calibration/predictions.npz'))
    rows = list(csv.DictReader((out / 'outcomes.csv').open()))
    for i, model in enumerate(provenance['models']):
        name = f"{model['arm']}_s{model['seed']}"
        cc = np.maximum(p['calibration_probabilities'][i], 1-p['calibration_probabilities'][i])
        ck = keys('val', p['calibration_indices'])
        ordered = sorted(zip(cc.tolist(), ck.tolist()), key=lambda v: (-v[0], v[1]))
        cutoff, boundary = ordered[4*len(cc)//5 - 1]
        original_cutoff = np.quantile(cc, .2, method='linear')
        for j, split in enumerate(['calibration', 'clean', 'noise']):
            prob = p['calibration_probabilities'][i] if j == 0 else p['test_probabilities'][i, j-1]
            ids = ck if j == 0 else keys('test', p['test_indices'])
            zero = a[f'{name}_{split}_zero']
            masks = {policy: [] for policy in ['original', 'tie_break', 'reject_zero', 'tie_break_reject_zero']}
            for k, (v, identity) in enumerate(zip(prob, ids)):
                c = max(float(v), float(np.float32(1)-v))
                base = c >= original_cutoff
                rank = c > cutoff or (c == cutoff and identity <= boundary)
                for policy, take in zip(masks, [base, rank, base and not zero[k], rank and not zero[k]]):
                    masks[policy].append(take)
            for policy, mask in masks.items():
                np.testing.assert_array_equal(mask, a[f'{name}_{split}_{policy}'])
                row = next(r for r in rows if r['arm'] == model['arm'] and int(r['seed']) == model['seed'] and r['split'] == split and r['policy'] == policy)
                assert sum(mask) == int(row['accepted'])
    with tempfile.TemporaryDirectory(prefix='study09-repeat-') as tmp:
        repeat = Path(tmp)
        main(repeat)
        for filename in ['diagnostics.csv', 'outcomes.csv', 'bootstrap_empty.csv', 'provenance.json']:
            assert (repeat / filename).read_bytes() == (out / filename).read_bytes(), filename
        b = dict(np.load(repeat / 'mechanism.npz'))
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    result = dict(previous_artifacts_unchanged=len(previous), model_weights_verified=6,
                  scalar_decision_rows_verified=len(rows), deterministic_arrays_verified=len(a),
                  bootstrap_draws_per_model=2000, control83_original_noise_empty=54,
                  input_hashes_verified=len(provenance['inputs_sha256']), passed=True)
    with (out / 'verification.json').open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    verify(parser.parse_args().output)
