import sys, unittest, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from external_eval import summarize, BOOT, BOOT_SEED
from external_stratify import (
    load_primary, point_rows, bootstrap_view, class_counts,
    METRIC_KEYS, ARMS_ORDER, SEEDS, VIEWS,
)


class PrimaryAlignmentTests(unittest.TestCase):
    def test_labels_and_views_align_with_stored_predictions(self):
        data = load_primary()
        self.assertEqual(len(data['y']), 2000)
        self.assertEqual(set(data['view']), {'AP', 'PA'})
        self.assertEqual(int((data['view'] == 'AP').sum()), 964)
        self.assertEqual(int((data['view'] == 'PA').sum()), 1036)
        # Class mix is the documented confound: AP mostly Lung Opacity, PA mostly Normal.
        ap, pa = data['view'] == 'AP', data['view'] == 'PA'
        self.assertEqual(int((data['y'][ap] == 1).sum()), 766)
        self.assertEqual(int((data['y'][ap] == 0).sum()), 198)
        self.assertEqual(int((data['y'][pa] == 1).sum()), 234)
        self.assertEqual(int((data['y'][pa] == 0).sum()), 802)
        self.assertTrue(np.array_equal(data['y'], data['pred']['labels__external']))

    def test_class_counts_match_sample_csv(self):
        data = load_primary()
        for v, n, n_pos, n_neg in [('AP', 964, 766, 198), ('PA', 1036, 234, 802)]:
            c = class_counts(data['view'] == v, data['y'], data['cls'])
            self.assertEqual(c['n'], n)
            self.assertEqual(c['n_positive'], n_pos)
            self.assertEqual(c['n_negative'], n_neg)
            self.assertEqual(c['class_counts']['Lung Opacity'], n_pos)
            self.assertEqual(c['class_counts']['Normal'], n_neg)
            self.assertAlmostEqual(c['prevalence'], n_pos / n)


class PointEstimateTests(unittest.TestCase):
    def test_clean_points_match_existing_results_csv(self):
        data = load_primary()
        rows = point_rows(data, 'clean')
        existing = [
            r for r in csv.DictReader((ROOT / 'reports/external/results.csv').open())
            if r['dataset'] in ('external_AP', 'external_PA') and r['condition'] == 'clean'
        ]
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(existing), 18)
        by = {(r['dataset'], r['arm'], int(r['seed'])): r for r in existing}
        keys = ['accuracy', 'auroc', 'sensitivity', 'specificity', 'coverage', 'retained_error']
        for r in rows:
            old = by[f"external_{r['view']}", r['arm'], r['seed']]
            self.assertEqual(int(old['n']), r['n'])
            for k in keys:
                self.assertAlmostEqual(float(old[k]), r[k], places=12)

    def test_summarize_within_view_uses_only_that_view(self):
        data = load_primary()
        arm, seed = 'control_28', 17
        p = data['pred'][f'external__{arm}__s{seed}__clean']
        cut = data['cuts'][arm, seed]
        mask = data['view'] == 'PA'
        s = summarize(data['y'][mask], p[mask], cut)
        self.assertEqual(s['n'], 1036)
        # A wrong mask would not reproduce the stored PA row.
        stored = next(
            r for r in csv.DictReader((ROOT / 'reports/external/results.csv').open())
            if r['dataset'] == 'external_PA' and r['arm'] == arm and int(r['seed']) == seed and r['condition'] == 'clean'
        )
        self.assertAlmostEqual(s['auroc'], float(stored['auroc']), places=12)
        self.assertAlmostEqual(s['accuracy'], float(stored['accuracy']), places=12)


class BootstrapTests(unittest.TestCase):
    def test_percentile_interval_contains_point_on_toy_data(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = np.clip(0.4 + 0.3 * y + rng.normal(0, 0.15, 200), 0, 1)
        # One fake "data" bundle is unnecessary: just check nanquantile on resamples of accuracy.
        n = len(y)
        draws = [rng.integers(0, n, n) for _ in range(400)]
        acc = np.array([((p[ix] >= 0.5) == y[ix]).mean() for ix in draws])
        lo, hi = np.quantile(acc, [0.025, 0.975])
        point = ((p >= 0.5) == y).mean()
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_committed_clean_summary_is_reproducible(self):
        data = load_primary()
        fresh = bootstrap_view(data, 'clean', replicates=BOOT, seed=BOOT_SEED)
        stored = json.loads((ROOT / 'reports/external/stratified_view.json').read_text())
        self.assertEqual(stored['bootstrap']['replicates'], BOOT)
        self.assertEqual(stored['bootstrap']['seed'], BOOT_SEED)
        self.assertEqual(stored['condition'], 'clean')
        for v in VIEWS:
            self.assertEqual(stored['results'][v]['n'], fresh[v]['n'])
            self.assertEqual(stored['results'][v]['distribution']['n_positive'], fresh[v]['distribution']['n_positive'])
            for arm in ARMS_ORDER:
                for k in METRIC_KEYS:
                    self.assertAlmostEqual(stored['results'][v][arm][k]['point'], fresh[v][arm][k]['point'], places=12)
                    np.testing.assert_allclose(stored['results'][v][arm][k]['ci'], fresh[v][arm][k]['ci'], atol=1e-12, rtol=0)
                    lo, hi = stored['results'][v][arm][k]['ci']
                    pt = stored['results'][v][arm][k]['point']
                    self.assertLessEqual(lo, pt + 1e-12)
                    self.assertLessEqual(pt, hi + 1e-12)
        for arm in ARMS_ORDER:
            for k in METRIC_KEYS:
                np.testing.assert_allclose(
                    stored['results']['AP_minus_PA'][arm][k]['ci'],
                    fresh['AP_minus_PA'][arm][k]['ci'],
                    atol=1e-12, rtol=0,
                )
                # The difference of points equals AP minus PA.
                self.assertAlmostEqual(
                    stored['results']['AP_minus_PA'][arm][k]['point'],
                    stored['results']['AP'][arm][k]['point'] - stored['results']['PA'][arm][k]['point'],
                    places=12,
                )


if __name__ == '__main__':
    unittest.main()
