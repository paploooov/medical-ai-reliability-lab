import csv
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('external_views', ROOT / 'scripts/summarize_external_views.py')
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


class ExternalViewTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / 'configs/external_rsna_sample.csv').open() as f:
            self.sample = list(csv.DictReader(f))
        with (ROOT / 'reports/external/results.csv').open() as f:
            self.results = list(csv.DictReader(f))

    def test_strata_reconstruct_overall_accuracy(self):
        rows = views.summarize_views(self.sample, self.results)
        self.assertEqual(len(rows), 24)
        for arm in ('control_28', 'augmented_28', 'control_64'):
            for condition in ('clean', 'noise', 'relative_blur', 'contrast'):
                strata = [r for r in rows if r['arm'] == arm and r['condition'] == condition]
                self.assertEqual(sum(r['n'] for r in strata), 2000)
                self.assertEqual(sum(r['positive'] for r in strata), 1000)
                overall = [float(r['accuracy']) for r in self.results if r['dataset'] == 'external'
                           and r['arm'] == arm and r['condition'] == condition]
                self.assertAlmostEqual(sum(r['n'] * r['accuracy'] for r in strata) / 2000,
                                       sum(overall) / 3)
                for r in strata:
                    self.assertAlmostEqual(r['accuracy'], r['prevalence'] * r['sensitivity']
                                           + (1-r['prevalence']) * r['specificity'])

    def test_missing_seed_and_wrong_sample_rejected(self):
        results = [r for r in self.results if not (r['dataset'] == 'external_AP' and r['seed'] == '17')]
        with self.assertRaises(ValueError):
            views.summarize_views(self.sample, results)
        with self.assertRaises(ValueError):
            views.summarize_views(self.sample[1:], self.results)


if __name__ == '__main__':
    unittest.main()
