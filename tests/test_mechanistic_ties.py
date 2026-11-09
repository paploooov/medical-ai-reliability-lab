import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from mechanistic_ties import keys, fit_rule, apply_rule, counts
from noise_diagnostics import inspect
from resolution_study import ResolutionCNN


class MechanisticTiesTests(unittest.TestCase):
    def test_exact_integer_target_and_rounding(self):
        for n in [5, 10, 262, 621]:
            c = np.full(n, .75)
            ids = keys('val', range(n))
            self.assertEqual(apply_rule(c, ids, fit_rule(c, ids)).sum(), 4*n//5)

    def test_priority_and_permutation_invariance(self):
        c = np.array([.6, .7, .7, .7, .9])
        ids = keys('val', range(5))
        rule = fit_rule(c, ids)
        self.assertEqual(rule, fit_rule(c[::-1], ids[::-1]))
        mask = apply_rule(c, ids, rule)
        self.assertTrue(mask[-1])
        self.assertFalse(mask[0])

    def test_fixed_transfer_does_not_force_test_coverage(self):
        c = np.full(10, .8)
        rule = fit_rule(c, keys('val', range(10)))
        self.assertFalse(apply_rule(np.full(5, .7), keys('test', range(5)), rule).any())

    def test_zero_rejection_is_not_error_detection_or_refill(self):
        keep = np.array([True, True, True])
        zero = np.array([True, True, False])
        wrong = np.array([False, False, True])
        self.assertEqual(counts(keep & ~zero, wrong)['risk'], 1)
        self.assertEqual(counts(keep & ~zero, wrong)['accepted'], 1)

    def test_empty_risk_is_undefined(self):
        self.assertIsNone(counts(np.zeros(3, bool), np.ones(3, bool))['risk'])

    def test_duplicate_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            fit_rule(np.ones(3), np.array(['same'] * 3))

    def test_zero_hidden_implies_bias_logit_without_saturation(self):
        model = ResolutionCNN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.layers[-1].bias.fill_(-1)
        model.eval()
        z, zero, _ = inspect(model, torch.zeros(2, 1, 28, 28))
        self.assertTrue(zero.all())
        np.testing.assert_array_equal(z, [-1., -1.])
        self.assertTrue((expit(z) < 1).all())

    def test_float_saturation_is_separate_from_bias_ties(self):
        self.assertEqual(float(expit(np.float32(100))), 1)
        self.assertLess(float(expit(np.float32(1))), 1)


if __name__ == '__main__':
    unittest.main()
