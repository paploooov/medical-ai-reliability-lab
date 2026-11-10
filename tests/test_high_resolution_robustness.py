import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from high_resolution_robustness import transform, quantize, interval


class RobustnessTests(unittest.TestCase):
    def test_noise_is_index_keyed_and_independent_of_order(self):
        x = np.full((224, 224), 128, np.uint8)
        a = transform(x, 17, 'noise_input', 'native_224')
        transform(x, 83, 'noise_input', 'native_224')
        np.testing.assert_array_equal(a, transform(x, 17, 'noise_input', 'native_224'))
        self.assertFalse(np.array_equal(a, transform(x, 83, 'noise_input', 'native_224')))
        np.testing.assert_array_equal(a, transform(x, 17, 'noise_input', 'downsample28_upsample224'))

    def test_noise_source_is_filtered_only_by_low_information_branch(self):
        x = np.full((224, 224), 128, np.uint8)
        source = transform(x, 17, 'noise_source', 'downsample28_upsample224')
        inputs = transform(x, 17, 'noise_input', 'downsample28_upsample224')
        self.assertLess(source.std(), inputs.std()/3)
        np.testing.assert_array_equal(transform(x,17,'noise_source','native_224'),
                                      transform(x,17,'noise_input','native_224'))

    def test_blur_does_not_mix_cases_and_quantization_is_explicit(self):
        x = np.full((224, 224), 77, np.uint8)
        np.testing.assert_array_equal(transform(x, 1, 'relative_blur', 'native_224'), x)
        np.testing.assert_array_equal(quantize(np.array([-1, 0, .5, 1, 2])), [0, 0, 128, 255, 255])
        with self.assertRaises(ValueError): transform(x, 1, 'unplanned', 'native_224')
        with self.assertRaises(ValueError): transform(x, 1, 'clean', 'other_view')

    def test_undefined_risk_is_not_reported_as_zero(self):
        stat = interval(float('nan'), np.array([np.nan, np.nan]))
        self.assertIsNone(stat['point'])
        self.assertIsNone(stat['ci'])
        self.assertEqual(stat['undefined_replicates'], 2)



# The independent verifier uses grouped confidence levels, not positional tie-breaking.
class IndependentRiskTests(unittest.TestCase):
    def test_fractional_ties_and_permutation(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/verify_high_resolution_robustness.py'
        spec=importlib.util.spec_from_file_location('robustness_verifier',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        y=np.array([1,0,1]);p=np.array([.9,.9,.4])
        self.assertAlmostEqual(module.grouped_risk(y,p,.5),.5)
        self.assertAlmostEqual(module.grouped_risk(y,p,1),2/3)
        order=np.array([2,0,1])
        self.assertAlmostEqual(module.grouped_risk(y[order],p[order],.5),.5)

if __name__ == '__main__': unittest.main()
