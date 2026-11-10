import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from high_resolution_severity import perturb,logit_risk,joint_band
from high_resolution_robustness import transform


class SeverityTests(unittest.TestCase):
    def test_zero_and_known_severities_reproduce_anchors(self):
        image=np.random.default_rng(7).integers(0,256,(224,224),dtype=np.uint8)
        for family in ['noise','blur']:np.testing.assert_array_equal(perturb(image,2,family,0),image)
        np.testing.assert_array_equal(perturb(image,2,'noise',.1),transform(image,2,'noise_input','native_224'))
        np.testing.assert_array_equal(perturb(image,2,'blur',8),transform(image,2,'relative_blur','native_224'))
        with self.assertRaises(ValueError):perturb(image,2,'noise',-.1)

    def test_logit_ranking_survives_probability_saturation(self):
        z=np.array([100.,50.,1.]);y=np.array([1,0,1])
        self.assertEqual(logit_risk(y,z,1/3),0)
        self.assertAlmostEqual(logit_risk(y,z,1),1/3)

    def test_joint_band_uses_maximum_grid_deviation(self):
        band=joint_band(np.array([0.,0.]),np.array([[1.,0.],[-1.,0.],[0.,2.],[0.,-2.]]))
        self.assertEqual(band['radius'],2)
        self.assertEqual(band['low'],[-2,-2])
        self.assertEqual(joint_band(np.array([1.]),np.ones((5,1)))['radius'],0)
        with self.assertRaises(ValueError):joint_band(np.array([0.,0.]),np.ones((5,1)))


if __name__=='__main__':unittest.main()
