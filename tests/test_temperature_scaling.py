import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from temperature_scaling import fit_temperature,nll


class TemperatureTests(unittest.TestCase):
    def test_recovers_known_temperature_from_exact_frequencies(self):
        # Each group has exactly 80% positives; true logit = log(4).
        y=np.tile([1,1,1,1,0],100)
        z=np.full(len(y),2*np.log(4))
        self.assertAlmostEqual(fit_temperature(y,z),2.0,places=5)

    def test_stable_nll_matches_independent_reference(self):
        y=np.array([0,1,0,1]);z=np.array([-3,2,1,-2.])
        self.assertAlmostEqual(nll(y,z),log_loss(y,expit(z)),places=12)
        self.assertEqual(nll(np.array([1,0]),np.array([-1000.,1000.])),1000)

    def test_boundary_and_invalid_inputs(self):
        self.assertEqual(fit_temperature(np.array([0,0]),np.array([2.,3.])),20)
        with self.assertRaises(ValueError):fit_temperature([0],[float('nan')])
        with self.assertRaises(ValueError):fit_temperature([],[])
        with self.assertRaises(ValueError):fit_temperature([0],[1],(0,1))

    def test_scaling_preserves_decisions_ranking_and_transformed_cutoff(self):
        z=np.array([-8.,-3.,-.5,0.,.5,2.,3.,8.])
        for t in [.05, .5, 3., 20.]:
            np.testing.assert_array_equal(z>=0,z/t>=0)
            np.testing.assert_array_equal(np.argsort(abs(z)),np.argsort(abs(z/t)))
            np.testing.assert_array_equal(abs(z)>=3,abs(z/t)>=3/t)

if __name__=='__main__':unittest.main()
