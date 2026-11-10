import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from five_seed_analysis import seed_interval,retained_error

class FiveSeedTests(unittest.TestCase):
    def test_seed_interval_has_expected_mean_and_width(self):
        s=seed_interval([1,2,3,4,5])
        self.assertEqual(s['mean'],3)
        self.assertAlmostEqual(s['ci'][1]-3,2.7764451051977987*np.sqrt(.5))
        with self.assertRaises(ValueError):seed_interval([1,np.nan])

    def test_empty_retained_set_is_undefined(self):
        self.assertTrue(np.isnan(retained_error(np.array([0,1]),np.array([.6,.6]),.9)))
        self.assertEqual(retained_error(np.array([0,1]),np.array([.6,.6]),.5),.5)

if __name__=='__main__':unittest.main()
