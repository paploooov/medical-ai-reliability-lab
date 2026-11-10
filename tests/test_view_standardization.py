import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from external_view_analysis import standardized_metrics,interval


class StandardizationTests(unittest.TestCase):
    def test_ratio_of_weighted_counts_not_average_of_risks(self):
        # Negatives: 80% accepted, 20% of all cases accepted and wrong.
        # Positives: 20% accepted, 10% of all cases accepted and wrong.
        cells=np.array([[.7,.8,.2],[.8,.2,.1]])
        accuracy,coverage,risk=standardized_metrics(cells)
        self.assertAlmostEqual(accuracy,.75)
        self.assertAlmostEqual(coverage,.5)
        self.assertAlmostEqual(risk,.3)
        self.assertNotAlmostEqual(risk,(.2/.8+.1/.2)/2)

    def test_empty_acceptance_undefined_and_batched_inputs(self):
        cells=np.array([[[1,0,0],[1,0,0]],[[1,1,0],[1,1,0]]])
        out=standardized_metrics(cells)
        self.assertTrue(np.isnan(out[0,2]))
        np.testing.assert_array_equal(out[1],[1,1,0])
        stat=interval(0,np.array([0,np.nan,1]))
        self.assertEqual(stat['undefined_replicates'],1)
        with self.assertRaises(ValueError):standardized_metrics(np.zeros((3,3)))

if __name__=='__main__':unittest.main()
