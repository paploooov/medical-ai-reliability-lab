import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from selective_review import acceptance_weights,risk
import numpy as np
class CoverageTests(unittest.TestCase):
 def test_exact_fraction_and_boundary_tie(self):
  w=acceptance_weights(np.array([.99,.9,.9,.5]),.5)
  np.testing.assert_allclose(w,[1,.5,.5,0]);self.assertEqual(w.sum(),2)
 def test_all_tied_does_not_use_labels(self):
  p=np.full(5,.9);np.testing.assert_allclose(acceptance_weights(p,.8),np.full(5,.8))
  self.assertAlmostEqual(risk(np.array([0,0,1,1,1]),p,.8),.4)
 def test_permutation_invariance(self):
  p=np.array([.9,.9,.6,.99]);y=np.array([0,1,0,1]);order=[3,1,2,0]
  self.assertAlmostEqual(risk(y,p,.8),risk(y[order],p[order],.8))
 def test_full_coverage_matches_error(self):
  p=np.array([.1,.9,.9]);y=np.array([0,0,1]);self.assertAlmostEqual(risk(y,p,1),1/3)
 def test_fractional_target(self):self.assertAlmostEqual(acceptance_weights(np.array([.9,.6,.7]),.8).sum(),2.4)
 def test_invalid_inputs(self):
  for p,q in [(np.array([.5]),0),(np.array([]),.8),(np.array([np.nan]),.8),(np.array([2.]),.8)]:
   with self.assertRaises(ValueError):acceptance_weights(p,q)
if __name__=='__main__':unittest.main()
