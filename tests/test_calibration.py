import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from calibration_study import cutoffs,summarize,percentile_summary
import numpy as np
class CalibrationTests(unittest.TestCase):
 def test_same_pool_permuted_same_cutoff(self):
  c=np.array([[.51,.6,.8,.9],[.7,.7,.7,.8]])
  np.testing.assert_allclose(cutoffs(c,np.arange(4)),cutoffs(c,np.array([3,1,0,2])))
 def test_tied_quantile_can_accept_everything(self):
  c=np.full((6,10),.7);cuts=cutoffs(c,np.arange(10));self.assertTrue((c>=cuts[:,None]).all())
 def test_identical_arms_zero_contrast(self):
  c=np.full((6,2,4),.9);w=np.zeros((6,2,4),dtype=bool);w[:,:,0]=True
  stats=summarize(c,w,np.full(6,.8),np.arange(4));self.assertEqual(stats[8],0);self.assertEqual(stats[0],.25);self.assertEqual(stats[4],1)
 def test_empty_acceptance_explicit(self):
  stats=summarize(np.full((6,2,4),.6),np.zeros((6,2,4),dtype=bool),np.full(6,.8),np.arange(4));self.assertTrue(np.isnan(stats[8]));self.assertEqual(stats[4],0)
 def test_undefined_replicates_counted(self):
  result=percentile_summary(np.array([np.nan,.1,.2]));self.assertEqual(result['finite'],2);self.assertEqual(result['undefined'],1)
if __name__=='__main__':unittest.main()
