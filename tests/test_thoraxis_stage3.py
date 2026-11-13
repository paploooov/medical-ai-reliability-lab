import sys,unittest
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from thoraxis_stage3_common import weighted_spearman,paired_metrics
from cache_thoraxis_bilinear import letterbox
from PIL import Image

class StageThree(unittest.TestCase):
    def test_bootstrap_ranks_equal_expanded_sample_with_ties(self):
        x=np.array([.1,.1,.3,.8]);y=np.array([1,2,2,0]);w=np.array([3,2,0,4]);expected=spearmanr(np.repeat(x,w),np.repeat(y,w)).statistic
        self.assertAlmostEqual(weighted_spearman(x,y,w),expected)
    def test_empty_constant_and_small_pairs_are_undefined(self):
        for x,y,w in [([1,1,1],[1,2,3],[1,1,1]),([1,2],[2,1],[1,1]),([],[],[])]:self.assertTrue(np.isnan(weighted_spearman(x,y,w)))
    def test_paired_identical_outputs_have_zero_change(self):
        y=np.array([0,0,1,1]);z=np.array([-2,1,2,-1.]);r=paired_metrics(y,z,z,.9,np.ones(4))
        self.assertEqual(r['decision_changes'],0);self.assertEqual(r['delta']['R80'],0);self.assertEqual(r['delta']['specificity'],0)
    def test_bilinear_letterbox_preserves_geometry(self):
        a=np.full((5,10),100,dtype=np.uint8);out=letterbox(Image.fromarray(a),20)
        self.assertEqual(out.shape,(20,20));self.assertTrue(np.all(out[:5]==0));self.assertTrue(np.all(out[5:15]==100));self.assertTrue(np.all(out[15:]==0))
if __name__=='__main__':unittest.main()
