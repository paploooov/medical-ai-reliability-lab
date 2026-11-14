import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_thoraxis_partial_spearman import estimate
class PartialSpearman(unittest.TestCase):
    def test_weighted_equals_expanded_cases(self):
        values=[np.array(v,dtype=float) for v in [[1,2,2,4,3,7],[4,1,3,3,7,6],[2,3,5,1,6,8],[5,2,3,4,7,8]]];w=np.array([2,1,0,3,1,2]);a=estimate(*values,w);b=estimate(*[np.repeat(v,w) for v in values],np.ones(sum(w)))
        self.assertAlmostEqual(a['partial'],b['partial']);self.assertAlmostEqual(a['raw'],b['raw'])
    def test_size_explained_score_is_undefined_not_zero(self):
        x=np.arange(1.,11);v=estimate(x,x[::-1],np.log(x),np.log(x+2),np.ones(10));self.assertTrue(np.isnan(v['partial']))
    def test_logarithm_preserves_rank_adjustment(self):
        x=np.array([2,4,1,8,5,3,9.]);y=np.array([3,2,1,5,7,6,4.]);w=np.array([10,4,9,2,8,6,7.]);h=np.array([7,8,3,2,6,4,1.])
        self.assertAlmostEqual(estimate(x,y,w,h,np.ones(7))['partial'],estimate(x,y,np.log(w),np.log(h),np.ones(7))['partial'])
if __name__=='__main__':unittest.main()
