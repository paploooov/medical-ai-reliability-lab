import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_thoraxis_h1_h2 import risk_weights,counts_metrics

class FrozenMetrics(unittest.TestCase):
    def test_boundary_tie_is_not_broken_by_label(self):
        c=np.array([.9,.8,.8,.8,.6]);w=risk_weights(c,np.ones(5))
        np.testing.assert_allclose(w,[1,1,1,1,0])
        w=risk_weights(np.array([.9,.8,.8,.8,.8]),np.ones(5))
        np.testing.assert_allclose(w,[1,.75,.75,.75,.75])
    def test_bootstrap_ties_share_sample_mass(self):
        w=risk_weights(np.array([.9,.8,.8]),np.array([2.,3.,5.]))
        np.testing.assert_allclose(w,[2,2.25,3.75]);self.assertAlmostEqual(w.sum(),8)
    def test_zero_mass_and_empty_class_are_undefined(self):
        np.testing.assert_array_equal(risk_weights(np.array([.5,.6]),np.zeros(2)),[0,0])
        d=dict(y=np.array([0,1]),pred=np.array([False,True]),error=np.array([False,False]),accept=np.array([False,False]),group=np.array(['a','b']))
        r=counts_metrics(d,np.array([True,True]),np.ones(2),np.array([.8,.8]),True)
        self.assertTrue(np.isnan(r['accepted_error']));self.assertTrue(np.isnan(r['group_any_error_one_sided95_upper']))
    def test_origin_uses_global_rank_weights(self):
        c=np.array([.99,.98,.97,.96,.51]);rw=risk_weights(c,np.ones(5))
        self.assertEqual(rw[4],0) # A low-confidence source subgroup gets no invented quota.
    def test_zero_errors_do_not_imply_zero_risk(self):
        d=dict(y=np.array([0,1]),pred=np.array([False,True]),error=np.array([False,False]),accept=np.array([True,True]),group=np.array(['a','b']))
        r=counts_metrics(d,np.ones(2,dtype=bool),np.ones(2),np.array([.8,.8]),True)
        self.assertAlmostEqual(r['group_any_error_one_sided95_upper'],1-np.sqrt(.05))
if __name__=='__main__':unittest.main()
