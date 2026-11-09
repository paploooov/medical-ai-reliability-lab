import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from transfer_study import variants,evaluate


class TransferTests(unittest.TestCase):
    def test_noise_deterministic_and_inputs_unchanged(self):
        x=np.arange(32,dtype=np.float32).reshape(2,4,4)/31
        original=x.copy();a=variants(x);b=variants(x)
        np.testing.assert_array_equal(x,original)
        for name in a:np.testing.assert_array_equal(a[name],b[name])
        self.assertGreaterEqual(a['noise'].min(),0);self.assertLessEqual(a['noise'].max(),1)

    def test_blur_preserves_constant_images_without_cross_case_mixing(self):
        x=np.stack([np.zeros((4,4)),np.ones((4,4))]).astype(np.float32)
        np.testing.assert_array_equal(variants(x)['blur'],x)

    def test_contrast_definition(self):
        x=np.array([[[0,.5,1]]],dtype=np.float32)
        np.testing.assert_array_equal(variants(x)['contrast'],[[[.25,.5,.75]]])

    def test_exact_ties_not_approximate_ties(self):
        p=np.array([.75,np.nextafter(np.float32(.75),np.float32(1))],dtype=np.float32)
        rows,_=evaluate(p,np.array([True,False]),np.array([1,0]),np.array(['a','b']),.75,(.75,'z'))
        self.assertEqual(rows[0]['threshold_tie_count'],1)
        self.assertEqual(rows[0]['ties_with_zero'],1)
        self.assertEqual(rows[0]['accuracy'],.5)
        self.assertEqual(rows[0]['risk'],.5)
        self.assertEqual(rows[1]['risk'],1)
        self.assertEqual(rows[1]['coverage'],.5)

    def test_zero_rejection_leaves_undefined_risk(self):
        rows,_=evaluate(np.array([.8]),np.array([True]),np.array([1]),np.array(['a']),.7,(.7,'a'))
        self.assertEqual(rows[0]['risk'],0)
        self.assertIsNone(rows[1]['risk'])
        self.assertEqual(rows[1]['accepted'],0)

    def test_original_threshold_and_rank_threshold_are_distinct(self):
        rows,masks=evaluate(np.array([.8,.9]),np.array([False,False]),np.array([1,1]),np.array(['a','b']),.75,(.85,'z'))
        self.assertEqual(rows[0]['accepted'],2)
        self.assertEqual(rows[1]['accepted'],1)
        np.testing.assert_array_equal(masks[1],[False,True])


if __name__=='__main__':unittest.main()
