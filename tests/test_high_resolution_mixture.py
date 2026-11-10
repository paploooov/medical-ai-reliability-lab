import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from high_resolution_mixed_augmentation import mixture


class MixtureTests(unittest.TestCase):
    def test_weights_keep_case_mass_and_omit_zero_weight_rows(self):
        y=np.array([0,1,1]);features={k:np.full((3,2),v) for k,v in [('clean',1),('noise',2),('blur',3)]}
        x,yy,w=mixture(features,y,dict(clean=.5,noise=.25,blur=.25))
        self.assertEqual(len(x),9);self.assertEqual(w.sum(),3)
        np.testing.assert_array_equal(yy,np.tile(y,3))
        self.assertAlmostEqual(w[yy==1].sum(),2)
        x,_,w=mixture(features,y,dict(clean=.75,noise=.25,blur=0))
        self.assertEqual(len(x),6);self.assertEqual(w.sum(),3)
        with self.assertRaises(ValueError):mixture(features,y,dict(clean=.5,noise=.25,blur=.5))
        with self.assertRaises(ValueError):mixture(features,y,dict(clean=1.1,noise=-.1))


if __name__=='__main__':unittest.main()
