import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import unittest
import numpy as np
import torch
from cnn_study import CNN
from scipy.special import expit
class CNNTests(unittest.TestCase):
    def test_shape_and_gradient(self):
        m=CNN();x=torch.rand(4,1,28,28);z=m(x)
        self.assertEqual(tuple(z.shape),(4,))
        torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.,0.,1.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_temperature_preserves_decisions(self):
        z=np.array([-5.,-1.,.1,2.,8.])
        for t in [.2,1.,3.]:
            np.testing.assert_array_equal(expit(z)>=.5,expit(z/t)>=.5)
            np.testing.assert_array_equal(np.argsort(abs(z)),np.argsort(abs(z/t)))
if __name__=='__main__':unittest.main()
