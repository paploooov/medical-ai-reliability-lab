import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch,numpy as np
from resolution_study import ResolutionCNN
from noise_diagnostics import inspect
class HiddenTests(unittest.TestCase):
 def test_zero_hidden_reduces_to_output_bias(self):
  m=ResolutionCNN()
  with torch.no_grad():
   for p in m.parameters():p.zero_()
   m.layers[-1].bias.fill_(-.4)
  z,zero,fracs=inspect(m,torch.rand(3,1,28,28))
  self.assertTrue(zero.all());np.testing.assert_allclose(z,-.4);self.assertTrue(all(np.all(v==1) for v in fracs.values()))
 def test_nonzero_hidden_not_marked_collapsed(self):
  m=ResolutionCNN()
  with torch.no_grad():
   for p in m.parameters():p.zero_()
   m.layers[-3].bias.fill_(1)
  _,zero,_=inspect(m,torch.rand(3,1,28,28));self.assertFalse(zero.any())
if __name__=='__main__':unittest.main()
