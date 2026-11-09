import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
from resolution_study import augment_batch
class AugmentationTests(unittest.TestCase):
 def test_noop(self):
  x=torch.rand(4,1,28,28);self.assertTrue(torch.equal(augment_batch(x,0,.1,torch.Generator()),x))
 def test_deterministic_bounded_nonmutating(self):
  x=torch.full((4,1,28,28),.5);copy=x.clone();a=augment_batch(x,1,.1,torch.Generator().manual_seed(7));b=augment_batch(x,1,.1,torch.Generator().manual_seed(7))
  self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(x,copy));self.assertFalse(torch.equal(a,x));self.assertTrue(((a>=0)&(a<=1)).all())
 def test_does_not_advance_global_rng(self):
  x=torch.ones(4,1,28,28);torch.manual_seed(42);state=torch.random.get_rng_state().clone();augment_batch(x,.5,.1,torch.Generator().manual_seed(7));self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
 def test_invalid(self):
  with self.assertRaises(ValueError):augment_batch(torch.ones(1,1,28,28),1.2,.1,torch.Generator())
if __name__=='__main__':unittest.main()
