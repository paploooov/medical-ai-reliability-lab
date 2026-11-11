import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from original_source_audit import unpad,appearance

class SourceAuditTests(unittest.TestCase):
    def test_unpadding_keeps_real_black_pixels(self):
        x=np.zeros((8,8),dtype=np.uint8);x[2:6]=128;x[3,3]=0
        c=unpad(x,100,50);self.assertEqual(c.shape,(4,8));self.assertEqual(c[1,3],0)
    def test_odd_padding_matches_preparation(self):
        x=np.arange(64).reshape(8,8);np.testing.assert_array_equal(unpad(x,8,5),x[1:6])
    def test_constant_image_has_no_texture(self):
        v=appearance(np.full((20,30),128,np.uint8));self.assertAlmostEqual(v[0],128/255);self.assertAlmostEqual(v[1],0);self.assertEqual(v[-1],0)

if __name__=='__main__':unittest.main()
