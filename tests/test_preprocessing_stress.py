import sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from original_preprocessing_stress import variant

class PreprocessingStressTests(unittest.TestCase):
    def test_gamma_preserves_padding_and_endpoints(self):
        a=np.array([[0,64],[128,255]],dtype=np.uint8)
        bright=variant(None,a,'gamma_0.8');dark=variant(None,a,'gamma_1.25')
        self.assertEqual(bright[0,0],0);self.assertEqual(dark[1,1],255)
        self.assertTrue((bright>=a).all());self.assertTrue((dark<=a).all())
    def test_crop_and_stretch_are_distinct(self):
        a=np.tile(np.arange(12,dtype=np.uint8),(6,1));im=Image.fromarray(a)
        crop=variant(im,None,'center_crop',size=6);np.testing.assert_array_equal(crop,a[:,3:9])
        self.assertFalse(np.array_equal(crop,variant(im,None,'stretch',size=6)))
    def test_bilinear_keeps_aspect_and_black_padding(self):
        a=variant(Image.new('L',(12,6),255),None,'bilinear',size=8)
        self.assertTrue((a[2:6]==255).all());self.assertTrue((a[:2]==0).all());self.assertTrue((a[6:]==0).all())
    def test_reference_geometry_uses_ties_to_even(self):
        from prepare_original_xray import letterbox
        for height,expected in [(641,320),(643,322)]:
            a=letterbox(Image.new('L',(1024,height),255),512)
            self.assertEqual(int((a[:,0]>0).sum()),expected)

if __name__=='__main__':unittest.main()
