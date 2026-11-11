import sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_original_xray import patient_id,letterbox

class OriginalXrayTests(unittest.TestCase):
    def test_repeated_patient_images_share_group(self):
        self.assertEqual(patient_id('person12_bacteria_3.jpeg'),patient_id('person12_virus_4.jpeg'))
        self.assertEqual(patient_id('NORMAL2-IM-0123-0001-0002.jpeg'),'normal2-im-0123')
        self.assertNotEqual(patient_id('IM-0123-0001.jpeg'),patient_id('NORMAL2-IM-0123-0001.jpeg'))
        with self.assertRaises(ValueError):patient_id('unknown.jpeg')
    def test_letterbox_preserves_shape_and_does_not_stretch(self):
        a=letterbox(Image.fromarray(np.full((50,100),255,np.uint8)),512)
        self.assertEqual(a.shape,(512,512));self.assertEqual(a.dtype,np.uint8)
        self.assertTrue((a[:128]==0).all());self.assertTrue((a[128:384]==255).all());self.assertTrue((a[384:]==0).all())
    def test_grayscale_rgb_consistency(self):
        a=np.arange(64,dtype=np.uint8).reshape(8,8)
        np.testing.assert_array_equal(letterbox(Image.fromarray(a),16),letterbox(Image.fromarray(np.repeat(a[:,:,None],3,axis=2)),16))
    def test_folded_scaler_preserves_nontrivial_logits(self):
        sys.path.insert(0,str(ROOT/'experiments'))
        from original_512 import folded_head
        rng=np.random.default_rng(7);x=rng.normal(size=(11,5));mean=rng.normal(size=5);scale=np.exp(rng.normal(size=5));coef=rng.normal(size=5);bias=.37
        w,b=folded_head(coef,bias,mean,scale)
        np.testing.assert_allclose(x@w+b,((x-mean)/scale)@coef+bias,atol=1e-12)
    def test_export_raw_pixel_normalization(self):
        import torch
        from export_original_coreml import RawGrayscaleModel
        model=RawGrayscaleModel(torch.nn.Identity())
        raw=torch.tensor([[[[0.,127.5,255.]]]])
        actual=model(raw).numpy()
        expected=(np.repeat(raw.numpy(),3,axis=1)/255-np.array([.485,.456,.406])[None,:,None,None])/np.array([.229,.224,.225])[None,:,None,None]
        np.testing.assert_allclose(actual,expected,atol=3e-7)
    def test_class_errors_and_empty_acceptance(self):
        from characterize_original_512 import class_metrics
        y=np.array([0,0,1,1]);z=np.array([-2.,2.,-2.,2.])
        np.testing.assert_allclose(class_metrics(y,z,.8),[.5,.5,1,1,.5,.5])
        rejected=class_metrics(y,z,.9)
        np.testing.assert_allclose(rejected[:4],[.5,.5,0,0]);self.assertTrue(np.isnan(rejected[4:]).all())
    def test_specificity_threshold_is_conservative_with_ties(self):
        sys.path.insert(0,str(ROOT/'experiments'))
        from original_threshold_transfer import threshold_for_specificity
        for scores in [np.arange(10,dtype=float),np.array([9.,9.,9.,6.,5.,4.,3.,2.,1.,0.])]:
            for target in [.9,.95,1.]:
                threshold=threshold_for_specificity(scores,target)
                self.assertGreaterEqual(np.mean(scores<threshold),target)
    def test_threshold_comparison_does_not_round_back_to_float32(self):
        from original_threshold_transfer import threshold_for_specificity,operating_point
        scores=np.array([1.,2.],dtype=np.float32);threshold=threshold_for_specificity(scores,1.)
        result=operating_point(np.array([0,0,1]),np.array([1.,2.,3.],dtype=np.float32),threshold)
        np.testing.assert_array_equal(result,[1.,1.])

if __name__=='__main__':unittest.main()
