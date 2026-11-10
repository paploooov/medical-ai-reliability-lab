import sys
from pathlib import Path
import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from high_resolution_augmentation import noisy_image,weighted_training
from high_resolution_robustness import transform


class AugmentedProbeTests(unittest.TestCase):
    def test_primary_noise_matches_prior_and_training_seed_differs(self):
        image=np.full((224,224),127,np.uint8)
        np.testing.assert_array_equal(noisy_image(image,83,42),transform(image,83,'noise_input','native_224'))
        self.assertFalse(np.array_equal(noisy_image(image,83,42),noisy_image(image,83,20261116)))

    def test_half_weights_preserve_loss_and_regularization_scale(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(60,4));y=(x[:,0]+rng.normal(size=60)>.2).astype(int)
        xx,yy,w=weighted_training(x,x,y)
        self.assertEqual(w.sum(),len(y))
        self.assertEqual(len(xx),2*len(x))
        one=LogisticRegression(C=.2,tol=1e-10,max_iter=3000).fit(x,y)
        two=LogisticRegression(C=.2,tol=1e-10,max_iter=3000).fit(xx,yy,sample_weight=w)
        np.testing.assert_allclose(one.decision_function(x),two.decision_function(x),atol=1e-8)
        with self.assertRaises(ValueError):weighted_training(x,x[:-1],y)


if __name__=='__main__':unittest.main()
