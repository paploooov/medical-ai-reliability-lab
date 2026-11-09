import sys,unittest,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from external_data import git_blob_sha1,to_uint8,centre_crop,preprocess,draw_sample
from external_eval import auroc,reliability,ece,risk_at,perturb,summarize
from selective_review import risk as repository_risk
from sklearn.metrics import roc_auc_score

class ExternalDataTests(unittest.TestCase):
 def test_git_blob_hash_matches_git(self):
  self.assertEqual(git_blob_sha1(b'hello\n'),'ce013625030ba8dba906f756967f9e9ca394464a')
 def test_uint8_passthrough_inversion_and_scaling(self):
  a=np.array([[0,255],[10,20]],dtype=np.uint8)
  np.testing.assert_array_equal(to_uint8(a),a);np.testing.assert_array_equal(to_uint8(a,'MONOCHROME1'),255-a)
  b=np.array([[0,4095],[0,4095]],dtype=np.uint16);np.testing.assert_array_equal(to_uint8(b),[[0,255],[0,255]])
  with self.assertRaises(ValueError):to_uint8(np.zeros((2,2,3),dtype=np.uint8))
 def test_centre_crop_uses_short_edge(self):
  a=np.arange(6*10).reshape(6,10);c=centre_crop(a);self.assertEqual(c.shape,(6,6));np.testing.assert_array_equal(c,a[:,2:8])
  self.assertEqual(centre_crop(a.T).shape,(6,6))
 def test_preprocess_shape_dtype_and_constant_image(self):
  out=preprocess(np.full((300,200),77,dtype=np.uint8),28);self.assertEqual(out.shape,(28,28));self.assertEqual(out.dtype,np.uint8);self.assertTrue((out==77).all())
  self.assertEqual(preprocess(np.zeros((1024,1024),dtype=np.uint8),64).shape,(64,64))
 def test_sample_is_deterministic_unique_and_stratified(self):
  rows=[(f'p{i:04d}','A' if i%3 else 'B') for i in range(300)]
  design={'A':(20,1,'primary'),'B':(10,0,'primary')}
  s1=draw_sample(rows,7,design);s2=draw_sample(list(reversed(rows)),7,design)
  self.assertEqual(s1,s2);self.assertEqual(len({r['patientId'] for r in s1}),30)
  self.assertEqual(sum(r['label'] for r in s1),20);self.assertNotEqual(s1,draw_sample(rows,8,design))
  with self.assertRaises(ValueError):draw_sample(rows,7,{'B':(1000,0,'primary')})
 def test_committed_sample_has_one_image_per_patient(self):
  import csv
  rows=list(csv.DictReader((ROOT/'configs/external_rsna_sample.csv').open()))
  self.assertEqual(len(rows),2500);self.assertEqual(len({r['patientId'] for r in rows}),2500)
  prim=[r for r in rows if r['arm']=='primary'];self.assertEqual(sum(int(r['label']) for r in prim),1000)
  self.assertEqual({r['rsna_class'] for r in rows if r['arm']=='secondary'},{'No Lung Opacity / Not Normal'})

class ExternalMetricTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(0);self.y=rng.integers(0,2,500);self.p=np.clip(.5*self.y+.25+rng.normal(0,.2,500),0,1)
 def test_auroc_matches_sklearn_with_ties(self):
  p=np.round(self.p,1);self.assertAlmostEqual(auroc(self.y,p),roc_auc_score(self.y,p),places=12)
  self.assertTrue(np.isnan(auroc(np.zeros(3),np.array([.1,.2,.3]))))
 def test_ece_perfect_and_known_value(self):
  y=np.array([1,1,0,0]);p=np.array([1.,1.,0.,0.]);self.assertAlmostEqual(ece(y,p),0.0)
  y=np.array([1,0]);p=np.array([.9,.9]);self.assertAlmostEqual(ece(y,p),.4)  # one bin: conf .9, acc .5
 def test_reliability_bins_count_all_cases(self):
  n,acc,mc=reliability(self.y,self.p);self.assertEqual(n.sum(),500);self.assertEqual(len(n),15)
  self.assertTrue(np.all((mc[n>0]>=.5)&(mc[n>0]<=1)))
 def test_risk_matches_repository_selective_review(self):
  for c in [.1,.5,.8,1.]:self.assertAlmostEqual(risk_at(self.y,self.p,c),repository_risk(self.y,self.p,c),places=12)
 def test_perturbations_match_repository_definitions(self):
  x=np.random.default_rng(1).random((3,28,28))
  np.testing.assert_array_equal(perturb(x,'clean',28),x)
  np.testing.assert_allclose(perturb(x,'noise',28),np.clip(x+np.random.default_rng(42).normal(0,.1,x.shape),0,1))
  np.testing.assert_allclose(perturb(x,'contrast',28),np.clip(.5+.5*(x-.5),0,1))
  self.assertAlmostEqual(perturb(x,'relative_blur',28).mean(),x.mean(),places=6)
  with self.assertRaises(ValueError):perturb(x,'rotate',28)
 def test_summary_fixed_cutoff(self):
  s=summarize(self.y,self.p,.75);conf=np.maximum(self.p,1-self.p)
  self.assertAlmostEqual(s['coverage'],(conf>=.75).mean());self.assertAlmostEqual(s['accuracy'],((self.p>=.5)==self.y).mean())
if __name__=='__main__':unittest.main()
