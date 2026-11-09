"""Scalar cross-check of saved sensitivity replicates and data provenance."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/calibration'
def scalar_statistics(probabilities,labels,thresholds,indices):
 risk=[];coverage=[];counts=[]
 for model in range(6):
  r=[];c=[];ns=[]
  for condition in range(2):
   p=probabilities[model,condition,indices];keep=np.maximum(p,1-p)>=thresholds[model];n=int(keep.sum());ns.append(n);c.append(n/len(indices));r.append(float(((p[keep]>=.5)!=labels[indices][keep]).mean()) if n else float('nan'))
  risk.append(r);coverage.append(c);counts.append(ns)
 r=np.array(risk).reshape(2,3,2).mean(1);c=np.array(coverage).reshape(2,3,2).mean(1)
 return np.r_[r.ravel(),c.ravel(),(r[1,1]-r[1,0])-(r[0,1]-r[0,0]),r[1,0]-r[0,0]],np.array(counts)
def main():
 provenance=json.loads((OUT/'provenance.json').read_text());assert hashlib.sha256((ROOT/'reports/augmentation/comparison_predictions.npz').read_bytes()).hexdigest()==provenance['source_predictions_sha256'];assert hashlib.sha256((ROOT/'configs/resolution_manifest.json').read_bytes()).hexdigest()==provenance['manifest_sha256']
 p=dict(np.load(OUT/'predictions.npz'));b=dict(np.load(OUT/'bootstrap.npz'));s=dict(np.load(OUT/'subsets.npz'));cc=np.maximum(p['calibration_probabilities'],1-p['calibration_probabilities']);baseline=np.quantile(cc,.2,axis=1)
 assert b['test_indices'].min()>=0 and b['test_indices'].max()<len(p['test_labels']);assert b['calibration_indices'].min()>=0 and b['calibration_indices'].max()<cc.shape[1]
 for i in range(len(b['joint'])):
  cut=np.quantile(cc[:,b['calibration_indices'][i]],.2,axis=1);np.testing.assert_allclose(cut,b['thresholds'][i],atol=1e-12)
  stats,counts=scalar_statistics(p['test_probabilities'],p['test_labels'],cut,b['test_indices'][i]);np.testing.assert_allclose(stats,b['joint'][i],atol=1e-12,equal_nan=True);np.testing.assert_array_equal(counts,b['accepted_counts'][i])
  fixed,_=scalar_statistics(p['test_probabilities'],p['test_labels'],baseline,b['test_indices'][i]);np.testing.assert_allclose(fixed,b['fixed'][i],atol=1e-12,equal_nan=True)
 subsets=0
 for size in [32,64,128,262]:
  for i,selection in enumerate(s[f'indices_n{size}']):
   assert len(np.unique(selection))==size and selection.min()>=0 and selection.max()<cc.shape[1]
   cut=np.quantile(cc[:,selection],.2,axis=1);np.testing.assert_allclose(cut,s[f'thresholds_n{size}'][i],atol=1e-12)
   stats,_=scalar_statistics(p['test_probabilities'],p['test_labels'],cut,np.arange(len(p['test_labels'])));np.testing.assert_allclose(stats,s[f'statistics_n{size}'][i],atol=1e-12,equal_nan=True);subsets+=1
 result={'paired_bootstrap_replicates_recomputed':len(b['joint']),'fixed_and_joint_checked':True,'subset_replicates_recomputed':subsets,'accepted_counts_checked':True,'source_hashes_verified':True}
 (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
