"""Verify saved checkpoints, provenance and results without fitting."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import torch
from scipy.special import expit
from resolution_study import ResolutionCNN,logits,tensor
from train import metrics
from error_detection import detection
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/resolution'
def main():
 torch.set_num_threads(2)
 run=json.loads((OUT/'run.json').read_text());manifest_path=ROOT/'configs/resolution_manifest.json';assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==run['manifest_sha256'];manifest=json.loads(manifest_path.read_text());p=np.load(OUT/'predictions.npz');idx=manifest['indices']
 np.testing.assert_array_equal(p['test_indices'],idx['test'])
 rows=list(csv.DictReader((OUT/'results.csv').open()))
 for row in rows:
  prob=p[f"r{row['resolution']}_s{row['seed']}_{row['condition']}"];assert np.isfinite(prob).all()
  values={**metrics(p['labels'],prob),**detection(p['labels'],prob,float(row['threshold']))}
  for key,value in values.items():
   if value is None:assert row[key]==''
   else:np.testing.assert_allclose(float(row[key]),value,atol=1e-12)
 for c in run['choices']:
  r=c['resolution'];path=ROOT/f"models/resolution/r{r}_s{c['seed']}.pt";assert hashlib.sha256(path.read_bytes()).hexdigest()==c['weights_sha256']
  model=ResolutionCNN();model.load_state_dict(torch.load(path,weights_only=True));model.eval();data=np.load(ROOT/'data'/('pneumoniamnist.npz' if r==28 else 'pneumoniamnist_64.npz'))
  np.testing.assert_array_equal(data['test_labels'][idx['test']].ravel(),p['labels'])
  prob=expit(logits(model,tensor(data['test_images'][idx['test']])));np.testing.assert_allclose(prob,p[f"r{r}_s{c['seed']}_clean"],atol=1e-7)
  cp=expit(logits(model,tensor(data['val_images'][idx['calibration']])));np.testing.assert_allclose(np.quantile(np.maximum(cp,1-cp),.2),c['threshold'],atol=1e-12)
 result={'verified_result_rows':len(rows),'verified_saved_models':len(run['choices']),'verified_calibration_thresholds':len(run['choices']),'test_representatives':len(p['labels']),'manifest_hash_verified':True}
 (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
