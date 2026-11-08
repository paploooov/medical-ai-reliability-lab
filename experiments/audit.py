"""Read-only artifact and exact-image overlap audit; no training."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
from sklearn.metrics import accuracy_score,brier_score_loss,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=np.load(ROOT/'data/pneumoniamnist.npz',allow_pickle=False)
 hashes={s:[hashlib.sha256(im.tobytes()).hexdigest() for im in d[s+'_images']] for s in ['train','val','test']}
 overlaps={a+'_'+b:len(set(hashes[a])&set(hashes[b])) for a,b in [('train','val'),('train','test'),('val','test')]}
 duplicate_counts={s:len(h)-len(set(h)) for s,h in hashes.items()}
 check=[]
 for folder in ['reports/metrics','reports/cnn']:
  p=np.load(ROOT/folder/'predictions.npz',allow_pickle=False)
  np.testing.assert_array_equal(p['labels'],d['test_labels'].ravel())
  rows=list(csv.DictReader((ROOT/folder/'results.csv').open()))
  for r in rows:
   key=f"{r['model']}_{r['condition']}" if 'model' in r else f"{r['seed']}_{r['mode']}_{r['condition']}"
   prob=p[key];assert len(prob)==624 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
   y=p['labels'];expected={'accuracy':accuracy_score(y,prob>=.5),'brier':brier_score_loss(y,prob),'auc':roc_auc_score(y,prob),'high_confidence_error':np.mean(((prob>=.5)!=y)&(np.maximum(prob,1-prob)>=.9))}
   for metric,value in expected.items():np.testing.assert_allclose(float(r[metric]),value,atol=1e-12)
  check.append({'archive':folder,'rows_verified':len(rows)})
 cnn=np.load(ROOT/'reports/cnn/predictions.npz');invariance=[]
 for seed in [17,42,83]:
  for key in cnn.files:
   if key.startswith(f'{seed}_raw_'):
    other=key.replace('_raw_','_temperature_');np.testing.assert_array_equal(cnn[key]>=.5,cnn[other]>=.5)
    invariance.append(key)
 result={'exact_pixel_overlap_unique_hashes':overlaps,'within_split_duplicate_excess':duplicate_counts,'verified_results':check,'cnn_decision_invariance_conditions':len(invariance),'limitations':'Pixel equality is not patient identity; resized duplicates and related acquisitions may remain undetected.'}
 (ROOT/'reports/audit/artifact_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
