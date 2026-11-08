"""Independent grouped-confidence calculation against saved review results."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/selective_review'
def main():
 source=ROOT/'reports/resolution/predictions.npz';p=np.load(source);meta=json.loads((OUT/'provenance.json').read_text());assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['predictions_sha256']
 rows=list(csv.DictReader((OUT/'matched_coverage.csv').open()))
 for row in rows:
  prob=p[f"r{row['resolution']}_s{row['seed']}_{row['condition']}"];conf=np.maximum(prob,1-prob);wrong=(prob>=.5)!=p['labels'];target=float(row['coverage'])*len(prob);remaining=target;errors=0.
  for level in sorted(set(conf.tolist()),reverse=True):
   group=conf==level;n=int(group.sum());taken=min(remaining,n);errors+=taken*float(wrong[group].mean());remaining-=taken
   if remaining<=1e-10:break
  np.testing.assert_allclose(errors/target,float(row['risk']),atol=1e-12);np.testing.assert_allclose(target,float(row['expected_accepted']),atol=1e-10)
 # Class subtotals must reproduce weighted accepted/error totals.
 classes=list(csv.DictReader((OUT/'class_breakdown.csv').open()))
 for row in [r for r in rows if float(r['coverage'])==.8]:
  match=[c for c in classes if all(c[k]==row[k] for k in ['resolution','seed','condition'])];assert len(match)==2
  accepted=sum(float(c['expected_accepted']) for c in match);err=sum(float(c['expected_accepted_errors']) for c in match)
  np.testing.assert_allclose(accepted,float(row['expected_accepted']),atol=1e-10);np.testing.assert_allclose(err/accepted,float(row['risk']),atol=1e-12)
 result={'result_rows_recomputed_independently':len(rows),'class_breakdown_rows':len(classes),'source_hash_verified':True,'method':'Iterate unique confidence groups, allocate remaining acceptance budget, average group errors.'}
 (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
