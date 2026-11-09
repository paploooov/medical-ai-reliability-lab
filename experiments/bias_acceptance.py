"""Do fixed calibration cutoffs accept bias-only predictions?"""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/budget'
def main():
 rows=[]
 for budget,folder in [(15,ROOT/'reports/resolution'),(45,OUT)]:
  run=json.loads((folder/'run.json').read_text());p=np.load(folder/'predictions.npz');d=np.load(OUT/f'noise_diagnostics_{budget}.npz');y=p['labels']
  for c in run['choices']:
   prefix=f"r{c['resolution']}_s{c['seed']}_noise";prob=p[prefix];zero=d[prefix+'_zero_hidden'];accepted=np.maximum(prob,1-prob)>=c['threshold'];wrong=(prob>=.5)!=y
   rows.append(dict(budget=budget,resolution=c['resolution'],seed=c['seed'],threshold=c['threshold'],zero_hidden=int(zero.sum()),zero_hidden_accepted=int((zero&accepted).sum()),zero_hidden_accepted_errors=int((zero&accepted&wrong).sum()),accepted_total=int(accepted.sum()),accepted_errors_total=int((accepted&wrong).sum())))
 with (OUT/'bias_acceptance.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 for r in rows:print(r)
if __name__=='__main__':main()
