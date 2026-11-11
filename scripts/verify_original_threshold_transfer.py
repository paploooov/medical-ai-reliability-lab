"""Independent order-statistic and full resampled-threshold verification."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    parent=ROOT/'reports/original_512';out=parent/'threshold_transfer';s=json.loads((out/'summary.json').read_text());p=np.load(parent/'predictions.npz');b=np.load(out/'bootstrap.npz');r=json.loads((parent/'manifest.json').read_text())['records'];y=p['labels'];z={a:p[a].astype(float) for a in s['config']['models'] if a in p};z['resnet50_224_probe']=np.load(parent/'large_encoder/head_and_predictions.npz')['logits'];groups={}
    for split in ['calibration','test']:
        ids=p[split+'_indices'];names=np.array([r[i]['group'] for i in ids]);groups[split]=[ids[names==g] for g in np.unique(names)]
    rng=np.random.default_rng(20261123)
    for draw in range(2000):
        ci=np.concatenate([groups['calibration'][j] for j in rng.integers(len(groups['calibration']),size=len(groups['calibration']))]);ti=np.concatenate([groups['test'][j] for j in rng.integers(len(groups['test']),size=len(groups['test']))])
        for a,v in z.items():
            for target in [.9,.95]:
                neg=v[ci][y[ci]==0];rank=len(neg)-int(np.floor((1-target)*len(neg)+1e-10))-1;cut=np.nextafter(np.partition(neg,rank)[rank],np.inf)
                assert (neg<cut).mean()>=target-1e-12
                observed=[cut,(v[ti][y[ti]==1]>=cut).mean(),(v[ti][y[ti]==0]<cut).mean()]
                np.testing.assert_allclose(observed,b[f'{a}__{target}'][draw],atol=1e-12)
    for k,v in s['results'].items():
        np.testing.assert_allclose(np.quantile(b[k.replace('|','__')][:,2],[.025,.975]),v['test_specificity_interval']['ci'],atol=1e-12)
    (out/'verification.json').write_text(json.dumps(dict(passed=True,resamples_verified=2000,model_target_combinations=6,thresholds_refit_and_verified=12000),indent=2)+'\n');print('Verified 12000 refitted thresholds and operating points')

if __name__=='__main__':main()
