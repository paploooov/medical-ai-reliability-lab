"""Reconstruct the additional larger-encoder comparison independently."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from prepare_original_xray import digest
from verify_high_resolution_robustness import grouped_risk
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports/original_512/large_encoder';s=json.loads((out/'summary.json').read_text());p=np.load(out/'head_and_predictions.npz');f=np.load(ROOT/'data/original_xray/resnet50_224_features.npz')['features'];np.testing.assert_allclose(f@p['weight']+p['bias'],p['logits'],atol=2e-5)
    candidates=s['selection']['candidates'];assert s['selection']['C']==min(candidates,key=lambda r:r['selection_nll'])['C']
    test=p['test_indices'];y=p['labels'][test];z=p['logits'][test];np.testing.assert_allclose(roc_auc_score(y,expit(z)),s['metrics']['auroc'],atol=1e-12);np.testing.assert_allclose(grouped_risk(y,expit(z)),s['metrics']['risk_at_80'],atol=1e-12)
    b=np.load(out/'bootstrap.npz');base=np.load(out.parent/'bootstrap.npz');np.testing.assert_allclose(b['resnet18'],base['resnet18_224_probe'],atol=1e-12)
    m=json.loads((out.parent/'manifest.json').read_text());groups=np.array([m['records'][i]['group'] for i in test]);members=[np.flatnonzero(groups==g) for g in np.unique(groups)];rng=np.random.default_rng(20261122)
    for j in range(2000):
        ids=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))]);np.testing.assert_allclose(roc_auc_score(y[ids],expit(z[ids])),b['resnet50'][j,0],atol=1e-12)
    np.testing.assert_allclose(np.quantile(b['resnet50'][:,0]-b['resnet18'][:,0],[.025,.975]),s['contrast']['auroc']['ci'],atol=1e-12)
    result=dict(passed=True,draws_verified=2000,reference_draws_identical=True,prediction_sha256=digest(out/'head_and_predictions.npz'));(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
