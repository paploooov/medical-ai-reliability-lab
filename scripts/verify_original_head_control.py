"""Independently verify the optimizer-matched control and paired comparison."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from prepare_original_xray import digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    parent=ROOT/'reports/original_512';out=parent/'head_control';s=json.loads((out/'summary.json').read_text());p=np.load(out/'head_and_predictions.npz');f=np.load(ROOT/'data/original_xray/mobile_512_features.npz')['features'];base=np.load(parent/'predictions.npz');test=base['test_indices'];y=base['labels'][test];b=np.load(out/'bootstrap.npz');bb=np.load(parent/'bootstrap.npz')
    np.testing.assert_allclose((f@p['weight'].ravel()+p['bias'][0]),p['logits'],atol=2e-5);np.testing.assert_allclose(b['fine'],bb['mobile_512_finetuned'],atol=1e-12)
    assert s['selected_epoch']==min(s['history'],key=lambda v:v['selection_nll'])['epoch']
    m=json.loads((parent/'manifest.json').read_text());groups=np.array([m['records'][i]['group'] for i in test]);members=[np.flatnonzero(groups==g) for g in np.unique(groups)];rng=np.random.default_rng(20261122);z=p['logits'][test]
    for i in range(2000):
        ids=np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))]);np.testing.assert_allclose(roc_auc_score(y[ids],expit(z[ids])),b['head'][i,0],atol=1e-12)
    np.testing.assert_allclose(np.quantile(b['fine'][:,0]-b['head'][:,0],[.025,.975]),s['fine_minus_head']['auroc']['ci'],atol=1e-12)
    result=dict(passed=True,draws_verified=2000,reference_draws_identical=True,prediction_sha256=digest(out/'head_and_predictions.npz'));(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
