"""Independently reconstruct external metrics and all bootstrap draws."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from verify_high_resolution_robustness import grouped_risk
ROOT=Path(__file__).resolve().parents[1]

def metrics(y,z,cut):
    p=expit(z);pred=z>=0;keep=np.maximum(p,1-p)>=cut
    return np.array([roc_auc_score(y,p),np.where(y==1,np.logaddexp(0,-z),np.logaddexp(0,z)).mean(),np.mean(pred==y),grouped_risk(y,p),keep.mean(),np.mean(pred[keep]!=y[keep]) if keep.any() else np.nan])

def main():
    out=ROOT/'reports/external_chest224';s=json.loads((out/'summary.json').read_text());p=np.load(out/'predictions.npz');sample=np.load(out/'sample.npz');b=np.load(out/'bootstrap.npz')
    for k in ['indices','labels','strata']:np.testing.assert_array_equal(p[k],sample[k])
    assert len(np.unique(p['indices']))==len(p['indices'])
    np.testing.assert_array_equal(sample['original_labels'][:,6],p['labels'])
    assert (sample['original_labels'][p['strata']=='no_finding'].sum(1)==0).all()
    assert (sample['original_labels'][p['strata']=='other_abnormal'].sum(1)>0).all()
    prior=np.load(ROOT/'reports/original_512/predictions.npz');prob=expit(prior['mobile_224_probe'][prior['calibration_indices']]);assert s['threshold']==float(np.quantile(np.maximum(prob,1-prob),.2))
    for path,key in [('reports/original_512/probe_heads.npz','original_heads_sha256'),('reports/original_512/predictions.npz','internal_reference_predictions_sha256'),('data/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth','encoder_sha256'),('configs/external_chest224.json','protocol_sha256'),('experiments/external_chest224.py','script_sha256')]:assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==s['provenance'][key]
    names=['auroc','nll','accuracy','risk_at_80','coverage','retained_error'];rng=np.random.default_rng(20261128);y=p['labels'];z=p['logits']
    for control in ['no_finding','other_abnormal']:
        ids=np.flatnonzero((p['strata']=='pneumonia')|(p['strata']==control));pos=ids[y[ids]==1];neg=ids[y[ids]==0]
        for key,value in zip(names,metrics(y[ids],z[ids],s['threshold'])):
            expected=s['results'][control][key]['point']
            if expected is None:assert np.isnan(value)
            else:np.testing.assert_allclose(value,expected,atol=1e-12)
        for i in range(2000):
            ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            np.testing.assert_allclose(metrics(y[ix],z[ix],s['threshold']),b[control][i],atol=1e-12,equal_nan=True)
        for j,key in enumerate(names):
            values=b[control][:,j];values=values[np.isfinite(values)]
            if len(values):np.testing.assert_allclose(np.quantile(values,[.025,.975]),s['results'][control][key]['ci'],atol=1e-12)
    report=dict(passed=True,comparisons=2,bootstrap_draws_verified=4000,sampling_label_consistency=True,summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print('External metrics and 4000 draws verified')
if __name__=='__main__':main()
