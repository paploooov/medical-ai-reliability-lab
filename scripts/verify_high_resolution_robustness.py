"""Independent grouped-tie risk and bootstrap checks for frozen probe robustness."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score,roc_auc_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
VIEWS=['native_224','downsample28_upsample224']


def grouped_risk(y,p,coverage=.8):
    confidence=np.maximum(p,1-p)
    _,inverse,counts=np.unique(-confidence,return_inverse=True,return_counts=True)
    before=np.cumsum(counts)-counts
    fractions=np.clip((coverage*len(y)-before)/counts,0,1)
    weights=fractions[inverse]
    return float(np.dot(weights,(p>=.5)!=y)/(coverage*len(y)))


def main():
    folder=ROOT/'reports/high_resolution/robustness'
    summary=json.loads((folder/'summary.json').read_text())
    baseline=json.loads((ROOT/'reports/high_resolution/summary.json').read_text())
    prior=np.load(ROOT/'reports/high_resolution/predictions_and_probes.npz')
    saved=np.load(folder/'predictions.npz');boot=np.load(folder/'bootstrap.npz');y=saved['labels']
    assert hashlib.sha256((ROOT/'reports/high_resolution/predictions_and_probes.npz').read_bytes()).hexdigest()==summary['provenance']['probes_sha256']
    np.testing.assert_array_equal(y,prior['test_labels'])
    np.testing.assert_array_equal(saved['test_indices'],prior['test_indices'])
    np.testing.assert_array_equal(saved[VIEWS[0]+'__noise_input'],saved[VIEWS[0]+'__noise_source'])
    for view in VIEWS:
        np.testing.assert_array_equal(saved[view+'__clean'],prior[view+'_test_logits'])
        for condition in summary['config']['conditions']:
            key=view+'|'+condition;z=saved[view+'__'+condition];p=expit(z)
            expected=summary['results'][key];keep=np.maximum(p,1-p)>=baseline['results'][view]['threshold']
            observed=dict(accuracy=accuracy_score(y,p>=.5),auroc=roc_auc_score(y,p),
                          brier=brier_score_loss(y,p),risk_at_80=grouped_risk(y,p),
                          coverage=keep.mean(),nll=np.mean(np.where(y==1,np.logaddexp(0,-z),np.logaddexp(0,z))))
            if keep.any():observed['retained_error']=1-accuracy_score(y[keep],p[keep]>=.5)
            else:assert expected['retained_error']['point'] is None
            for metric,value in observed.items():np.testing.assert_allclose(value,expected[metric]['point'],atol=1e-12)
    def contrast(ix):
        risks={(v,c):grouped_risk(y[ix],expit(saved[v+'__'+c][ix])) for v in VIEWS for c in ['clean','noise_input']}
        return (risks[VIEWS[0],'noise_input']-risks[VIEWS[0],'clean'])-(risks[VIEWS[1],'noise_input']-risks[VIEWS[1],'clean'])
    np.testing.assert_allclose(contrast(np.arange(len(y))),summary['primary']['point'],atol=1e-12)
    rng=np.random.default_rng(summary['config']['bootstrap_seed'])
    for b in range(summary['config']['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y))
        np.testing.assert_allclose(contrast(ix),boot['primary'][b],atol=1e-12)
    np.testing.assert_allclose(np.quantile(boot['primary'],[.025,.975]),summary['primary']['ci'],atol=1e-12)
    checks=dict(passed=True,conditions=5,views=2,primary_bootstrap_replicates_verified=2000,
                clean_logits_byte_values_unchanged=True,native_noise_placement_identity=True,
                summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('All ten condition/view metrics and all 2,000 primary bootstrap draws independently verified.')


if __name__=='__main__':main()
