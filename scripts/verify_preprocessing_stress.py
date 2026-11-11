"""Independent confusion, stable-loss and primary paired-flip verification."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from verify_high_resolution_robustness import grouped_risk
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports/original_512/preprocessing';s=json.loads((out/'summary.json').read_text());p=np.load(out/'predictions.npz');b=np.load(out/'bootstrap.npz');old=np.load(out.parent/'predictions.npz');np.testing.assert_array_equal(p['baseline'],old['mobile_512_finetuned'][p['indices']].astype(float));test=np.flatnonzero(p['splits']=='test');y=p['labels'][test]
    for c,expected in s['results'].items():
        z=p[c][test];pred=z>=0;obs=dict(accuracy=(pred==y).mean(),sensitivity=pred[y==1].mean(),specificity=(~pred[y==0]).mean(),auroc=roc_auc_score(y,z),nll=np.where(y==1,np.logaddexp(0,-z),np.logaddexp(0,z)).mean(),risk_at_80=grouped_risk(y,expit(z)),flip_fraction=np.mean(pred!=(p['baseline'][test]>=0)))
        for k,v in obs.items():np.testing.assert_allclose(v,expected[k]['point'],atol=1e-12)
    groups=p['groups'];members=[test[groups[test]==g] for g in np.unique(groups[test])];rng=np.random.default_rng(20261126)
    for i in range(2000):
        ids=np.concatenate([members[j] for j in rng.integers(len(members),size=len(members))])
        for c in s['results']:
            yy=p['labels'][ids];zz=p[c][ids];pred=zz>=0
            values=[np.mean(pred!=(p['baseline'][ids]>=0)),np.mean(pred==yy),np.mean(pred[yy==1]),np.mean(~pred[yy==0]),np.where(yy==1,np.logaddexp(0,-zz),np.logaddexp(0,zz)).mean(),grouped_risk(yy,expit(zz)),roc_auc_score(yy,expit(zz))]
            np.testing.assert_allclose(values,b[c][i],atol=1e-12)
    names=['flip_fraction','accuracy','sensitivity','specificity','nll','risk_at_80','auroc']
    for c in s['results']:
        for j,key in enumerate(names):
            np.testing.assert_allclose(np.quantile(b[c][:,j],[.025,.975]),s['results'][c][key]['ci'],atol=1e-12)
            if c!='baseline' and j>0:
                np.testing.assert_allclose(np.quantile(b[c][:,j]-b['baseline'][:,j],[.025,.975]),s['contrasts'][c][key]['ci'],atol=1e-12)
    np.testing.assert_allclose(np.quantile(b['center_crop'][:,0],[.025,.975]),s['primary']['ci'],atol=1e-12)
    (out/'verification.json').write_text(json.dumps(dict(passed=True,conditions=6,all_metric_bootstrap_draws_verified=12000,paired_contrast_intervals_verified=True,baseline_unchanged=True),indent=2)+'\n');print('Preprocessing predictions and 12000 complete metric draws verified')

if __name__=='__main__':main()
