"""Check fold isolation, weighted optima, OOF reconstruction and primary bootstrap."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]

def auc(y,z):
    n1=y.sum();n0=len(y)-n1;r=rankdata(z)
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def main():
    out=ROOT/'reports/original_512/source_audit';s=json.loads((out/'summary.json').read_text());p=np.load(out/'features_and_oof.npz');models=np.load(out/'fold_models.npz');boot=np.load(out/'bootstrap.npz');x=p['features'];y=p['source'];clinical=p['clinical_labels'];fold=p['fold'];groups=p['groups'];gradients=[]
    records=json.loads((out.parent/'manifest.json').read_text())['records'];np.testing.assert_allclose(x[:,:2],np.log([[r['width'],r['height']] for r in records]),atol=1e-12)
    images=np.load(ROOT/'data/original_xray/images_512.npy',mmap_mode='r')
    sampled=np.unique(np.linspace(0,len(records)-1,24,dtype=int))
    for i in sampled:
        r=records[i];w=round(512*r['width']/max(r['width'],r['height']));h=round(512*r['height']/max(r['width'],r['height']));left=(512-w)//2;top=(512-h)//2;a=np.asarray(images[i,top:top+h,left:left+w],dtype=float)/255
        rh=max(1,round(.1*h));rw=max(1,round(.1*w));border=np.concatenate([a[:rh].ravel(),a[-rh:].ravel(),a[rh:-rh,:rw].ravel(),a[rh:-rh,-rw:].ravel()]);middle=a[h//4:h-h//4,w//4:w-w//4]
        observed=[a.sum()/a.size,np.sqrt(np.mean((a-a.mean())**2)),*np.percentile(a,[5,95]),border.mean(),middle.mean(),(np.abs(a[1:]-a[:-1]).mean()+np.abs(a[:,1:]-a[:,:-1]).mean())/2]
        np.testing.assert_allclose(observed,x[i,2:],atol=1e-12)
    for label in [0,1]:
        ids=np.flatnonzero(clinical==label)
        for f in range(5):
            tr=ids[fold[ids]!=f];va=ids[fold[ids]==f];assert not set(groups[tr])&set(groups[va])
            for name,cols in s['sets'].items():
                key=f'{label}__{name}';prefix=f'{key}__fold{f}__';mean=models[prefix+'mean'];scale=models[prefix+'scale'];coef=models[prefix+'coef'].ravel();intercept=models[prefix+'intercept'][0]
                np.testing.assert_allclose(x[tr][:,cols].mean(0),mean,atol=1e-12);np.testing.assert_allclose(x[tr][:,cols].std(0),scale,atol=1e-12)
                tx=(x[tr][:,cols]-mean)/scale;vx=(x[va][:,cols]-mean)/scale;np.testing.assert_allclose(vx@coef+intercept,p[key][va],atol=1e-12)
                weights=len(tr)/(2*np.bincount(y[tr])[y[tr]]);residual=weights*(expit(tx@coef+intercept)-y[tr]);gradient=np.r_[(tx.T@residual+coef)/len(tr),residual.mean()];gradients.append(float(np.abs(gradient).max()))
                assert gradients[-1]<5e-4,gradients[-1]
        for name in s['sets']:np.testing.assert_allclose(auc(y[ids],p[f'{label}__{name}'][ids]),s['results'][f'{label}__{name}']['auroc'],atol=1e-12)
    ids=np.flatnonzero(clinical==0);members={v:[ids[(groups[ids]==g)&(y[ids]==v)] for g in np.unique(groups[ids][y[ids]==v])] for v in [0,1]};rng=np.random.default_rng(20261125)
    for i in range(2000):
        chosen=np.concatenate([np.concatenate([members[v][j] for j in rng.integers(len(members[v]),size=len(members[v]))]) for v in [0,1]])
        np.testing.assert_allclose(auc(y[chosen],p['0__dimensions'][chosen]),boot['0__dimensions'][i],atol=1e-12)
    np.testing.assert_allclose(np.quantile(boot['0__dimensions'],[.025,.975]),s['primary']['ci'],atol=1e-12)
    result=dict(passed=True,fold_models_verified=30,appearance_cases_independently_verified=len(sampled),max_weighted_gradient=max(gradients),primary_bootstrap_replicates_verified=2000,source_auc_implementations='sklearn versus independent rank statistic',group_overlap_between_training_and_scoring=0)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
