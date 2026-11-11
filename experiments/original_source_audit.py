"""Group-held-out prediction of dataset origin, never of disease."""
import os,sys,json,csv,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from prepare_original_xray import digest

FEATURES=['log_width','log_height','mean','std','p05','p95','rim_mean','center_mean','edge_energy']
SETS={'dimensions':[0,1],'appearance':list(range(2,9)),'combined':list(range(9))}

def unpad(image,width,height):
    size=image.shape[0]
    if image.shape!=(size,size) or min(width,height)<=0:raise ValueError('Expected square image and positive dimensions')
    scale=size/max(width,height);w=max(1,round(width*scale));h=max(1,round(height*scale));x=(size-w)//2;y=(size-h)//2
    return image[y:y+h,x:x+w]

def appearance(image):
    x=np.asarray(image,dtype=float)/255;h,w=x.shape;ry=max(1,round(.1*h));rx=max(1,round(.1*w));rim=np.ones_like(x,dtype=bool);rim[ry:h-ry,rx:w-rx]=False
    center=x[h//4:h-h//4,w//4:w-w//4];edge=(np.abs(np.diff(x,axis=0)).mean()+np.abs(np.diff(x,axis=1)).mean())/2
    return [x.mean(),x.std(),*np.quantile(x,[.05,.95]),x[rim].mean(),center.mean(),edge]

def main():
    out=ROOT/'reports/original_512/source_audit';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/original_source_audit.json').read_text());mp=ROOT/'reports/original_512/manifest.json';manifest=json.loads(mp.read_text());r=manifest['records'];cache=ROOT/'data/original_xray';meta=json.loads((cache/'cache_provenance.json').read_text())
    assert digest(mp)==meta['manifest_sha256'];assert digest(cache/'images_512.npy')==meta['images']['512']
    images=np.load(cache/'images_512.npy',mmap_mode='r');x=[]
    for i,v in enumerate(r):
        x.append([np.log(v['width']),np.log(v['height']),*appearance(unpad(images[i],v['width'],v['height']))])
        if i%1000==0:print('Features',i,len(r),flush=True)
    x=np.array(x);assert np.isfinite(x).all();clinical=np.array([v['label'] for v in r]);source=np.array([int(v['split']=='test') for v in r]);groups=np.array([v['group'] for v in r]);fold=np.full(len(r),-1);scores={};models={};results={};boot={}
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for label in [0,1]:
        ids=np.flatnonzero(clinical==label);splitter=StratifiedGroupKFold(5,shuffle=True,random_state=20261124)
        for f,(train,val) in enumerate(splitter.split(x[ids],source[ids],groups[ids])):
            tr,va=ids[train],ids[val];assert not set(groups[tr])&set(groups[va]);assert np.all(fold[va]==-1);fold[va]=f
            for name,cols in SETS.items():
                key=f'{label}__{name}';scores.setdefault(key,np.full(len(r),np.nan));scaler=StandardScaler().fit(x[tr][:,cols]);model=LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(scaler.transform(x[tr][:,cols]),source[tr]);scores[key][va]=model.decision_function(scaler.transform(x[va][:,cols]))
                models[f'{key}__fold{f}__mean']=scaler.mean_;models[f'{key}__fold{f}__scale']=scaler.scale_;models[f'{key}__fold{f}__coef']=model.coef_;models[f'{key}__fold{f}__intercept']=model.intercept_
        members={s:[ids[(groups[ids]==g)&(source[ids]==s)] for g in np.unique(groups[ids][source[ids]==s])] for s in [0,1]};rng=np.random.default_rng(20261125+label)
        for name in SETS:
            key=f'{label}__{name}';assert np.isfinite(scores[key][ids]).all();results[key]=dict(auroc=float(roc_auc_score(source[ids],scores[key][ids])),n=len(ids),groups=len(np.unique(groups[ids])));boot[key]=[]
        for _ in range(2000):
            chosen=np.concatenate([np.concatenate([members[s][j] for j in rng.integers(len(members[s]),size=len(members[s]))]) for s in [0,1]])
            for name in SETS:boot[f'{label}__{name}'].append(roc_auc_score(source[chosen],scores[f'{label}__{name}'][chosen]))
        for name in SETS:
            key=f'{label}__{name}';results[key]['ci']=np.quantile(boot[key],[.025,.975]).tolist();print(key,results[key],flush=True)
    assert np.all(fold>=0)
    distributions=[]
    for label in [0,1]:
        for src in [0,1]:
            ids=(clinical==label)&(source==src)
            for j,name in enumerate(FEATURES):
                q=np.quantile(x[ids,j],[.25,.5,.75]);distributions.append(dict(clinical_label=label,source=src,feature=name,n=int(ids.sum()),q25=q[0],median=q[1],q75=q[2]))
    p=np.load(ROOT/'reports/original_512/predictions.npz');np.testing.assert_array_equal(p['labels'],clinical);z=p['mobile_512_finetuned'];normal=(clinical==0)&(source==1);associations={}
    for j,name in enumerate(FEATURES):
        fp=normal&(z>=0);tn=normal&(z<0);associations[name]=dict(spearman_logit=float(spearmanr(x[normal,j],z[normal]).statistic),false_positive_median=float(np.median(x[fp,j])),true_negative_median=float(np.median(x[tn,j])))
    np.savez_compressed(out/'features_and_oof.npz',features=x,clinical_labels=clinical,source=source,groups=groups,fold=fold,**scores);np.savez_compressed(out/'fold_models.npz',**models);np.savez_compressed(out/'bootstrap.npz',**{k:np.array(v) for k,v in boot.items()})
    with (out/'distributions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=distributions[0]);w.writeheader();w.writerows(distributions)
    summary=dict(config=cfg,features=FEATURES,sets=SETS,results=results,primary=results['0__dimensions'],test_normal_associations=associations,provenance=dict(script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'configs/original_source_audit.json'),manifest_sha256=digest(mp),image_sha256=meta['images']['512'],diagnostic_predictions_sha256=digest(ROOT/'reports/original_512/predictions.npz')))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
