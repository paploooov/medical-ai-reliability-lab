"""Prespecified source, score/logit, and paired interpolation analyses."""
import json,sys
import numpy as np
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from scipy.special import expit
from thoraxis_stage3_common import ROOT,OUT,guard,save,digest,interval,weighted_spearman,paired_metrics,csvout
SEEDS=[17,42,83,123,2026];SPLITS=list(range(20261112,20261117));B=2000

def load_oof():
    meta=json.loads((OUT/'source_oof_manifest.json').read_text());assert digest(OUT/meta['file'])==meta['sha256']
    return dict(np.load(OUT/meta['file'],allow_pickle=False))

def bootstrap_mass(groups,rng):
    unique,ix=np.unique(groups,return_inverse=True);n=len(unique)
    return np.array([np.bincount(rng.integers(n,size=n),minlength=n)[ix] for _ in range(B)])

def source():
    guard();z=load_oof();y=z['label'];p=z['score'];values=[]
    masses=bootstrap_mass(z['group'],np.random.default_rng(20261117))
    for w in masses:
        values.append(roc_auc_score(y,p,sample_weight=w) if np.dot(w,y)>0 and np.dot(w,1-y)>0 else np.nan)
    result=dict(auroc=roc_auc_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p>=.5),counts=np.bincount(y).tolist(),groups=len(set(z['group'])),
        group_bootstrap=interval(values),replicates=B,bootstrap_seed=20261117,folds=[])
    for fold in range(5):
        m=z['fold']==fold;result['folds'].append(dict(fold=fold,n=int(m.sum()),groups=len(set(z['group'][m])),counts=np.bincount(y[m],minlength=2).tolist(),auroc=roc_auc_score(y[m],p[m]),balanced_accuracy=balanced_accuracy_score(y[m],p[m]>=.5)))
    save(OUT/'source_cnn_results.json',result);np.savez_compressed(OUT/'source_auc_bootstrap.npz',auroc=values);print(json.dumps(result),flush=True)

def medical():
    manifest=json.loads((OUT/'prediction_manifest.json').read_text());assert manifest['completed'];out={}
    for entry in manifest['models']:
        assert digest(OUT/entry['file'])==entry['sha256']
        with np.load(OUT/entry['file'],allow_pickle=False) as z:
            m=z['partition']=='test';out[entry['run_id']]={k:z[k][m] for k in ['image_id','group','origin','label','logit']};out[entry['run_id']]['cutoff']=float(z['cutoff'])
    return out

def joint_draws(models):
    universe=sorted(set().union(*(set(z['group']) for z in models.values())));ix={g:i for i,g in enumerate(universe)};rng=np.random.default_rng(20261117);n=len(universe)
    multiplicities=np.array([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(B)],dtype=np.int16)
    return multiplicities,ix

def correlation():
    guard();oof=load_oof();lookup={n:i for i,n in enumerate(oof['image_id'])};assert len(lookup)==len(oof['image_id'])
    models=medical();draws,gi=joint_draws(models);rows=[];series=[];origins=[]
    for name,z in models.items():
        normal=z['label']==0;ids=z['image_id'][normal];assert all(n in lookup for n in ids),'Missing OOF pairs; stop'
        ix=np.array([lookup[n] for n in ids]);score=oof['score'][ix];logit=z['logit'][normal];groups=z['group'][normal];origin=z['origin'][normal];fold=oof['fold'][ix]
        assert np.array_equal(oof['group'][ix],groups)
        masses=draws[:,[gi[g] for g in groups]]
        for src in ['all','train','test']:
            sm=np.ones(len(ids),dtype=bool) if src=='all' else origin==src
            v=score[sm]
            origins.append(dict(run_id=name,origin=src,n=int(sm.sum()),groups=len(set(groups[sm])),score_mean=float(v.mean()) if len(v) else None,score_median=float(np.median(v)) if len(v) else None,score_q25=float(np.quantile(v,.25)) if len(v) else None,score_q75=float(np.quantile(v,.75)) if len(v) else None))
            for outer in ['all',0,1,2,3,4]:
                mask=sm & (np.ones(len(ids),dtype=bool) if outer=='all' else fold==outer);x=score[mask];y=logit[mask]
                rho=weighted_spearman(x,y,np.ones(len(x)))
                values=[weighted_spearman(x,y,w[mask]) for w in masses]
                rows.append(dict(run_id=name,origin=src,source_outer_fold=outer,n=int(mask.sum()),groups=len(set(groups[mask])),rho=rho,bootstrap=interval(values)))
                if outer=='all':series.append((name,src,rho,np.array(values)))
        print('Correlation',name,flush=True)
    summaries=[]
    # Average only defined correlations; count them explicitly, never multiply case numbers.
    for src in ['all','train','test']:
        for arm in ['official','mixed']:
            chosen=[(n,r,v) for n,s,r,v in series if s==src and n.startswith('official')==(arm=='official')]
            finite=[r for n,r,v in chosen if np.isfinite(r)]
            group_means=[]
            for b in range(B):
                vv=[v[b] for n,r,v in chosen if np.isfinite(v[b])];group_means.append(np.mean(vv) if vv else np.nan)
            seed_means=[];split_means=[]
            for seed in SEEDS:
                vv=[r for n,r,v in chosen if n.endswith('_seed'+str(seed)) and np.isfinite(r)];seed_means.append(float(np.mean(vv)) if vv else np.nan)
            if arm=='mixed':
                for split in SPLITS:
                    vv=[r for n,r,v in chosen if n.startswith(str(split)+'_') and np.isfinite(r)];split_means.append(float(np.mean(vv)) if vv else np.nan)
            summaries.append(dict(arm=arm,origin=src,defined_models=len(finite),mean=float(np.mean(finite)) if finite else np.nan,range=[min(finite),max(finite)] if finite else None,seed_means=seed_means,split_means=split_means,group_bootstrap_of_mean=interval(group_means)))
    save(OUT/'source_logit_correlations.json',dict(rows=rows,summaries=summaries,paired_missing=0,replicates=B,bootstrap_seed=20261117,undefined_rule='Fewer than three sampled pairs or constant rank variable; intervals conditional on defined draws; no p-value tests.'))
    save(OUT/'source_scores_by_origin.json',origins)

def h3():
    guard();models=medical();manifest=json.loads((OUT/'bilinear_prediction_manifest.json').read_text());assert manifest['completed'];draws,gi=joint_draws(models);rows=[];boot={}
    for entry in manifest['models']:
        name=entry['run_id'];z=models[name];assert digest(OUT/entry['file'])==entry['sha256']
        with np.load(OUT/entry['file'],allow_pickle=False) as b:
            for key in ['image_id','label','group','origin']:assert np.array_equal(z[key],b[key]),key
            assert float(b['cutoff'])==z['cutoff'];new=b['logit'].copy()
        mass=np.ones(len(new));point=paired_metrics(z['label'],z['logit'],new,z['cutoff'],mass)
        for kernel,zz in [('lanczos',z['logit']),('bilinear',new)]:
            prob=expit(zz);accepted=np.maximum(prob,1-prob)>=z['cutoff'];error=(prob>=.5)!=z['label'];n=len(set(z['group'][accepted]));k=len(set(z['group'][accepted&error]))
            point[kernel]['accepted_groups']=n;point[kernel]['accepted_groups_with_error']=k
            point[kernel]['zero_error_group95_upper']=1-.05**(1/n) if n and k==0 else None
        samples={k:[] for k in ['R80','specificity','acceptance','accepted_error','decision_change_rate']}
        for counts in draws[:,[gi[g] for g in z['group']]]:
            val=paired_metrics(z['label'],z['logit'],new,z['cutoff'],counts)
            for k in samples:samples[k].append(val['decision_change_rate'] if k=='decision_change_rate' else val['delta'][k])
        boot[name]=samples;rows.append(dict(run_id=name,**point,group_bootstrap={k:interval(v) for k,v in samples.items()}));print('H3',name,flush=True)
    summaries=[]
    for arm in ['official','mixed']:
        rr=[r for r in rows if r['run_id'].startswith('official')==(arm=='official')]
        for metric in ['R80','specificity','acceptance','accepted_error','decision_change_rate']:
            value=lambda r:r['decision_change_rate'] if metric=='decision_change_rate' else r['delta'][metric]
            vv=[value(r) for r in rr];seed_means=[np.mean([value(r) for r in rr if r['run_id'].endswith('_seed'+str(s))]) for s in SEEDS]
            split_means=[np.mean([value(r) for r in rr if r['run_id'].startswith(str(s)+'_')]) for s in SPLITS] if arm=='mixed' else []
            samples=np.array([boot[r['run_id']][metric] for r in rr]);mean_samples=np.mean(samples,axis=0)
            summaries.append(dict(arm=arm,metric=metric,mean=np.mean(vv),range=[min(vv),max(vv)],seed_means=seed_means,seed_sd=np.std(seed_means,ddof=1),split_means=split_means,split_sd=np.std(split_means,ddof=1) if split_means else None,group_bootstrap=interval(mean_samples)))
    save(OUT/'h3_results.json',dict(direction='bilinear minus Lanczos',exploratory=True,rows=rows,summaries=summaries,replicates=B,bootstrap_seed=20261117,zero_error_note='No zero-risk inference from zero errors or [0,0] bootstrap; group bound assumes independent filename groups, not image/patient risk.'))
if __name__=='__main__':{'source':source,'correlation':correlation,'h3':h3}[sys.argv[1]]()
