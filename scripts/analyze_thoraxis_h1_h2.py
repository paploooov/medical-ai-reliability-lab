"""Frozen H1/H2 analysis of saved predictions, protocol c44117a."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import t,beta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/original_512/thoraxis_30'
SEEDS=[17,42,83,123,2026];SPLITS=list(range(20261112,20261117))

def save(path,obj):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,np.bool_):return bool(x)
        return x
    path.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def csvout(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def risk_weights(confidence,mass):
    """Allocate precisely 80% sampled mass; equal fractions across boundary ties."""
    levels,inverse=np.unique(confidence,return_inverse=True)
    totals=np.bincount(inverse,weights=mass,minlength=len(levels))
    higher=np.cumsum(totals[::-1])[::-1]-totals
    need=.8*mass.sum()
    fraction=np.divide(np.clip(need-higher,0,totals),totals,out=np.zeros_like(totals),where=totals>0)
    return mass*fraction[inverse]

def ratio(a,b):return float(a/b) if b>0 else np.nan

def selection_summary(values):
    a=np.array(values);mean=float(a.mean());sd=float(a.std(ddof=1));half=t.ppf(.975,4)*sd/np.sqrt(5)
    return dict(values=a,mean=mean,sd=sd,range=[a.min(),a.max()],t95=[mean-half,mean+half],df=4)

def counts_metrics(d,mask,mass,rw,details=False):
    y=d['y'];pred=d['pred'];err=d['error'];acc=d['accept'];groups=d['group'];w=mass*mask;aw=w*acc;rankw=rw*mask
    tn=np.sum(w*((y==0)&(~pred)));fp=np.sum(w*((y==0)&pred));tp=np.sum(w*((y==1)&pred));fn=np.sum(w*((y==1)&(~pred)))
    n=w.sum();an=aw.sum();rn=rankw.sum();ae=np.dot(aw,err);re=np.dot(rankw,err)
    fp_rejected=np.sum(w*((y==0)&pred&~acc));tn_rejected=np.sum(w*((y==0)&~pred&~acc))
    atp=np.sum(aw*((y==1)&pred));afn=np.sum(aw*((y==1)&~pred))
    rates=dict(error=ratio(np.dot(w,err),n),specificity=ratio(tn,tn+fp),sensitivity=ratio(tp,tp+fn),
               acceptance=ratio(an,n),accepted_error=ratio(ae,an),R80=ratio(re,rn),R80_coverage=ratio(rn,n),
               fp_capture=ratio(fp_rejected,fp),accepted_sensitivity=ratio(atp,atp+afn),
               detected_and_accepted_sensitivity=ratio(atp,tp+fn))
    if not details:return rates
    accepted_group_ids=set(groups[(aw>0)].tolist());bad_groups=set(groups[(aw>0)&err].tolist());ng=len(accepted_group_ids);nb=len(bad_groups)
    upper=beta.ppf(.95,nb+1,ng-nb) if ng and nb<ng else (1. if ng else np.nan)
    return dict(**rates,n=int(n),groups=len(set(groups[mask].tolist())),normal=int(tn+fp),pneumonia=int(tp+fn),tn=int(tn),fp=int(fp),tp=int(tp),fn=int(fn),
        accepted_n=int(an),accepted_errors=int(ae),R80_mass=float(rn),R80_errors=float(re),
        rejected_false_positives=int(fp_rejected),rejected_true_negatives=int(tn_rejected),accepted_tp=int(atp),accepted_fn=int(afn),
        accepted_groups=ng,accepted_groups_with_error=nb,group_any_error_one_sided95_upper=upper,
        zero_error_warning='Group bound assumes independent filename groups; no patient/image risk bound. Fractional R80 has no binomial bound.')

def confidence_detail(conf):
    if not len(conf):return dict(n=0,quantiles=None,ecdf=[],histogram=[0]*10)
    values,counts=np.unique(conf,return_counts=True)
    return dict(n=len(conf),quantiles=dict(zip(['q25','median','q75','q90'],np.quantile(conf,[.25,.5,.75,.9]).tolist())),
                ecdf=[{'confidence':float(v),'cumulative_fraction':float(c)} for v,c in zip(values,np.cumsum(counts)/len(conf))],
                histogram=np.histogram(conf,bins=np.linspace(.5,1,11))[0].tolist())

def main():
    manifest=json.loads((OUT/'prediction_manifest.json').read_text());assert manifest['completed'] and len(manifest['models'])==30
    models={};metrics=[];confidence=[];model_rows=[]
    for entry in manifest['models']:
        path=OUT/entry['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        with np.load(path,allow_pickle=False) as z:
            test=z['partition']=='test';cal=z['partition']=='calibration';prob=expit(z['logit'][test]);y=z['label'][test]
            d=dict(y=y,pred=prob>=.5,confidence=np.maximum(prob,1-prob),group=z['group'][test],origin=z['origin'][test],image_id=z['image_id'][test])
            d['error']=d['pred']!=y;d['accept']=d['confidence']>=float(z['cutoff']);d['cutoff']=float(z['cutoff'])
            cy=z['label'][cal];cp=expit(z['logit'][cal]);d['calibration_specificity']=float(np.mean(cp[cy==0]<.5))
        name=entry['run_id'];rw=risk_weights(d['confidence'],np.ones(len(y)));assert np.isclose(rw.sum(),.8*618)
        d['R80']=float(np.dot(rw,d['error'])/rw.sum());d['strata']=[]
        for origin in ['all','train','test']:
            origin_mask=np.ones(len(y),dtype=bool) if origin=='all' else d['origin']==origin
            for cls in ['all','NORMAL','PNEUMONIA']:
                mask=origin_mask & (np.ones(len(y),dtype=bool) if cls=='all' else y==(0 if cls=='NORMAL' else 1))
                row=dict(run_id=name,origin=origin,cls=cls,**counts_metrics(d,mask,np.ones(len(y)),rw,True))
                metrics.append(row);d['strata'].append((origin,cls,mask))
            for label,mask in [('TN',(y==0)&~d['pred']),('FP',(y==0)&d['pred']),('TP',(y==1)&d['pred']),('FN',(y==1)&~d['pred'])]:
                confidence.append(dict(run_id=name,origin=origin,stratum=label,**confidence_detail(d['confidence'][origin_mask&mask])))
        overall=metrics[-9];model_rows.append(dict(run_id=name,R80=d['R80'],specificity=overall['specificity'],calibration_specificity=d['calibration_specificity'],
            specificity_gap=d['calibration_specificity']-overall['specificity'],cutoff=d['cutoff']))
        models[name]=d
    contrast=np.array([[models[f'official_seed{s}']['R80']-models[f'{split}_seed{s}']['R80'] for s in SEEDS] for split in SPLITS])
    csvout(OUT/'h1_contrasts.csv',[dict(split_seed=split,**{f'training_seed_{s}':float(contrast[i,j]) for j,s in enumerate(SEEDS)}) for i,split in enumerate(SPLITS)])
    csvout(OUT/'h1_model_metrics.csv',model_rows);save(OUT/'h2_metrics.json',metrics);save(OUT/'h2_confidence.json',dict(histogram_edges=np.linspace(.5,1,11),strata=confidence))
    groups=sorted(set().union(*(set(d['group']) for d in models.values())));gi={g:i for i,g in enumerate(groups)}
    for d in models.values():d['group_index']=np.array([gi[g] for g in d['group']])
    rng=np.random.default_rng(20261117);B=2000;primary=[];boot_metrics={};boot_cal_gap={k:[] for k in models}
    for b in range(B):
        gm=np.bincount(rng.integers(0,len(groups),size=len(groups)),minlength=len(groups));r={}
        for name,d in models.items():
            mass=gm[d['group_index']];rw=risk_weights(d['confidence'],mass);r[name]=ratio(np.dot(rw,d['error']),rw.sum())
            for origin,cls,mask in d['strata']:
                rates=counts_metrics(d,mask,mass,rw)
                for metric,value in rates.items():boot_metrics.setdefault((name,origin,cls,metric),[]).append(value)
                if origin=='all' and cls=='all':boot_cal_gap[name].append(d['calibration_specificity']-rates['specificity'])
        primary.append(np.mean([r[f'official_seed{s}']-r[f'{split}_seed{s}'] for split in SPLITS for s in SEEDS]))
        if b%250==0:print('Bootstrap',b,'/',B,flush=True)
    def interval(a):
        a=np.array(a);good=a[np.isfinite(a)]
        return dict(percentile95=np.quantile(good,[.025,.975]).tolist() if len(good) else None,defined=len(good),undefined=len(a)-len(good))
    save(OUT/'h2_bootstrap.json',dict(unit='Shared filename-group multiplicities; conditional on fixed models and thresholds; intervals conditional on defined replicates. Undefined: empty class/accepted set.',
        metrics=[dict(run_id=k[0],origin=k[1],cls=k[2],metric=k[3],**interval(v)) for k,v in boot_metrics.items()],
        specificity_gap=[dict(run_id=k,**interval(v)) for k,v in boot_cal_gap.items()]))
    np.savez_compressed(OUT/'h1_bootstrap_samples.npz',contrast=np.array(primary))
    seed=selection_summary(contrast.mean(axis=0));split=selection_summary(contrast.mean(axis=1));bootstrap=interval(primary)
    supported=bool(contrast.mean()>0 and seed['t95'][0]>0 and split['t95'][0]>0 and bootstrap['percentile95'][0]>0)
    state=json.loads((ROOT/'data/thoraxis_source_shift/training_batch/state.json').read_text());epochs={str(e):sum(v['result']['selected_epoch']==e for v in state['runs'].values()) for e in range(4)}
    result=dict(protocol='c44117a',primary_mean=float(contrast.mean()),contrast_table=contrast,seed=seed,split=split,group_bootstrap=dict(**bootstrap,replicates=B,seed=20261117,unique_groups=len(groups)),
        success_rule_met=supported,negative_contrasts=int((contrast<0).sum()),positive_contrasts=int((contrast>0).sum()),zero_contrasts=int((contrast==0).sum()),selected_epochs=epochs,
        interpretation='Descriptive consistency rule, not a causal test or joint confidence interval; five official models reused once per seed.')
    save(OUT/'h1_results.json',result);print(json.dumps(result,default=lambda x:x.tolist() if hasattr(x,'tolist') else x),flush=True)
if __name__=='__main__':main()
