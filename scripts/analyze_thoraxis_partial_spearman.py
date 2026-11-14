"""Post hoc exploratory size-adjusted Spearman; saved predictions only."""
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.stats import rankdata
from prepare_original_xray import digest
from analyze_thoraxis_h1_h2 import save
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/original_512/thoraxis_30';PLAN=ROOT/'configs/thoraxis_posthoc_partial_spearman.json'

def estimate(x,y,width,height,w):
    w=np.asarray(w,dtype=float);keep=w>0;w=w[keep]
    values=[np.asarray(v)[keep] for v in (x,y,width,height)]
    if w.sum()<3:return dict(raw=np.nan,partial=np.nan,design_rank=0,reason='fewer_than_three_sampled_cases')
    def standardized_ranks(v):
        _,ix=np.unique(v,return_inverse=True);mass=np.bincount(ix,weights=w);r=(np.cumsum(mass)-.5*mass+.5)[ix]
        r-=np.average(r,weights=w);sd=np.sqrt(np.average(r*r,weights=w));return r/sd if sd>0 else np.zeros_like(r)
    rx,ry,rw,rh=[standardized_ranks(v) for v in values]
    def corr(a,b):
        a=a-np.average(a,weights=w);b=b-np.average(b,weights=w);aa=np.average(a*a,weights=w);bb=np.average(b*b,weights=w)
        return float(np.average(a*b,weights=w)/np.sqrt(aa*bb)) if min(aa,bb)>1e-12 else np.nan
    design=np.column_stack([np.ones(len(w)),rw,rh]);sw=np.sqrt(w);a=design*sw[:,None]
    coeff,_,rank,_=np.linalg.lstsq(a,np.column_stack([rx,ry])*sw[:,None],rcond=None)
    raw=corr(rx,ry)
    if w.sum()<=rank+1:return dict(raw=raw,partial=np.nan,design_rank=int(rank),reason='too_few_cases_for_design_rank')
    residual=np.column_stack([rx,ry])-design@coeff;partial=corr(residual[:,0],residual[:,1])
    return dict(raw=raw,partial=partial,design_rank=int(rank),reason=None if np.isfinite(partial) else 'constant_or_numerically_zero_residual')

def interval(a):
    a=np.asarray(a);good=a[np.isfinite(a)]
    return dict(percentile95=np.quantile(good,[.025,.975]).tolist() if len(good) else None,defined=len(good),undefined=len(a)-len(good))

def main():
    plan=json.loads(PLAN.read_text());remote=subprocess.check_output(['git','ls-remote','origin','main'],cwd=ROOT,text=True).split()[0]
    assert subprocess.check_output(['git','show',remote+':configs/thoraxis_posthoc_partial_spearman.json'],cwd=ROOT)==PLAN.read_bytes()
    assert subprocess.check_output(['git','show',remote+':scripts/analyze_thoraxis_partial_spearman.py'],cwd=ROOT)==Path(__file__).read_bytes()
    for p,h in plan['inputs_sha256'].items():assert digest(ROOT/p)==h,p
    started=dict(utc=datetime.now(timezone.utc).isoformat(),remote_main=remote,plan_sha256=digest(PLAN),script_sha256=digest(__file__))
    with (OUT/'partial_spearman_started.json').open('x') as f:json.dump(started,f,indent=2)
    original={r['name']:r for r in json.loads((ROOT/'reports/original_512/manifest.json').read_text())['records']}
    oof=dict(np.load(OUT/'source_oof.npz',allow_pickle=False));index={n:i for i,n in enumerate(oof['image_id'])};assert len(index)==len(oof['image_id'])
    predictions={};manifest=json.loads((OUT/'prediction_manifest.json').read_text());assert manifest['completed'] and len(manifest['models'])==30
    for e in manifest['models']:
        assert digest(OUT/e['file'])==e['sha256']
        with np.load(OUT/e['file'],allow_pickle=False) as z:
            m=z['partition']=='test';predictions[e['run_id']]={k:z[k][m] for k in ['image_id','group','origin','label','logit']}
    unique=sorted(set().union(*(set(z['group']) for z in predictions.values())));gi={g:i for i,g in enumerate(unique)}
    rng=np.random.default_rng(20261117);B=2000;n=len(unique);draws=np.array([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(B)],dtype=np.int16)
    rows=[];boots={};max_raw_difference=0.;max_independent_difference=0.;independent_n=0
    previous={(r['run_id'],r['origin']):r['rho'] for r in json.loads((OUT/'source_logit_correlations.json').read_text())['rows'] if r['source_outer_fold']=='all'}
    for name,z in predictions.items():
        m=z['label']==0;ids=z['image_id'][m];groups=z['group'][m];origin=z['origin'][m];y=z['logit'][m]
        assert len(ids)==231 and all(i in index for i in ids)
        ix=np.array([index[i] for i in ids]);x=oof['score'][ix];assert np.array_equal(groups,oof['group'][ix])
        widths=np.array([original[i]['width'] for i in ids]);heights=np.array([original[i]['height'] for i in ids]);assert np.all(widths>0) and np.all(heights>0)
        width=np.log(widths);height=np.log(heights);mass=draws[:,[gi[g] for g in groups]]
        for src in ['all','train','test']:
            mask=np.ones(len(ids),dtype=bool) if src=='all' else origin==src
            vv=[v[mask] for v in [x,y,width,height]];point=estimate(*vv,np.ones(int(mask.sum())))
            if previous[name,src] is not None:max_raw_difference=max(max_raw_difference,abs(point['raw']-previous[name,src]))
            if np.isfinite(point['partial']):
                # Independent complete-case identity: inverse correlation matrix of ranks.
                ranks=np.column_stack([rankdata(v,method='average') for v in vv]);cor=np.corrcoef(ranks,rowvar=False)
                if np.linalg.matrix_rank(cor)==4:
                    precision=np.linalg.inv(cor);independent=-precision[0,1]/np.sqrt(precision[0,0]*precision[1,1]);max_independent_difference=max(max_independent_difference,abs(independent-point['partial']));independent_n+=1
            raw=[];partial=[];reasons={}
            for w in mass:
                v=estimate(*vv,w[mask]);raw.append(v['raw']);partial.append(v['partial'])
                if v['reason']:reasons[v['reason']]=reasons.get(v['reason'],0)+1
            raw=np.array(raw);partial=np.array(partial);boots[name,src]=dict(raw=raw,partial=partial)
            rows.append(dict(run_id=name,origin=src,n=int(mask.sum()),groups=len(set(groups[mask])),**point,delta=point['partial']-point['raw'],raw_bootstrap=interval(raw),partial_bootstrap=interval(partial),delta_bootstrap=interval(partial-raw),undefined_reasons=reasons))
        print('Partial Spearman',name,flush=True)
    assert max_raw_difference<1e-12 and max_independent_difference<1e-10
    summaries=[]
    for arm in ['official','mixed']:
        for src in ['all','train','test']:
            rr=[r for r in rows if r['run_id'].startswith('official')==(arm=='official') and r['origin']==src]
            result=dict(arm=arm,origin=src,models=len(rr))
            for key in ['raw','partial']:
                point=np.array([r[key] for r in rr]);samples=np.array([boots[r['run_id'],src][key] for r in rr]);means=np.mean(samples,axis=0)
                result[key]=dict(mean=float(point.mean()),range=[float(point.min()),float(point.max())],defined_models=int(np.isfinite(point).sum()),group_bootstrap=interval(means),
                    seed_means=[float(np.mean([r[key] for r in rr if r['run_id'].endswith('_seed'+str(s))])) for s in [17,42,83,123,2026]],
                    split_means=[float(np.mean([r[key] for r in rr if r['run_id'].startswith(str(s)+'_')])) for s in range(20261112,20261117)] if arm=='mixed' else [])
            deltas=np.array([boots[r['run_id'],src]['partial']-boots[r['run_id'],src]['raw'] for r in rr]);result['delta']=dict(mean=float(np.mean([r['delta'] for r in rr])),group_bootstrap=interval(np.mean(deltas,axis=0)))
            summaries.append(result)
    result=dict(exploratory_posthoc=True,decision_after_stage3_results=True,in_protocol_c44117a=False,gate=started,rows=rows,summaries=summaries,bootstrap_replicates=B,bootstrap_seed=20261117,group_universe=n,limits=plan['limits'],checks=dict(raw_spearman_max_difference=max_raw_difference,independent_inverse_matrix_cases=independent_n,independent_partial_max_difference=max_independent_difference))
    save(OUT/'partial_spearman_dimensions.json',result)
    print(json.dumps(result['checks']),flush=True)
if __name__=='__main__':main()
