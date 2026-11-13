"""Stage-three guards and fixed numerical definitions; protocol c44117a."""
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from prepare_original_xray import digest
from analyze_thoraxis_h1_h2 import save,risk_weights,ratio,csvout
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/original_512/thoraxis_30'
PLAN=ROOT/'configs/thoraxis_stage3_execution.json'

def guard():
    plan=json.loads(PLAN.read_text())
    remote=subprocess.check_output(['git','ls-remote','origin','main'],cwd=ROOT,text=True).strip().split()[0]
    for path,expected in plan['frozen_inputs'].items():
        assert digest(ROOT/path)==expected, f'Changed frozen input: {path}'
        content=subprocess.check_output(['git','show',f'{remote}:{path}'],cwd=ROOT)
        import hashlib
        assert hashlib.sha256(content).hexdigest()==expected,f'Unpublished code/input: {path}'
    assert subprocess.check_output(['git','show',f'{remote}:configs/thoraxis_stage3_execution.json'],cwd=ROOT)==PLAN.read_bytes()
    return dict(remote_main=remote,checked_at_utc=datetime.now(timezone.utc).isoformat())

def interval(values):
    a=np.asarray(values);valid=a[np.isfinite(a)]
    return dict(percentile95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,defined=len(valid),undefined=len(a)-len(valid))

def weighted_spearman(x,y,w):
    keep=np.asarray(w)>0;x=np.asarray(x)[keep];y=np.asarray(y)[keep];w=np.asarray(w)[keep]
    if w.sum()<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return np.nan
    def ranks(v):
        _,ix=np.unique(v,return_inverse=True);mass=np.bincount(ix,weights=w)
        return (np.cumsum(mass)-.5*mass+.5)[ix]
    a=ranks(x);b=ranks(y);a-=np.average(a,weights=w);b-=np.average(b,weights=w)
    den=np.sqrt(np.dot(w,a*a)*np.dot(w,b*b))
    return float(np.dot(w,a*b)/den) if den>0 else np.nan

def paired_metrics(labels,old,new,cut,mass):
    from scipy.special import expit
    result={};decisions=[]
    for kernel,z in [('lanczos',old),('bilinear',new)]:
        p=expit(z);pred=p>=.5;conf=np.maximum(p,1-p);accept=conf>=cut;err=pred!=labels;rw=risk_weights(conf,mass)
        result[kernel]=dict(R80=ratio(np.dot(rw,err),rw.sum()),specificity=ratio(np.sum(mass*((labels==0)&~pred)),np.sum(mass*(labels==0))),
            acceptance=ratio(np.dot(mass,accept),mass.sum()),accepted_error=ratio(np.dot(mass,accept&err),np.dot(mass,accept)),
            accepted_n=float(np.dot(mass,accept)),accepted_errors=float(np.dot(mass,accept&err)))
        decisions.append(pred)
    result['delta']={k:result['bilinear'][k]-result['lanczos'][k] for k in ['R80','specificity','acceptance','accepted_error']}
    result['decision_changes']=float(np.dot(mass,decisions[0]!=decisions[1]));result['decision_change_rate']=ratio(result['decision_changes'],mass.sum())
    result['normal_to_pneumonia']=float(np.dot(mass,~decisions[0]&decisions[1]));result['pneumonia_to_normal']=float(np.dot(mass,decisions[0]&~decisions[1]))
    return result
