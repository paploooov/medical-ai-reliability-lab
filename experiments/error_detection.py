"""Exploratory follow-up: can confidence distinguish errors from correct predictions?"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import json, csv, hashlib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def detection(y,p,threshold):
    wrong=(p>=.5)!=y; confidence=np.maximum(p,1-p); flag=confidence<threshold
    keep=~flag
    return dict(error_detection_auroc=float(roc_auc_score(wrong,1-confidence)) if 0<wrong.sum()<len(y) else None,error_detection_ap=float(average_precision_score(wrong,1-confidence)) if wrong.any() else None,error_prevalence=float(wrong.mean()),coverage=float(keep.mean()),retained_error_rate=float(wrong[keep].mean()) if keep.any() else None,error_recall=float(flag[wrong].mean()) if wrong.any() else None,flag_precision=float(wrong[flag].mean()) if flag.any() else None,flagged=int(flag.sum()),errors=int(wrong.sum()))

def risk_curve(y,p):
    conf=np.maximum(p,1-p);wrong=((p>=.5)!=y).astype(float)
    thresholds=np.unique(conf)[::-1]
    coverage=[];risk=[]
    for t in thresholds:
        kept=conf>=t;coverage.append(kept.mean());risk.append(wrong[kept].mean())
    return coverage,risk

def main():
    data=np.load(ROOT/'data/pneumoniamnist.npz');pred=np.load(ROOT/'reports/metrics/predictions.npz')
    x=data['val_images'].reshape(-1,784)/255
    rows=[]; details={};fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,name in zip(axes,['logistic','mlp']):
        path=ROOT/f'models/{name}.npz'; w=np.load(path)
        logits=x@w['coef'].T+w['intercept'] if name=='logistic' else np.maximum(x@w['w0']+w['b0'],0)@w['w1']+w['b1']
        vp=expit(logits.ravel()); threshold=float(np.quantile(np.maximum(vp,1-vp),.2))
        details[name]=dict(validation_confidence_threshold=threshold,target_validation_coverage=.8,actual_validation_coverage=float((np.maximum(vp,1-vp)>=threshold).mean()),weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for key in pred.files:
            if not key.startswith(name+'_'):continue
            condition=key[len(name)+1:];p=pred[key]
            row=dict(model=name,condition=condition,threshold=threshold,**detection(pred['labels'],p,threshold));rows.append(row)
            if condition in ['clean','blur_2','contrast_0.25']:
                coverage,risk=risk_curve(pred['labels'],p);ax.plot(coverage,risk,label=condition)
        ax.set(title=name,xlabel='Coverage: fraction of predictions retained',ylabel='Error rate among retained predictions',xlim=(0,1),ylim=(0,.5));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Does rejecting uncertain predictions reduce errors?');fig.tight_layout();fig.savefig(ROOT/'reports/figures/error_detection.png',dpi=160)
    with (ROOT/'reports/metrics/error_detection.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    details['analysis_type']='Exploratory follow-up after pilot test inspection; no model retraining or test-selected rejection threshold.'
    (ROOT/'reports/metrics/error_detection.json').write_text(json.dumps(details,indent=2))
    for r in rows:
        if r['condition'] in ['clean','blur_2','contrast_0.25']:print(json.dumps(r))
if __name__=='__main__':main()
