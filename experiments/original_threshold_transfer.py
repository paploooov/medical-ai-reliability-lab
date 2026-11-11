"""Re-fit calibration decision thresholds inside a two-sample group bootstrap."""
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from original_512 import ROOT
from high_resolution_robustness import interval
from prepare_original_xray import digest

def threshold_for_specificity(negative_logits,target):
    scores=np.sort(np.asarray(negative_logits,dtype=np.float64))[::-1]
    if not len(scores) or not 0<target<=1:raise ValueError('Need negatives and target in (0,1]')
    allowed=int(np.floor((1-target)*len(scores)+1e-10))
    return float(np.nextafter(scores[allowed],np.inf))

def operating_point(y,z,threshold):
    positive=np.asarray(z,dtype=np.float64)>=threshold
    return np.array([positive[y==1].mean(),(~positive[y==0]).mean()])

def main():
    parent=ROOT/'reports/original_512';out=parent/'threshold_transfer';out.mkdir(exist_ok=False);cfg=json.loads((ROOT/'configs/original_512_threshold_transfer.json').read_text());p=np.load(parent/'predictions.npz');r=json.loads((parent/'manifest.json').read_text())['records'];y=p['labels'];cal=p['calibration_indices'];test=p['test_indices'];z={a:p[a].astype(float) for a in cfg['models'] if a in p};z['resnet50_224_probe']=np.load(parent/'large_encoder/head_and_predictions.npz')['logits'].astype(float)
    groups={};split_metrics={}
    for split in ['train','selection','calibration','test']:
        ids=p[split+'_indices'];names=np.array([r[i]['group'] for i in ids]);groups[split]=[ids[names==g] for g in np.unique(names)];split_metrics[split]={a:dict(auroc=roc_auc_score(y[ids],v[ids]),nll=float(np.mean(np.logaddexp(0,v[ids])-y[ids]*v[ids])),sensitivity=float(operating_point(y[ids],v[ids],0)[0]),specificity=float(operating_point(y[ids],v[ids],0)[1])) for a,v in z.items()}
    points={};draws={}
    for a,v in z.items():
        for target in cfg['specificity_targets']:
            key=f'{a}|{target}';cut=threshold_for_specificity(v[cal][y[cal]==0],target);cm=operating_point(y[cal],v[cal],cut);tm=operating_point(y[test],v[test],cut)
            assert cm[1]>=target-1e-12;points[key]=dict(logit_threshold=cut,probability_threshold=float(expit(cut)),calibration_sensitivity=float(cm[0]),calibration_specificity=float(cm[1]),test_sensitivity=float(tm[0]),test_specificity=float(tm[1]));draws[key]=[]
    rng=np.random.default_rng(20261123)
    for _ in range(2000):
        ci=np.concatenate([groups['calibration'][j] for j in rng.integers(0,len(groups['calibration']),len(groups['calibration']))]);ti=np.concatenate([groups['test'][j] for j in rng.integers(0,len(groups['test']),len(groups['test']))])
        for a,v in z.items():
            for target in cfg['specificity_targets']:
                cut=threshold_for_specificity(v[ci][y[ci]==0],target);draws[f'{a}|{target}'].append([cut,*operating_point(y[ti],v[ti],cut)])
    draws={k:np.array(v) for k,v in draws.items()};results={}
    for k,point in points.items():
        target=float(k.split('|')[1]);b=draws[k];results[k]=dict(**point,test_sensitivity_interval=interval(point['test_sensitivity'],b[:,1]),test_specificity_interval=interval(point['test_specificity'],b[:,2]),bootstrap_fraction_below_specificity_target=float((b[:,2]<target).mean()))
    summary=dict(config=cfg,split_metrics=split_metrics,results=results,provenance=dict(script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'configs/original_512_threshold_transfer.json'),predictions_sha256=digest(parent/'predictions.npz')))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez_compressed(out/'bootstrap.npz',**{k.replace('|','__'):v for k,v in draws.items()});print(json.dumps(results,indent=2))

if __name__=='__main__':main()
