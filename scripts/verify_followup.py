"""Independent numerical checks for the November 10 follow-up analyses."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parents[1]


def independent_ece(y,p):
    confidence=np.maximum(p,1-p);correct=(p>=.5)==y
    total=0.0
    for i in range(15):
        mask=(confidence>i/15)&(confidence<=(i+1)/15)
        if mask.any():total+=mask.sum()*abs(correct[mask].mean()-confidence[mask].mean())
    return total/len(y)


def main():
    folder=ROOT/'reports/temperature_scaling'
    summary=json.loads((folder/'summary.json').read_text());stored=np.load(folder/'logits.npz')
    y=stored['test_labels'];cy=stored['calibration_labels'];checked=0
    for model in summary['models']:
        prefix=f"{model['arm']}_s{model['seed']}";t=model['temperature'];zc=stored[prefix+'_calibration_logits']
        # First-order optimality of the convex inverse-temperature problem.
        assert abs(np.mean(zc*(expit(zc/t)-cy)))<1e-6
        assert model['calibration_nll_after']<=model['calibration_nll_before']+1e-10
        for condition in summary['config']['conditions']:
            z=stored[f'{prefix}_{condition}_logits']
            for temperature in [1,t]:
                p=expit(z/temperature)
                expected_nll=np.mean(np.where(y==1,np.logaddexp(0,-z/temperature),np.logaddexp(0,z/temperature)))
                stage='before' if temperature==1 else 'after'
                with (folder/'results.csv').open() as f:
                    row=next(r for r in csv.DictReader(f) if r['arm']==model['arm'] and int(r['seed'])==model['seed']
                             and r['condition']==condition and r['stage']==stage)
                np.testing.assert_allclose(float(row['nll']),expected_nll,atol=1e-12)
                np.testing.assert_allclose(float(row['ece']),independent_ece(y,p),atol=1e-12)
                checked+=1
    source=ROOT/'reports/external/predictions.npz';p=np.load(source)
    with (ROOT/'configs/external_rsna_sample.csv').open() as f:sample=[r for r in csv.DictReader(f) if r['arm']=='primary']
    y=p['labels__external'];view=np.array([r['view'] for r in sample])
    summary=json.loads((ROOT/'reports/external/view_analysis/summary.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==summary['provenance']['predictions_sha256']
    with (ROOT/'reports/external/results.csv').open() as f:results=list(csv.DictReader(f))
    for key,value in summary['results'].items():
        arm,condition,v=key.split('|');mask=view==v;yy=y[mask]
        weights=np.where(yy==1,.5/(yy==1).sum(),.5/(yy==0).sum())
        accuracy=[];coverage=[];risk=[]
        for seed in [17,42,83]:
            pp=p[f'external__{arm}__s{seed}__{condition}'][mask]
            threshold=float(next(r['threshold'] for r in results if r['dataset']=='external' and r['arm']==arm
                                 and int(r['seed'])==seed and r['condition']==condition))
            keep=np.maximum(pp,1-pp)>=threshold
            accuracy.append(accuracy_score(yy,pp>=.5,sample_weight=weights))
            coverage.append(weights[keep].sum())
            risk.append(1-accuracy_score(yy[keep],pp[keep]>=.5,sample_weight=weights[keep]))
        for metric,values in [('balanced_accuracy',accuracy),('standardized_coverage',coverage),('standardized_retained_error',risk)]:
            np.testing.assert_allclose(value[metric]['point'],np.mean(values),atol=1e-12)
            checked+=1
    # Replay the first 16 joint bootstrap draws independently from saved logits.
    sens=json.loads((folder/'sensitivity/summary.json').read_text())
    draws=np.load(folder/'sensitivity/bootstrap.npz')
    models=json.loads((folder/'summary.json').read_text())['models']
    rng=np.random.default_rng(sens['config']['seed'])
    for b in range(16):
        ci=rng.integers(0,len(cy),len(cy));ti=rng.integers(0,len(stored['test_labels']),len(stored['test_labels']))
        for i,m in enumerate(models):
            prefix=f"{m['arm']}_s{m['seed']}";t=draws['temperatures'][b,i]
            zc=stored[prefix+'_calibration_logits'][ci];z=stored[prefix+'_clean_logits'][ti]
            yy=stored['test_labels'][ti]
            derivative=np.mean(zc*(expit(zc/t)-cy[ci]))
            if .05<t<20:assert abs(derivative)<1e-6
            elif t==.05:assert derivative<=1e-6
            else:assert derivative>=-1e-6
            before=np.mean(np.where(yy==1,np.logaddexp(0,-z),np.logaddexp(0,z)))
            after=np.mean(np.where(yy==1,np.logaddexp(0,-z/t),np.logaddexp(0,z/t)))
            np.testing.assert_allclose(draws['nll_differences'][b,i],after-before,atol=1e-12)
    print('96 calibration-refit replicates independently checked.')
    print(f'{checked} independent point-metric checks and six temperature optimality checks passed.')


if __name__=='__main__':main()
