"""Post-hoc 2x2 diagnostic: training augmentation versus selected C values."""
from pathlib import Path
import argparse,hashlib,json,warnings
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from high_resolution_augmentation import standardize,weighted_training
from high_resolution_robustness import interval
from external_eval import risk_at
ROOT=Path(__file__).resolve().parents[1]


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/high_resolution_regularization_sensitivity.json').read_text())
    probes=np.load(ROOT/'reports/high_resolution/predictions_and_probes.npz')
    original=np.load(ROOT/'reports/high_resolution/augmentation/predictions.npz')
    clean=np.load(ROOT/'data/high_resolution_features/native_224_train.npz')
    noisy=np.load(ROOT/'data/high_resolution_augmentation/train_noise_20261116.npz')
    np.testing.assert_array_equal(clean['indices'],noisy['indices'])
    np.testing.assert_array_equal(clean['labels'],noisy['labels'])
    x=standardize(clean['features'],probes);n=standardize(noisy['features'],probes);ytrain=clean['labels']
    ax,ay,aw=weighted_training(x,n,ytrain)
    test={c:np.load(ROOT/('data/high_resolution_features/native_224_test.npz' if c=='clean'
                         else 'data/high_resolution_augmentation/test_noise_42.npz')) for c in cfg['conditions']}
    y=original['labels'];saved={};rows=[];heads={}
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for c in cfg['C_values']:
        for arm in ['control','augmentation']:
            model=LogisticRegression(C=c,solver='lbfgs',max_iter=3000)
            if arm=='control':model.fit(x,ytrain)
            else:model.fit(ax,ay,sample_weight=aw)
            heads[f'{arm}_C{c}_coef']=model.coef_;heads[f'{arm}_C{c}_intercept']=model.intercept_
            for condition in cfg['conditions']:
                np.testing.assert_array_equal(test[condition]['labels'],y)
                z=model.decision_function(standardize(test[condition]['features'],probes))
                saved[arm,c,condition]=z
                if (arm,c) in [('control',.01),('augmentation',.1)]:
                    np.testing.assert_allclose(z,original[arm+'__'+condition],atol=1e-9)
                rows.append(dict(arm=arm,C=c,condition=condition,risk_at_80=risk_at(y,expit(z),.8)))
    rng=np.random.default_rng(cfg['bootstrap_seed']);values={(c,condition):[] for c in cfg['C_values'] for condition in cfg['conditions']}
    for _ in range(cfg['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y))
        for c,condition in values:
            values[c,condition].append(risk_at(y[ix],expit(saved['augmentation',c,condition][ix]),.8)
                                      -risk_at(y[ix],expit(saved['control',c,condition][ix]),.8))
    summary=dict(config=cfg,results=rows,contrasts={})
    for (c,condition),boot in values.items():
        point=risk_at(y,expit(saved['augmentation',c,condition]),.8)-risk_at(y,expit(saved['control',c,condition]),.8)
        summary['contrasts'][f'C{c}|{condition}']=interval(point,np.array(boot))
    summary['provenance']=dict(parent_summary_sha256=hashlib.sha256((ROOT/'reports/high_resolution/augmentation/summary.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'predictions_and_heads.npz',labels=y,**heads,**{f'{a}_C{c}_{condition}':z for (a,c,condition),z in saved.items()})
    np.savez_compressed(out/'bootstrap.npz',**{f'C{c}_{condition}':v for (c,condition),v in values.items()})
    print(json.dumps(summary['contrasts'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/high_resolution/augmentation/regularization_sensitivity')
    main(p.parse_args().output)
