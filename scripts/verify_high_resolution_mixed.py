"""Independent fixed-head, mixture-gradient and paired-risk checks."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_high_resolution_robustness import grouped_risk


def main():
    folder=ROOT/'reports/high_resolution/mixed_augmentation';s=json.loads((folder/'summary.json').read_text())
    heads=np.load(folder/'new_heads.npz');saved=np.load(folder/'predictions.npz');boot=np.load(folder/'bootstrap.npz');y=saved['labels']
    old=np.load(ROOT/'reports/high_resolution/severity/predictions.npz');ap=np.load(ROOT/'reports/high_resolution/augmentation/predictions.npz')
    probes=np.load(ROOT/'reports/high_resolution/predictions_and_probes.npz')
    np.testing.assert_array_equal(y,old['labels']);np.testing.assert_array_equal(saved['test_indices'],old['test_indices'])
    np.testing.assert_array_equal(heads['scaler_mean'],probes['native_224_scaler_mean']);np.testing.assert_array_equal(heads['scaler_scale'],probes['native_224_scaler_scale'])
    train={k:np.load(ROOT/path) for k,path in [('clean','data/high_resolution_features/native_224_train.npz'),('noise','data/high_resolution_augmentation/train_noise_20261116.npz'),('blur','data/high_resolution_mixed/train_blur4.npz')]}
    gradients={}
    for arm,selection in s['selection'].items():
        assert selection['C']==min(selection['candidates'],key=lambda r:r['selection_nll'])['C']
        assert selection['total_weight']==4682
        names=[k for k,w in selection['weights'].items() if w>0]
        for k in names:np.testing.assert_array_equal(train[k]['indices'],train['clean']['indices'])
        x=np.concatenate([train[k]['features'] for k in names]).astype(np.float32)
        x-=heads['scaler_mean'];x/=heads['scaler_scale']
        yy=np.tile(train['clean']['labels'],len(names));w=np.concatenate([np.full(len(train['clean']['labels']),selection['weights'][k]) for k in names])
        coef=heads[arm+'_coef'].ravel();residual=w*(expit(x@coef+heads[arm+'_intercept'][0])-yy)
        gradient=(x.T@residual+coef/selection['C'])/w.sum();ig=residual.sum()/w.sum()
        assert np.max(abs(gradient))<2e-4 and abs(ig)<2e-4
        gradients[arm]=float(max(np.max(abs(gradient)),abs(ig)))
        cal=np.load(ROOT/'data/high_resolution_features/native_224_calibration.npz')
        z=((cal['features']-heads['scaler_mean'])/heads['scaler_scale'])@coef+heads[arm+'_intercept'][0]
        np.testing.assert_allclose(z,heads[arm+'_calibration_logits'],atol=2e-5)
        pc=expit(heads[arm+'_calibration_logits'])
        np.testing.assert_allclose(np.quantile(np.maximum(pc,1-pc),.2),selection['cutoff'],atol=1e-12)
    conditions=[k.split('|')[1] for k in s['results'] if k.startswith('control|')]
    feature_hashes={}
    for condition in conditions:
        for arm,previous_arm in [('control','control'),('noise','augmentation')]:
            previous=old if previous_arm+'__'+condition in old.files else ap
            np.testing.assert_allclose(saved[arm+'__'+condition],previous[previous_arm+'__'+condition],atol=1e-10)
        if condition=='clean':path=ROOT/'data/high_resolution_features/native_224_test.npz'
        elif condition=='noise_0.1':path=ROOT/'data/high_resolution_augmentation/test_noise_42.npz'
        elif condition=='blur_8':path=ROOT/'data/high_resolution_augmentation/test_relative_blur.npz'
        elif condition in ['noise_7','noise_99','contrast']:path=ROOT/f'data/high_resolution_augmentation/test_{condition}.npz'
        else:path=ROOT/f'data/high_resolution_severity/{condition}.npz'
        d=np.load(path);feature_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        x=(d['features']-heads['scaler_mean'])/heads['scaler_scale']
        for arm in ['mixed','reduced_noise']:
            np.testing.assert_allclose(x@heads[arm+'_coef'].ravel()+heads[arm+'_intercept'][0],saved[arm+'__'+condition],atol=2e-5)
        for arm in ['control','noise','mixed','reduced_noise']:
            np.testing.assert_allclose(grouped_risk(y,expit(saved[arm+'__'+condition])),s['results'][arm+'|'+condition]['risk_at_80']['point'],atol=1e-12)
    rng=np.random.default_rng(s['config']['bootstrap_seed']);primary=[]
    for b in range(s['config']['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y));risks={}
        for arm in ['noise','mixed','reduced_noise']:
            risks[arm]=grouped_risk(y[ix],expit(saved[arm+'__blur_4'][ix]))
            np.testing.assert_allclose(risks[arm],boot[arm+'__blur_4'][b,0],atol=1e-12)
        primary.append(risks['mixed']-risks['noise'])
    np.testing.assert_allclose(np.quantile(primary,[.025,.975]),s['primary']['ci'],atol=1e-12)
    (folder/'verification.json').write_text(json.dumps(dict(passed=True,conditions=len(conditions),arms=4,
        mixture_max_abs_gradients=gradients,primary_bootstrap_draws_verified=2000,reduced_noise_blur_bootstrap_draws_verified=2000,
        previous_heads_unchanged=True,feature_sha256=feature_hashes,
        summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print('Both weighted fits, 52 point risks, preserved controls and 2000 blur draws for three arms verified.')


if __name__=='__main__':main()
