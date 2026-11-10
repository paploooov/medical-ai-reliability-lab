"""Independent severity endpoints, fixed-head reconstruction and joint-band audit."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,accuracy_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_high_resolution_robustness import grouped_risk


def main():
    folder=ROOT/'reports/high_resolution/severity';s=json.loads((folder/'summary.json').read_text())
    saved=np.load(folder/'predictions.npz');boot=np.load(folder/'bootstrap.npz');y=saved['labels']
    probes=np.load(ROOT/'reports/high_resolution/predictions_and_probes.npz')
    head=np.load(ROOT/'reports/high_resolution/augmentation/augmented_head.npz')
    old=np.load(ROOT/'reports/high_resolution/augmentation/predictions.npz')
    parent=json.loads((ROOT/'reports/high_resolution/augmentation/summary.json').read_text())
    base=json.loads((ROOT/'reports/high_resolution/summary.json').read_text())
    np.testing.assert_array_equal(y,old['labels']);np.testing.assert_array_equal(saved['test_indices'],old['test_indices'])
    assert hashlib.sha256((ROOT/'reports/high_resolution/augmentation/augmented_head.npz').read_bytes()).hexdigest()==s['provenance']['head_sha256']
    anchors={'clean':('data/high_resolution_features/native_224_test.npz','clean'),
             'noise_0.1':('data/high_resolution_augmentation/test_noise_42.npz','noise_42'),
             'blur_8':('data/high_resolution_augmentation/test_relative_blur.npz','relative_blur')}
    grid=s['joint_grid_band']['grid'];points=[];cache_hashes={}
    for name in grid:
        path=ROOT/anchors[name][0] if name in anchors else ROOT/f'data/high_resolution_severity/{name}.npz'
        features=np.load(path);np.testing.assert_array_equal(features['indices'],saved['test_indices'])
        cache_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        xx=(features['features']-head['scaler_mean'])/head['scaler_scale'];risks=[]
        for arm in ['control','augmentation']:
            coef=probes['native_224_coefficient'] if arm=='control' else head['coefficient']
            intercept=probes['native_224_intercept'] if arm=='control' else head['intercept']
            z=saved[arm+'__'+name];np.testing.assert_allclose(xx@coef.ravel()+intercept[0],z,atol=2e-5)
            if name in anchors:np.testing.assert_allclose(z,old[arm+'__'+anchors[name][1]],atol=1e-10)
            row=s['results'][arm+'|'+name];p=expit(z)
            cutoff=base['results']['native_224']['threshold'] if arm=='control' else parent['selection']['augmented_cutoff']
            keep=np.maximum(p,1-p)>=cutoff
            metrics=dict(auroc=roc_auc_score(y,p),accuracy=accuracy_score(y,p>=.5),brier=brier_score_loss(y,p),
                         coverage=keep.mean(),risk_at_80=grouped_risk(y,p),
                         nll=np.mean(np.where(y==1,np.logaddexp(0,-z),np.logaddexp(0,z))))
            if keep.any():metrics['retained_error']=1-accuracy_score(y[keep],p[keep]>=.5)
            else:assert row['retained_error']['point'] is None
            for key,value in metrics.items():np.testing.assert_allclose(value,row[key]['point'],atol=1e-12)
            risks.append(metrics['risk_at_80'])
        points.append(risks[1]-risks[0])
    rng=np.random.default_rng(s['config']['bootstrap_seed']);primary=[]
    for b in range(s['config']['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y))
        for name in (grid if b<25 else ['noise_0.2']):
            value=grouped_risk(y[ix],expit(saved['augmentation__'+name][ix]))-grouped_risk(y[ix],expit(saved['control__'+name][ix]))
            np.testing.assert_allclose(value,boot['risk_difference_grid'][b,grid.index(name)],atol=1e-12)
            if name=='noise_0.2':primary.append(value)
    np.testing.assert_allclose(np.quantile(primary,[.025,.975]),s['primary']['ci'],atol=1e-12)
    maxima=np.sort(np.max(np.abs(boot['risk_difference_grid']-np.array(points)[None,:]),axis=1))
    position=.95*(len(maxima)-1);lower=int(position)
    radius=maxima[lower]+(position-lower)*(maxima[lower+1]-maxima[lower])
    np.testing.assert_allclose(radius,s['joint_grid_band']['radius'],atol=1e-12)
    np.testing.assert_allclose(np.array(points)-radius,s['joint_grid_band']['low'],atol=1e-12)
    np.testing.assert_allclose(np.array(points)+radius,s['joint_grid_band']['high'],atol=1e-12)
    result=dict(passed=True,grid_points=len(grid),fixed_head_reconstructions=2*len(grid),
                primary_bootstrap_draws=2000,full_grid_bootstrap_draws=25,joint_band_checked=True,
                feature_sha256=cache_hashes,summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('20 fixed-head reconstructions, all point metrics, 2000 primary draws and joint grid band verified.')


if __name__=='__main__':main()
