"""Independent saved-head, clean-tradeoff and paired risk verification."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score,roc_auc_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_high_resolution_robustness import grouped_risk


def main():
    folder=ROOT/'reports/high_resolution/augmentation'
    summary=json.loads((folder/'summary.json').read_text());selection=summary['selection']
    head=np.load(folder/'augmented_head.npz');saved=np.load(folder/'predictions.npz')
    original=np.load(ROOT/'reports/high_resolution/predictions_and_probes.npz')
    old=np.load(ROOT/'reports/high_resolution/robustness/predictions.npz')
    baseline=json.loads((ROOT/'reports/high_resolution/summary.json').read_text())
    boot=np.load(folder/'bootstrap.npz');y=saved['labels']
    assert hashlib.sha256((ROOT/'reports/high_resolution/predictions_and_probes.npz').read_bytes()).hexdigest()==summary['provenance']['original_probes_sha256']
    np.testing.assert_array_equal(y,original['test_labels'])
    np.testing.assert_array_equal(head['scaler_mean'],original['native_224_scaler_mean'])
    np.testing.assert_array_equal(head['scaler_scale'],original['native_224_scaler_scale'])
    assert selection['total_training_weight']==4682
    assert selection['actual_training_rows']==9364
    assert selection['chosen_C']==min(selection['candidates'],key=lambda row:row['selection_nll'])['C']
    for file,digest in summary['feature_sha256'].items():
        path=ROOT/'data/high_resolution_augmentation'/file
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    # Check the weighted logistic objective's gradient, independently of sklearn.fit.
    clean_train=np.load(ROOT/'data/high_resolution_features/native_224_train.npz')
    noisy_train=np.load(ROOT/'data/high_resolution_augmentation/train_noise_20261116.npz')
    np.testing.assert_array_equal(clean_train['labels'],noisy_train['labels'])
    np.testing.assert_array_equal(clean_train['indices'],noisy_train['indices'])
    xx=np.concatenate([clean_train['features'],noisy_train['features']]).astype(np.float32)
    xx-=head['scaler_mean'];xx/=head['scaler_scale']
    yy=np.tile(clean_train['labels'],2);coef=head['coefficient'].ravel()
    prob=expit(xx@coef+head['intercept'][0]);residual=.5*(prob-yy)
    gradient=(xx.T@residual+coef/selection['chosen_C'])/selection['total_training_weight']
    intercept_gradient=residual.sum()/selection['total_training_weight']
    assert np.max(np.abs(gradient))<2e-4, 'Weighted training gradient exceeds solver tolerance'
    assert abs(intercept_gradient)<2e-4
    # Reconstruct the chosen head on selection and calibration with float64 arithmetic.
    for split in ['selection','calibration']:
        features=np.load(ROOT/f'data/high_resolution_features/native_224_{split}.npz')
        z=((features['features']-head['scaler_mean'])/head['scaler_scale'])@head['coefficient'].ravel()+head['intercept'][0]
        if split=='calibration':np.testing.assert_allclose(z,head['calibration_logits'],atol=2e-5)
        else:
            yy=features['labels'];loss=np.mean(np.where(yy==1,np.logaddexp(0,-z),np.logaddexp(0,z)))
            np.testing.assert_allclose(loss,selection['selection_nll'],atol=2e-6)
    pcal=expit(head['calibration_logits'])
    np.testing.assert_allclose(np.quantile(np.maximum(pcal,1-pcal),.2),selection['augmented_cutoff'],atol=1e-12)
    for condition in summary['config']['conditions']:
        original_condition='noise_input' if condition=='noise_42' else condition
        if 'native_224__'+original_condition in old.files:
            np.testing.assert_allclose(saved['control__'+condition],old['native_224__'+original_condition],atol=1e-10)
        path=(ROOT/'data/high_resolution_features/native_224_test.npz' if condition=='clean'
              else ROOT/f'data/high_resolution_augmentation/test_{condition}.npz')
        features=np.load(path)
        np.testing.assert_array_equal(features['labels'],y)
        np.testing.assert_array_equal(features['indices'],saved['test_indices'])
        z=((features['features']-head['scaler_mean'])/head['scaler_scale'])@head['coefficient'].ravel()+head['intercept'][0]
        np.testing.assert_allclose(z,saved['augmentation__'+condition],atol=2e-5)
        for arm in ['control','augmentation']:
            z=saved[arm+'__'+condition];p=expit(z)
            row=summary['results'][arm+'|'+condition]
            cutoff=baseline['results']['native_224']['threshold'] if arm=='control' else selection['augmented_cutoff']
            keep=np.maximum(p,1-p)>=cutoff
            metrics=dict(accuracy=accuracy_score(y,p>=.5),auroc=roc_auc_score(y,p),brier=brier_score_loss(y,p),
                         nll=np.mean(np.where(y==1,np.logaddexp(0,-z),np.logaddexp(0,z))),
                         coverage=keep.mean(),risk_at_80=grouped_risk(y,p))
            if keep.any():metrics['retained_error']=1-accuracy_score(y[keep],p[keep]>=.5)
            else:assert row['retained_error']['point'] is None
            for key,value in metrics.items():np.testing.assert_allclose(value,row[key]['point'],atol=1e-12)
    rng=np.random.default_rng(summary['config']['bootstrap_seed']);difference=[]
    for b in range(summary['config']['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y))
        # Independently replay both primary noise effect and the clean tradeoff.
        for condition in ['noise_42','clean']:
            risks=[]
            for arm in ['control','augmentation']:
                risk=grouped_risk(y[ix],expit(saved[arm+'__'+condition][ix]))
                np.testing.assert_allclose(risk,boot[arm+'__'+condition][b,-1],atol=1e-12)
                risks.append(risk)
            if condition=='noise_42':difference.append(risks[1]-risks[0])
    np.testing.assert_allclose(np.quantile(difference,[.025,.975]),summary['primary']['ci'],atol=1e-12)
    # Verify every matched-C diagnostic cell and a paired sample of its bootstrap draws.
    matched_folder=folder/'regularization_sensitivity'
    matched=json.loads((matched_folder/'summary.json').read_text())
    md=np.load(matched_folder/'predictions_and_heads.npz');mb=np.load(matched_folder/'bootstrap.npz')
    np.testing.assert_array_equal(md['labels'],y)
    for row in matched['results']:
        key=f"{row['arm']}_C{row['C']}_{row['condition']}"
        np.testing.assert_allclose(grouped_risk(y,expit(md[key])),row['risk_at_80'],atol=1e-12)
    rng=np.random.default_rng(matched['config']['bootstrap_seed'])
    for b in range(25):
        ix=rng.integers(0,len(y),len(y))
        for c in matched['config']['C_values']:
            for condition in matched['config']['conditions']:
                value=grouped_risk(y[ix],expit(md[f'augmentation_C{c}_{condition}'][ix]))-grouped_risk(y[ix],expit(md[f'control_C{c}_{condition}'][ix]))
                np.testing.assert_allclose(value,mb[f'C{c}_{condition}'][b],atol=1e-12)
    for key,value in matched['contrasts'].items():
        np.testing.assert_allclose(np.quantile(mb[key.replace('|','_')],[.025,.975]),value['ci'],atol=1e-12)
    result=dict(passed=True,conditions=6,arms=2,primary_and_clean_bootstrap_draws_verified=2000,
                matched_C_cells_verified=8,matched_C_bootstrap_entries_verified=100,
                weighted_training_max_abs_gradient=float(np.max(np.abs(gradient))),
                weighted_training_intercept_gradient=float(intercept_gradient),
                prior_control_predictions_unchanged=True,scaler_unchanged=True,
                summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Head reconstruction, all twelve condition/arm metrics and 2,000 paired primary/clean bootstrap draws verified.')


if __name__=='__main__':main()
