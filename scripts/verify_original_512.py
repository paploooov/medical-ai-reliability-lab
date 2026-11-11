"""Independent reconstruction of the new cohort and fitted-model endpoints."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_original_xray import digest
from verify_high_resolution_robustness import grouped_risk

def main():
    out=ROOT/'reports/original_512';s=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text());r=manifest['records'];p=np.load(out/'predictions.npz');heads=np.load(out/'probe_heads.npz');boots=np.load(out/'bootstrap.npz');test=p['test_indices'];y=p['labels'][test]
    assert len({v['pixel_sha256'] for v in r})==len(r)
    groups={a:{r[i]['group'] for i in p[a+'_indices']} for a in ['train','selection','calibration','test']}
    for a in groups:
        for b in groups:
            if a!=b:assert not groups[a]&groups[b]
    for arm in s['results']:
        z=p[arm][test];v=s['results'][arm]
        np.testing.assert_allclose(roc_auc_score(y,expit(z)),v['auroc']['point'],atol=1e-12)
        # Fine-tuning stores float32 logits; stable float64 reconstruction allows
        # for the original float32 logaddexp rounding, not model differences.
        np.testing.assert_allclose(log_loss(y,expit(z.astype(np.float64))),v['nll']['point'],atol=1e-6)
        np.testing.assert_allclose(grouped_risk(y,expit(z)),v['risk_at_80']['point'],atol=1e-12)
        if arm.endswith('_probe'):
            key=arm.removesuffix('_probe');f=np.load(ROOT/f'data/original_xray/{key}_features.npz')['features']
            np.testing.assert_allclose(f@heads[arm+'_weight']+heads[arm+'_bias'],p[arm],atol=2e-5,rtol=1e-5)
            choices=s['selection'][arm]['candidates'];assert s['selection'][arm]['C']==min(choices,key=lambda v:v['selection_nll'])['C']
    hist=s['selection']['mobile_512_finetuned']['history'];assert s['selection']['mobile_512_finetuned']['selected_epoch']==min(hist,key=lambda v:v['selection_nll'])['epoch']
    names=np.array([r[i]['group'] for i in test]);unique=np.unique(names);members=[np.flatnonzero(names==g) for g in unique];rng=np.random.default_rng(s['config']['bootstrap']['seed']);differences=[]
    for b in range(2000):
        ids=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))]);values={}
        for a in ['mobile_224_probe','mobile_512_probe','mobile_512_finetuned','resnet18_224_probe']:
            values[a]=roc_auc_score(y[ids],expit(p[a][test][ids]));np.testing.assert_allclose(values[a],boots[a][b,0],atol=1e-12)
        differences.append(values['mobile_512_probe']-values['mobile_224_probe'])
    np.testing.assert_allclose(np.quantile(differences,[.025,.975]),s['primary']['ci'],atol=1e-12)
    numerical={a:dict(probability_saturation=int(np.isin(expit(p[a][test]),[0,1]).sum()),logit_minus_probability_auroc=float(roc_auc_score(y,p[a][test])-roc_auc_score(y,expit(p[a][test])))) for a in s['results']}
    result=dict(passed=True,patient_group_overlap=0,pixel_duplicates=0,primary_cluster_bootstrap_draws_verified=2000,all_four_auroc_draws_verified=True,selection_rules_verified=True,numerical_sensitivity=numerical,prediction_sha256=digest(out/'predictions.npz'),manifest_sha256=digest(out/'manifest.json'))
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
