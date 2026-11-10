"""Targeted mixed corruption training, with a matched reduced-noise control."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import argparse,csv,hashlib,json,warnings
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from high_resolution_augmentation import standardize
from high_resolution_severity import perturb
from high_resolution_probe import evaluate
from high_resolution_robustness import interval
from external_eval import risk_at
from temperature_scaling import nll


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mixture(features,labels,weights):
    if any(v<0 for v in weights.values()) or not np.isclose(sum(weights.values()),1):raise ValueError('Mixture weights must sum to one')
    names=[k for k,v in weights.items() if v>0]
    if any(len(features[k])!=len(labels) for k in names):raise ValueError('Unpaired features')
    return np.concatenate([features[k] for k in names]),np.tile(labels,len(names)),np.concatenate([np.full(len(labels),weights[k]) for k in names])


def selective(y,z,cut):
    p=expit(z);keep=np.maximum(p,1-p)>=cut
    return np.array([risk_at(y,p,.8),keep.mean(),np.mean((p[keep]>=.5)!=y[keep]) if keep.any() else np.nan])


def main(output):
    import torch
    from torchvision.models import resnet18
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg_path=ROOT/'configs/high_resolution_mixed_augmentation.json';cfg=json.loads(cfg_path.read_text())
    parent=json.loads((ROOT/'reports/high_resolution/augmentation/summary.json').read_text())
    severity=json.loads((ROOT/'reports/high_resolution/severity/summary.json').read_text())
    base=json.loads((ROOT/'reports/high_resolution/summary.json').read_text())
    probes=np.load(ROOT/'reports/high_resolution/predictions_and_probes.npz')
    aug=np.load(ROOT/'reports/high_resolution/augmentation/augmented_head.npz')
    clean={s:np.load(ROOT/f'data/high_resolution_features/native_224_{s}.npz') for s in ['train','selection','calibration','test']}
    for split,d in clean.items():
        name=f'native_224_{split}.npz';assert sha(ROOT/'data/high_resolution_features'/name)==base['provenance']['feature_sha256'][name]
    noisy_path=ROOT/'data/high_resolution_augmentation/train_noise_20261116.npz'
    assert sha(noisy_path)==parent['feature_sha256'][noisy_path.name]
    noisy=np.load(noisy_path);np.testing.assert_array_equal(noisy['indices'],clean['train']['indices'])
    archive=ROOT/'data/pneumoniamnist_224.npz';checkpoint=ROOT/'data/torch/hub/checkpoints/resnet18-f37072fd.pth'
    assert sha(archive)==base['provenance']['archive_sha256'];assert sha(checkpoint)==base['provenance']['encoder_sha256']
    provenance=dict(script_sha256=sha(__file__),config_sha256=sha(cfg_path),archive_sha256=sha(archive),encoder_sha256=sha(checkpoint),
                    original_probes_sha256=sha(ROOT/'reports/high_resolution/predictions_and_probes.npz'),
                    original_augmented_head_sha256=sha(ROOT/'reports/high_resolution/augmentation/augmented_head.npz'),
                    helpers_sha256={n:sha(ROOT/'experiments'/n) for n in ['high_resolution_augmentation.py','high_resolution_severity.py','high_resolution_probe.py','high_resolution_robustness.py','external_eval.py','temperature_scaling.py']})
    cache=ROOT/'data/high_resolution_mixed';cache.mkdir(exist_ok=True);meta=cache/'provenance.json'
    if meta.exists():assert json.loads(meta.read_text())==provenance
    else:meta.write_text(json.dumps(provenance,indent=2)+'\n')
    path=cache/'train_blur4.npz'
    if path.exists():
        blur=np.load(path);np.testing.assert_array_equal(blur['indices'],clean['train']['indices']);bf=blur['features']
    else:
        torch.set_num_threads(2);encoder=resnet18(weights=None);encoder.load_state_dict(torch.load(checkpoint,weights_only=True));encoder.fc=torch.nn.Identity();encoder.eval()
        for param in encoder.parameters():param.requires_grad_(False)
        mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
        indices=clean['train']['indices'];images=np.load(archive)['train_images'][indices];chunks=[]
        for start in range(0,len(images),32):
            batch=np.stack([perturb(im,i,'blur',4) for im,i in zip(images[start:start+32],indices[start:start+32])])
            x=torch.tensor(batch[:,None],dtype=torch.float32).repeat(1,3,1,1)/255
            with torch.inference_mode():chunks.append(encoder((x-mean)/std).numpy())
            if start%512==0:print('Blur training features',min(start+32,len(images)),len(images),flush=True)
        bf=np.concatenate(chunks);np.savez_compressed(path,features=bf,indices=indices)
    features={k:standardize(v,probes) for k,v in [('clean',clean['train']['features']),('noise',noisy['features']),('blur',bf)]}
    selection_x=standardize(clean['selection']['features'],probes);cal_x=standardize(clean['calibration']['features'],probes)
    heads={'control':(probes['native_224_coefficient'],probes['native_224_intercept']), 'noise':(aug['coefficient'],aug['intercept'])}
    cutoffs={'control':base['results']['native_224']['threshold'],'noise':parent['selection']['augmented_cutoff']}
    selections={};parameters={};warnings.filterwarnings('error',category=ConvergenceWarning)
    for arm,weights in cfg['arms'].items():
        x,y,w=mixture(features,clean['train']['labels'],weights);candidates=[]
        for c in [.01,.1,1.,10.]:
            model=LogisticRegression(C=c,solver='lbfgs',max_iter=3000).fit(x,y,sample_weight=w)
            loss=nll(clean['selection']['labels'],model.decision_function(selection_x));candidates.append((loss,c,model))
        loss,c,model=min(candidates,key=lambda v:v[0]);heads[arm]=(model.coef_,model.intercept_)
        zcal=model.decision_function(cal_x);pc=expit(zcal);cutoffs[arm]=float(np.quantile(np.maximum(pc,1-pc),.2))
        selections[arm]=dict(C=c,selection_nll=loss,cutoff=cutoffs[arm],total_weight=float(w.sum()),rows=len(y),weights=weights,
                            candidates=[dict(C=cv,selection_nll=l,iterations=int(m.n_iter_[0])) for l,cv,m in candidates])
        parameters[arm+'_coef']=model.coef_;parameters[arm+'_intercept']=model.intercept_;parameters[arm+'_calibration_logits']=zcal
        print('Selected',arm,selections[arm],flush=True)
    (out/'selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    np.savez_compressed(out/'new_heads.npz',**parameters,scaler_mean=probes['native_224_scaler_mean'],scaler_scale=probes['native_224_scaler_scale'])
    sp=np.load(ROOT/'reports/high_resolution/severity/predictions.npz');ap=np.load(ROOT/'reports/high_resolution/augmentation/predictions.npz')
    conditions=list(severity['joint_grid_band']['grid'])+['noise_7','noise_99','contrast'];y=clean['test']['labels'];logits={};rows=[]
    lookup={'clean':ROOT/'data/high_resolution_features/native_224_test.npz','noise_0.1':ROOT/'data/high_resolution_augmentation/test_noise_42.npz','blur_8':ROOT/'data/high_resolution_augmentation/test_relative_blur.npz'}
    for condition in conditions:
        if condition in lookup:path=lookup[condition]
        elif condition in ['noise_7','noise_99','contrast']:path=ROOT/f'data/high_resolution_augmentation/test_{condition}.npz'
        else:path=ROOT/f'data/high_resolution_severity/{condition}.npz'
        d=np.load(path);np.testing.assert_array_equal(d['indices'],sp['test_indices']);x=standardize(d['features'],probes)
        for arm,(coef,intercept) in heads.items():
            z=x@coef.ravel()+intercept[0];logits[arm,condition]=z
            if arm in ['control','noise']:
                old=sp if condition in severity['joint_grid_band']['grid'] else ap
                np.testing.assert_allclose(z,old[('augmentation' if arm=='noise' else arm)+'__'+condition],atol=1e-10)
            rows.append(dict(arm=arm,condition=condition,threshold=cutoffs[arm],**evaluate(y,z,cutoffs[arm])))
    with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    point={key:selective(y,z,cutoffs[key[0]]) for key,z in logits.items()};boots={key:[] for key in logits};rng=np.random.default_rng(cfg['bootstrap_seed'])
    for _ in range(cfg['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y))
        for key,z in logits.items():boots[key].append(selective(y[ix],z[ix],cutoffs[key[0]]))
    boots={key:np.array(v) for key,v in boots.items()};metrics=['risk_at_80','coverage','retained_error']
    summary=dict(config=cfg,provenance=provenance,selection=selections,results={},contrasts={})
    for condition in conditions:
        for arm in heads:
            key=arm,condition;summary['results'][arm+'|'+condition]={m:interval(point[key][i],boots[key][:,i]) for i,m in enumerate(metrics)}
        for comparator in ['noise','reduced_noise','control']:
            a,b=('mixed',condition),(comparator,condition)
            summary['contrasts'][f'mixed_minus_{comparator}|{condition}']={m:interval(point[a][i]-point[b][i],boots[a][:,i]-boots[b][:,i]) for i,m in enumerate(metrics)}
    summary['primary']=summary['contrasts']['mixed_minus_noise|blur_4']['risk_at_80']
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'predictions.npz',labels=y,test_indices=sp['test_indices'],**{a+'__'+c:z for (a,c),z in logits.items()})
    np.savez_compressed(out/'bootstrap.npz',**{a+'__'+c:v for (a,c),v in boots.items()})
    print('PRIMARY',summary['primary'],flush=True)
    for condition in ['clean','noise_0.1','noise_0.2','blur_2','blur_4','blur_8','blur_16']:
        print(condition,{a:summary['results'][a+'|'+condition]['risk_at_80']['point'] for a in heads},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/high_resolution/mixed_augmentation')
    main(p.parse_args().output)
