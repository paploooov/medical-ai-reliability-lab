"""Larger frozen encoder on the identical original-source cohort."""
import time,json,csv,warnings
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet50,ResNet50_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.special import expit
from original_512 import ROOT,CACHE,features,folded_head,vector
from prepare_original_xray import digest
from high_resolution_probe import evaluate
from high_resolution_robustness import interval
from temperature_scaling import nll

def main():
    torch.set_num_threads(1);torch.manual_seed(20261121);out=ROOT/'reports/original_512/large_encoder';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/original_512_large.json').read_text());manifest=json.loads((out.parent/'manifest.json').read_text());cache=json.loads((CACHE/'cache_provenance.json').read_text());assert cache['manifest_sha256']==digest(out.parent/'manifest.json');assert cache['images']['224']==digest(CACHE/'images_224.npy')
    records=manifest['records'];y=np.array([r['label'] for r in records]);ix={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','selection','calibration','test']}
    model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval();model.fc=torch.nn.Identity()
    for p in model.parameters():p.requires_grad_(False)
    f=features(model,224,'resnet50_224',manifest);scaler=StandardScaler().fit(f[ix['train']]);x=scaler.transform(f);candidates=[]
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for c in [.01,.1,1.,10.]:
        m=LogisticRegression(C=c,solver='lbfgs',max_iter=3000).fit(x[ix['train']],y[ix['train']]);loss=nll(y[ix['selection']],m.decision_function(x[ix['selection']]));candidates.append((loss,c,m))
    loss,c,m=min(candidates,key=lambda v:v[0]);z=m.decision_function(x);p=expit(z[ix['calibration']]);cut=float(np.quantile(np.maximum(p,1-p),.2));w,b=folded_head(m.coef_.ravel(),m.intercept_[0],scaler.mean_,scaler.scale_)
    np.savez_compressed(out/'head_and_predictions.npz',weight=w,bias=np.array(b),logits=z,labels=y,**{k+'_indices':v for k,v in ix.items()})
    selection=dict(C=c,selection_nll=loss,candidates=[dict(C=c,selection_nll=l) for l,c,_ in candidates]);(out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('Selected large encoder',selection,flush=True)
    base_path=out.parent/'summary.json'
    while not base_path.exists():time.sleep(10)
    base=json.loads(base_path.read_text());bp=np.load(out.parent/'predictions.npz');bcut=float(np.quantile(np.maximum(expit(bp['resnet18_224_probe'][ix['calibration']]),1-expit(bp['resnet18_224_probe'][ix['calibration']])),.2));test=ix['test'];yt=y[test];bz=bp['resnet18_224_probe'][test]
    groups=np.array([records[i]['group'] for i in test]);members=[np.flatnonzero(groups==g) for g in np.unique(groups)];rng=np.random.default_rng(20261122);large=[];small=[]
    for _ in range(2000):
        ids=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))]);large.append(vector(yt[ids],z[test][ids],cut));small.append(vector(yt[ids],bz[ids],bcut))
    large=np.array(large);small=np.array(small);point=vector(yt,z[test],cut)-vector(yt,bz,bcut);metrics=['auroc','nll','accuracy','risk_at_80','coverage','retained_error']
    summary=dict(config=cfg,selection=selection,metrics=evaluate(yt,z[test],cut),parameters=sum(p.numel() for p in model.parameters())+len(w)+1,contrast={k:interval(point[i],large[:,i]-small[:,i]) for i,k in enumerate(metrics)},provenance=dict(script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'configs/original_512_large.json'),manifest_sha256=digest(out.parent/'manifest.json'),feature_sha256=digest(CACHE/'resnet50_224_features.npz'),encoder_sha256=digest(ROOT/'data/torch/hub/checkpoints/resnet50-0676ba61.pth')))
    np.savez_compressed(out/'bootstrap.npz',resnet50=large,resnet18=small);(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print('COMPLETE larger encoder',summary['contrast']['auroc'],flush=True)

if __name__=='__main__':main()
