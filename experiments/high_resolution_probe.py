"""Controlled 224px versus low-information paired linear-probe experiment."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/torch'))
import argparse
import csv
import hashlib
import json
import warnings
import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from external_eval import summarize, risk_coverage, COVERAGES, ece, auroc
from temperature_scaling import nll


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def low_information(image):
    """Keep input dimensions fixed while discarding high-frequency information."""
    p=Image.fromarray(image)
    return np.asarray(p.resize((28,28),Image.Resampling.BICUBIC).resize((224,224),Image.Resampling.BICUBIC))


def audit_data(archive,small,indices):
    seen={};audit={};within=[]
    for original in ['train','val','test']:
        np.testing.assert_array_equal(archive[original+'_labels'],small[original+'_labels'])
        if archive[original+'_images'].shape[1:]!=(224,224):raise ValueError('Expected genuine 224px archive')
    for split,idx in indices.items():
        original='val' if split in ['selection','calibration'] else split
        images=archive[original+'_images'];audit[split]=len(idx)
        for i in idx:
            digest=hashlib.sha256(images[i].tobytes()).hexdigest()
            if digest in seen:
                previous=seen[digest]
                if previous[0]!=split:raise ValueError(f'224px cross-split duplicate: {previous}, {(split,i)}')
                within.append([previous,(split,i)])
            else:seen[digest]=(split,i)
    if within:raise ValueError(f'224px within-split duplicate requires protocol update: {within[:3]}')
    return dict(retained_counts=audit,labels_match_28px=True,exact_duplicates_in_retained_cases=0,
                scope='Exact pixel audit only; no patient identity assurance')


def prepare_features(cfg,indices,audit):
    import torch
    import torchvision
    from torchvision.models import resnet18,ResNet18_Weights
    torch.set_num_threads(2)
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc=torch.nn.Identity();model.eval()
    for p in model.parameters():p.requires_grad_(False)
    checkpoint=ROOT/'data/torch/hub/checkpoints/resnet18-f37072fd.pth'
    provenance=dict(feature_script_sha256=sha(__file__), archive_sha256=sha(ROOT/'data/pneumoniamnist_224.npz'),
                    manifest_sha256=sha(ROOT/'configs/resolution_manifest.json'),
                    protocol_sha256=sha(ROOT/'configs/high_resolution_probe.json'),
                    encoder_sha256=sha(checkpoint),torch=torch.__version__,torchvision=torchvision.__version__)
    cache=ROOT/'data/high_resolution_features';cache.mkdir(exist_ok=True)
    meta=cache/'provenance.json'
    if meta.exists():
        if json.loads(meta.read_text())!=provenance:raise ValueError('Feature-cache provenance mismatch')
    else:meta.write_text(json.dumps(provenance,indent=2)+'\n')
    data=np.load(ROOT/'data/pneumoniamnist_224.npz');features={};labels={}
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,:,None,None]
    for split,ix in indices.items():
        original='val' if split in ['selection','calibration'] else split
        images=data[original+'_images'][ix];labels[split]=data[original+'_labels'][ix].ravel()
        for view in cfg['views']:
            path=cache/f'{view}_{split}.npz'
            if path.exists():
                saved=np.load(path);np.testing.assert_array_equal(saved['indices'],ix)
                np.testing.assert_array_equal(saved['labels'],labels[split]);features[view,split]=saved['features']
                print(f'Cached {view}/{split}: {len(ix)} cases',flush=True);continue
            batches=[]
            for start in range(0,len(images),32):
                batch=images[start:start+32]
                if view=='downsample28_upsample224':batch=np.stack([low_information(x) for x in batch])
                x=torch.tensor(batch[:,None],dtype=torch.float32).repeat(1,3,1,1)/255
                with torch.inference_mode():batches.append(model((x-mean)/std).numpy())
                if start%512==0:print(f'Features {view}/{split}: {min(start+32,len(images))}/{len(images)}',flush=True)
            features[view,split]=np.concatenate(batches)
            assert features[view,split].shape==(len(ix),512)
            np.savez_compressed(path,features=features[view,split],labels=labels[split],indices=ix)
    provenance['feature_sha256']={p.name:sha(p) for p in cache.glob('*.npz')}
    provenance['audit']=audit
    return features,labels,provenance


def evaluate(y,z,cut):
    p=expit(z)
    metrics=summarize(y,p,cut)
    metrics.update(nll=nll(y,z),brier=float(np.mean((p-y)**2)))
    return metrics


def bootstrap_metrics(y,z,cut):
    p=expit(z);pred=p>=.5;keep=np.maximum(p,1-p)>=cut
    return np.array([auroc(y,p),np.mean(pred==y),nll(y,z),np.mean((p-y)**2),ece(y,p),
                     keep.mean(),np.mean(pred[keep]!=y[keep]) if keep.any() else np.nan])


def main(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/high_resolution_probe.json').read_text())
    archive=ROOT/'data/pneumoniamnist_224.npz'
    assert hashlib.md5(archive.read_bytes()).hexdigest()==cfg['source_md5']
    indices=json.loads((ROOT/'configs/resolution_manifest.json').read_text())['indices']
    data=np.load(archive);small=np.load(ROOT/'data/pneumoniamnist.npz')
    audit=audit_data(data,small,indices)
    features,labels,provenance=prepare_features(cfg,indices,audit)
    fits={};arrays={};rows=[];choices=[]
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for view in cfg['views']:
        scaler=StandardScaler().fit(features[view,'train'])
        x=scaler.transform(features[view,'train']);vx=scaler.transform(features[view,'selection'])
        candidates=[]
        for c in [.01,.1,1.,10.]:
            model=LogisticRegression(C=c,solver='lbfgs',max_iter=3000).fit(x,labels['train'])
            loss=nll(labels['selection'],model.decision_function(vx));candidates.append((loss,c,model))
            choices.append(dict(view=view,C=c,selection_nll=loss,iterations=int(model.n_iter_[0])))
        loss,c,model=min(candidates,key=lambda v:v[0])
        zcal=model.decision_function(scaler.transform(features[view,'calibration']));pcal=expit(zcal)
        cut=float(np.quantile(np.maximum(pcal,1-pcal),.2,method='linear'))
        z=model.decision_function(scaler.transform(features[view,'test']))
        fits[view]=dict(C=c,selection_nll=loss,threshold=cut,**evaluate(labels['test'],z,cut))
        arrays[view+'_test_logits']=z;arrays[view+'_calibration_logits']=zcal
        arrays[view+'_coefficient']=model.coef_;arrays[view+'_intercept']=model.intercept_
        arrays[view+'_scaler_mean']=scaler.mean_;arrays[view+'_scaler_scale']=scaler.scale_
        rows.append(dict(view=view,**fits[view]))
        print(view,fits[view],flush=True)
    y=labels['test'];rng=np.random.default_rng(cfg['bootstrap_seed']);boots=[]
    for _ in range(cfg['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y));values=[]
        for view in cfg['views']:values.append(bootstrap_metrics(y[ix],arrays[view+'_test_logits'][ix],fits[view]['threshold']))
        boots.append(values[0]-values[1])
    boots=np.array(boots);point=[bootstrap_metrics(y,arrays[v+'_test_logits'],fits[v]['threshold']) for v in cfg['views']]
    keys=['auroc','accuracy','nll','brier','ece','coverage','retained_error']
    delta={key:dict(point=float(point[0][i]-point[1][i]),ci=np.nanquantile(boots[:,i],[.025,.975]).tolist(),
                    undefined_replicates=int(np.isnan(boots[:,i]).sum())) for i,key in enumerate(keys)}
    provenance['script_sha256']=sha(__file__)
    summary=dict(config=cfg,provenance=provenance,selection_candidates=choices,results=fits,
                 native_minus_low_information=delta,
                 interval_scope='Paired test-representative bootstrap conditional on encoder, selected probes and cutoffs; no patient clusters or model-selection uncertainty')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(output/'predictions_and_probes.npz',**arrays,**{s+'_labels':v for s,v in labels.items()},test_indices=indices['test'])
    np.savez_compressed(output/'bootstrap.npz',native_minus_low_information=boots)
    fig,ax=plt.subplots(figsize=(6,4))
    for view in cfg['views']:ax.plot(COVERAGES,risk_coverage(y,expit(arrays[view+'_test_logits'])),label=view)
    ax.set(xlabel='Coverage',ylabel='Error among retained cases',ylim=(0,.4),title='Frozen ResNet-18: paired information-resolution comparison')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'risk_coverage.png',dpi=180);fig.savefig(output/'risk_coverage.pdf');plt.close(fig)
    print('Primary AUROC difference',delta['auroc'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/high_resolution')
    main(parser.parse_args().output)
