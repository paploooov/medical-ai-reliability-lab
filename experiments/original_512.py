"""Original radiographs: resolution, larger encoder, and partial fine-tuning."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/torch'))
import sys,time,json,csv,copy,warnings
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights,resnet18,ResNet18_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.special import expit
from prepare_original_xray import digest
from high_resolution_probe import evaluate
from high_resolution_robustness import interval
from temperature_scaling import nll
from external_eval import risk_at,auroc

CACHE=ROOT/'data/original_xray';OUT=ROOT/'reports/original_512'

def normalize(images):
    x=torch.tensor(np.asarray(images).copy(),dtype=torch.float32)[:,None].repeat(1,3,1,1)/255
    return (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]

def folded_head(coef,intercept,mean,scale):
    weight=coef/scale
    return weight,intercept-np.sum(weight*mean)

def mobile_parts(model):
    return model.features[:10],torch.nn.Sequential(model.features[10:],torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())

def features(model,size,key,manifest):
    images=np.load(CACHE/f'images_{size}.npy',mmap_mode='r');path=CACHE/f'{key}_features.npz';n=len(images)
    if path.exists():return np.load(path)['features']
    pooled=[];cached=None;t=time.monotonic()
    if key=='mobile_512':
        prefix,tail=mobile_parts(model)
        with torch.inference_mode():shape=tuple(prefix(normalize(images[:1])).shape[1:])
        cached=np.lib.format.open_memmap(CACHE/'mobile512_prefix.npy',mode='w+',dtype='float32',shape=(n,*shape))
    for start in range(0,n,16):
        x=normalize(images[start:start+16])
        with torch.inference_mode():
            if key=='mobile_512':
                z=prefix(x);cached[start:start+len(x)]=z.numpy();f=tail(z)
            elif key.startswith('mobile'):f=model.avgpool(model.features(x)).flatten(1)
            else:f=model(x)
        pooled.append(f.numpy())
        if start%512==0:print('Features',key,start,n,'seconds',round(time.monotonic()-t),flush=True)
    if cached is not None:cached.flush();del cached
    result=np.concatenate(pooled);np.savez_compressed(path,features=result)
    return result

def vector(y,z,cut):
    p=expit(z);keep=np.maximum(p,1-p)>=cut
    return np.array([auroc(y,p),nll(y,z),np.mean((p>=.5)==y),risk_at(y,p,.8),keep.mean(),np.mean((p[keep]>=.5)!=y[keep]) if keep.any() else np.nan])

def main():
    torch.set_num_threads(1);torch.manual_seed(20261121);np.random.seed(20261121)
    cfg=json.loads((ROOT/'configs/original_512.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    cache_meta=json.loads((CACHE/'cache_provenance.json').read_text());assert cache_meta['manifest_sha256']==digest(OUT/'manifest.json')
    for size,h in cache_meta['images'].items():assert digest(CACHE/f'images_{size}.npy')==h
    records=manifest['records'];y=np.array([r['label'] for r in records]);ix={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','selection','calibration','test']}
    provenance=dict(script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'configs/original_512.json'),manifest_sha256=digest(OUT/'manifest.json'),cache=cache_meta,torch=torch.__version__)
    meta=CACHE/'model_cache_provenance.json'
    if meta.exists():assert json.loads(meta.read_text())==provenance
    else:meta.write_text(json.dumps(provenance,indent=2)+'\n')
    mobile=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    resnet=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval();resnet.fc=torch.nn.Identity()
    for model in [mobile,resnet]:
        for p in model.parameters():p.requires_grad_(False)
    selection={};all_logits={};cutoffs={};probe_arrays={};models={}
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for key,size,encoder in [('mobile_224',224,mobile),('mobile_512',512,mobile),('resnet18_224',224,resnet)]:
        f=features(encoder,size,key,manifest);scaler=StandardScaler().fit(f[ix['train']]);x=scaler.transform(f)
        candidates=[]
        for c in [.01,.1,1.,10.]:
            m=LogisticRegression(C=c,solver='lbfgs',max_iter=3000).fit(x[ix['train']],y[ix['train']]);loss=nll(y[ix['selection']],m.decision_function(x[ix['selection']]))
            candidates.append((loss,c,m))
        loss,c,m=min(candidates,key=lambda v:v[0]);arm=key+'_probe';z=m.decision_function(x);all_logits[arm]=z
        cal=expit(z[ix['calibration']]);cutoffs[arm]=float(np.quantile(np.maximum(cal,1-cal),.2))
        selection[arm]=dict(C=c,selection_nll=loss,candidates=[dict(C=c,selection_nll=l) for l,c,_ in candidates])
        w,b=folded_head(m.coef_.ravel(),m.intercept_[0],scaler.mean_,scaler.scale_)
        probe_arrays[arm+'_weight']=w;probe_arrays[arm+'_bias']=np.array(b)
        print('Selected',arm,selection[arm],flush=True)
        models[arm]=dict(parameters=sum(p.numel() for p in encoder.parameters())-(sum(p.numel() for p in encoder.classifier.parameters()) if key.startswith('mobile') else 0)+len(w)+1,feature_dimension=len(w),input_size=size)
    np.savez_compressed(OUT/'probe_heads.npz',**probe_arrays)
    # Only selection labels are inspected during fitting/checkpoint selection.
    _,tail=mobile_parts(mobile);tail=copy.deepcopy(tail).eval();head=torch.nn.Linear(576,1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(probe_arrays['mobile_512_probe_weight'][None],dtype=torch.float32));head.bias.copy_(torch.tensor([float(probe_arrays['mobile_512_probe_bias'])],dtype=torch.float32))
    for p in tail.parameters():p.requires_grad_(True)
    for module in tail.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            for p in module.parameters():p.requires_grad_(False)
    net=torch.nn.Sequential(tail,head).eval();prefix=np.load(CACHE/'mobile512_prefix.npy',mmap_mode='r')
    def predict(indices):
        chunks=[]
        with torch.inference_mode():
            for start in range(0,len(indices),32):chunks.append(net(torch.tensor(prefix[indices[start:start+32]].copy())).ravel().numpy())
        return np.concatenate(chunks)
    initial=predict(ix['selection']);np.testing.assert_allclose(initial,all_logits['mobile_512_probe'][ix['selection']],atol=1e-3,rtol=1e-4)
    best_loss=nll(y[ix['selection']],initial);best=copy.deepcopy(net.state_dict());best_epoch=0
    history=[dict(epoch=0,selection_nll=best_loss)];optimizer=torch.optim.AdamW([{'params':[p for p in tail.parameters() if p.requires_grad],'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=1e-4)
    rng=np.random.default_rng(cfg['seed'])
    for epoch in range(1,4):
        order=rng.permutation(ix['train']);total=0.;t=time.monotonic()
        for start in range(0,len(order),16):
            batch=order[start:start+16];x=torch.tensor(prefix[batch].copy());target=torch.tensor(y[batch],dtype=torch.float32)
            optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(net(x).ravel(),target);loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
            if start%512==0:print('Train epoch',epoch,start,len(order),'seconds',round(time.monotonic()-t),flush=True)
        val=nll(y[ix['selection']],predict(ix['selection']));history.append(dict(epoch=epoch,training_nll=total/len(order),selection_nll=val))
        if val<best_loss:best_loss=val;best=copy.deepcopy(net.state_dict());best_epoch=epoch
        torch.save({'state':net.state_dict(),'epoch':epoch},CACHE/f'finetune_epoch{epoch}.pt')
        print('Epoch',history[-1],flush=True)
    net.load_state_dict(best);arm='mobile_512_finetuned';all_logits[arm]=predict(np.arange(len(y)));cal=expit(all_logits[arm][ix['calibration']]);cutoffs[arm]=float(np.quantile(np.maximum(cal,1-cal),.2))
    selection[arm]=dict(selected_epoch=best_epoch,selection_nll=best_loss,history=history)
    models[arm]=dict(**models['mobile_512_probe'],trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))
    # Local deployment artifact: full encoder + adapted last blocks + folded head.
    for j,block in enumerate(tail[0]):mobile.features[10+j]=copy.deepcopy(block)
    mobile.classifier=copy.deepcopy(head);mobile.eval()
    package=CACHE/'mobile512_research.pt';torch.save({'state_dict':mobile.state_dict(),'input_size':512,'cutoff':cutoffs[arm],'selected_epoch':best_epoch},package)
    sample=normalize(np.load(CACHE/'images_512.npy',mmap_mode='r')[ix['selection'][:2]])
    with torch.inference_mode():np.testing.assert_allclose(mobile(sample).ravel().numpy(),all_logits[arm][ix['selection'][:2]],atol=2e-4,rtol=1e-4)
    torch.jit.trace(mobile,sample[:1]).save(str(CACHE/'mobile512_research.ts'))
    test=ix['test'];yt=y[test];rows=[]
    for arm,z in all_logits.items():rows.append(dict(arm=arm,threshold=cutoffs[arm],**evaluate(yt,z[test],cutoffs[arm])))
    with (OUT/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'predictions.npz',labels=y,**{s+'_indices':v for s,v in ix.items()},**all_logits)
    groups=np.array([records[i]['group'] for i in test]);unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique]
    metric_names=['auroc','nll','accuracy','risk_at_80','coverage','retained_error'];point={a:vector(yt,z[test],cutoffs[a]) for a,z in all_logits.items()};boots={a:[] for a in all_logits};rng=np.random.default_rng(cfg['bootstrap']['seed'])
    for _ in range(cfg['bootstrap']['replicates']):
        idx=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))])
        for a,z in all_logits.items():boots[a].append(vector(yt[idx],z[test][idx],cutoffs[a]))
    boots={a:np.array(v) for a,v in boots.items()};summary=dict(config=cfg,provenance=provenance,selection=selection,models=models,results={},contrasts={},deployment=dict(checkpoint=str(package.relative_to(ROOT)),checkpoint_sha256=digest(package),torchscript_sha256=digest(CACHE/'mobile512_research.ts'),verified_selection_examples=2,coreml_converted=False))
    for a in all_logits:summary['results'][a]={m:interval(point[a][i],boots[a][:,i]) for i,m in enumerate(metric_names)}
    for a,b in [('mobile_512_probe','mobile_224_probe'),('mobile_512_finetuned','mobile_512_probe'),('resnet18_224_probe','mobile_224_probe')]:summary['contrasts'][a+'_minus_'+b]={m:interval(point[a][i]-point[b][i],boots[a][:,i]-boots[b][:,i]) for i,m in enumerate(metric_names)}
    summary['primary']=summary['contrasts']['mobile_512_probe_minus_mobile_224_probe']['auroc']
    eligible=np.array([max(records[i]['width'],records[i]['height'])>=512 for i in test]);summary['native_long_edge_subgroup']=dict(n=int(eligible.sum()),results={a:evaluate(yt[eligible],z[test][eligible],cutoffs[a]) for a,z in all_logits.items()})
    np.savez_compressed(OUT/'bootstrap.npz',**boots);(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print('COMPLETE primary',summary['primary'],flush=True)

if __name__=='__main__':main()
