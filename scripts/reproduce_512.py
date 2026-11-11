"""Isolated full 512px training reproduction. Test evaluation requires opt-in."""
import argparse,copy,hashlib,io,json,os,platform,resource,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from prepare_original_xray import digest,letterbox
ROOT=Path(__file__).resolve().parents[1]

def normalize(a):
    x=torch.tensor(np.asarray(a).copy(),dtype=torch.float32)[:,None].repeat(1,3,1,1)/255
    return (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]

def nll(y,z):return float(np.mean(np.logaddexp(0,z)-y*z))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=20261121);parser.add_argument('--archive',type=Path);parser.add_argument('--evaluate-test',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stages={}
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    cfg=json.loads((ROOT/'configs/original_512.json').read_text());manifest_path=ROOT/'reports/original_512/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['archive_sha256']==cfg['archive_sha256']
    archive=args.archive or out/'ChestXRay2017.zip'
    if not archive.exists():
        urllib.request.urlretrieve(cfg['archive_url'],archive.with_suffix('.partial'))
        assert digest(archive.with_suffix('.partial'))==cfg['archive_sha256'];archive.with_suffix('.partial').rename(archive)
    assert digest(archive)==cfg['archive_sha256']
    os.environ['TORCH_HOME']=str(out/'torch')
    model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    for p in model.parameters():p.requires_grad_(False)
    stages['source_verification_and_weights_seconds']=time.monotonic()-start
    records=[r for r in manifest['records'] if r['split']!='test'];assert all(r['split'] in ['train','selection','calibration'] for r in records)
    y=np.array([r['label'] for r in records]);ix={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','selection','calibration']}
    prefix=model.features[:10];tail=torch.nn.Sequential(model.features[10:],torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten());t=time.monotonic();pooled=[]
    with zipfile.ZipFile(archive) as zf:
        for first in range(0,len(records),16):
            batch=np.stack([letterbox(Image.open(io.BytesIO(zf.read(r['name']))),512) for r in records[first:first+16]])
            with torch.inference_mode():z=prefix(normalize(batch));f=tail(z)
            if first==0:cache=np.lib.format.open_memmap(out/'prefix.npy',mode='w+',dtype='float32',shape=(len(records),*z.shape[1:]))
            cache[first:first+len(batch)]=z.numpy();pooled.append(f.numpy())
            if first%512==0:print('Prepare',first,len(records),round(time.monotonic()-t),flush=True)
    cache.flush();features=np.concatenate(pooled);stages['decode_resize_features_seconds']=time.monotonic()-t;t=time.monotonic()
    scaler=StandardScaler().fit(features[ix['train']]);x=scaler.transform(features);candidates=[]
    for c in [.01,.1,1.,10.]:
        probe=LogisticRegression(C=c,solver='lbfgs',max_iter=3000).fit(x[ix['train']],y[ix['train']]);candidates.append((nll(y[ix['selection']],probe.decision_function(x[ix['selection']])),c,probe))
    loss,c,probe=min(candidates,key=lambda v:v[0]);w=probe.coef_.ravel()/scaler.scale_;b=probe.intercept_[0]-np.sum(w*scaler.mean_)
    head=torch.nn.Linear(576,1)
    with torch.no_grad():head.weight.copy_(torch.tensor(w[None],dtype=torch.float32));head.bias.copy_(torch.tensor([b],dtype=torch.float32))
    for p in tail.parameters():p.requires_grad_(True)
    for module in tail.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            for p in module.parameters():p.requires_grad_(False)
    net=torch.nn.Sequential(tail,head).eval()
    def predict(indices):
        with torch.inference_mode():return np.concatenate([net(torch.tensor(cache[indices[i:i+32]].copy())).ravel().numpy() for i in range(0,len(indices),32)])
    initial=predict(ix['selection']);np.testing.assert_allclose(initial,probe.decision_function(x[ix['selection']]),atol=1e-3,rtol=1e-4)
    best_loss=nll(y[ix['selection']],initial);best=copy.deepcopy(net.state_dict());best_epoch=0;history=[dict(epoch=0,selection_nll=best_loss)]
    stages['head_fit_seconds']=time.monotonic()-t;t=time.monotonic();rng=np.random.default_rng(args.seed)
    optimizer=torch.optim.AdamW([{'params':[p for p in tail.parameters() if p.requires_grad],'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=1e-4)
    for epoch in range(1,4):
        epoch_start=time.monotonic();order=rng.permutation(ix['train']);total=0.
        for first in range(0,len(order),16):
            ids=order[first:first+16];optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(net(torch.tensor(cache[ids].copy())).ravel(),torch.tensor(y[ids],dtype=torch.float32));loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
        val=nll(y[ix['selection']],predict(ix['selection']));history.append(dict(epoch=epoch,training_nll=total/len(order),selection_nll=val,seconds=time.monotonic()-epoch_start))
        if val<best_loss:best_loss=val;best=copy.deepcopy(net.state_dict());best_epoch=epoch
        print('Epoch',history[-1],flush=True)
    stages['three_epoch_training_seconds']=time.monotonic()-t;t=time.monotonic();net.load_state_dict(best)
    cal=expit(predict(ix['calibration']));cut=float(np.quantile(np.maximum(cal,1-cal),.2));model.classifier=head
    checkpoint=out/'THORAXIS-512-R1-reproduced.pt';torch.save(dict(state_dict=model.state_dict(),input_size=512,cutoff=cut,selected_epoch=best_epoch),checkpoint)
    # Validate exported full model against cached-tail inference on selection only.
    with zipfile.ZipFile(archive) as zf:
        sample=np.stack([letterbox(Image.open(io.BytesIO(zf.read(records[i]['name']))),512) for i in ix['selection'][:2]])
    with torch.inference_mode():np.testing.assert_allclose(model(normalize(sample)).ravel().numpy(),predict(ix['selection'][:2]),atol=2e-4,rtol=1e-4)
    stages['calibration_export_seconds']=time.monotonic()-t
    result=dict(seed=args.seed,split='existing published manifest; no resplit',counts={s:len(v) for s,v in ix.items()},selected_epoch=best_epoch,history=history,selected_C=c,stages=stages,total_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cpu_threads=1,checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest_path),archive_sha256=digest(archive),script_sha256=digest(__file__),test_evaluated=False)
    (out/'training_run.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.evaluate_test:
        from sklearn.metrics import roc_auc_score
        tests=[r for r in manifest['records'] if r['split']=='test'];scores=[]
        with zipfile.ZipFile(archive) as zf,torch.inference_mode():
            for first in range(0,len(tests),16):
                a=np.stack([letterbox(Image.open(io.BytesIO(zf.read(r['name']))),512) for r in tests[first:first+16]]);scores.extend(model(normalize(a)).ravel().tolist())
        labels=np.array([r['label'] for r in tests]);scores=np.array(scores);np.savez_compressed(out/'test_predictions.npz',labels=labels,logits=scores)
        (out/'test_evaluation.json').write_text(json.dumps(dict(n=len(labels),auroc=roc_auc_score(labels,expit(scores)),accuracy=float(np.mean((scores>=0)==labels))),indent=2)+'\n')
    print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
