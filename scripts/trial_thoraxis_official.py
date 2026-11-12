"""One official seed-17 technical fit on verified cache; no test inference path."""
import copy,hashlib,io,json,platform,resource,time,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from prepare_original_xray import digest,letterbox
ROOT=Path(__file__).resolve().parents[1]

def normalize(a):
    x=torch.tensor(np.asarray(a).copy(),dtype=torch.float32)[:,None].repeat(1,3,1,1)/255
    return (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]

def nll(y,z):return float(np.mean(np.logaddexp(0,z)-y*z))

def main():
    from prepare_thoraxis_cache import verify, records as source_records, encoder, ARCHIVE
    start=time.monotonic(); stages={}; torch.set_num_threads(1); torch.manual_seed(17); np.random.seed(17)
    out=ROOT/'data/thoraxis_source_shift/official_seed17'
    out.mkdir(parents=True,exist_ok=False)
    _, arrays=verify()
    manifest_path=ROOT/'reports/original_512/manifest.json'; archive=ARCHIVE
    original=source_records(); ids=np.array([i for i,r in enumerate(original) if r['split']!='test'])
    records=[original[i] for i in ids]
    assert len(records)==4753 and all(r['split'] in ('train','selection','calibration') for r in records)
    # Materialize only development rows: the medical model cannot index test rows.
    cache=arrays['prefix.npy'][ids].copy(); features=arrays['pooled.npy'][ids].copy()
    del arrays
    y=np.array([r['label'] for r in records]); ix={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ('train','selection','calibration')}
    model=encoder(); tail=torch.nn.Sequential(model.features[10:],torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
    stages['cache_verification_and_load_seconds']=time.monotonic()-start; t=time.monotonic()
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
        assert all(0<=i<len(records) and records[i]['split']!='test' for i in indices)
        with torch.inference_mode():return np.concatenate([net(torch.tensor(cache[indices[i:i+32]].copy())).ravel().numpy() for i in range(0,len(indices),32)])
    initial=predict(ix['selection']);np.testing.assert_allclose(initial,probe.decision_function(x[ix['selection']]),atol=1e-3,rtol=1e-4)
    best_loss=nll(y[ix['selection']],initial);best=copy.deepcopy(net.state_dict());best_epoch=0;history=[dict(epoch=0,selection_nll=best_loss)]
    stages['head_fit_seconds']=time.monotonic()-t;t=time.monotonic();rng=np.random.default_rng(17)
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
    result=dict(seed=17,split='existing published manifest; no resplit',counts={s:len(v) for s,v in ix.items()},selected_epoch=best_epoch,history=history,selected_C=c,stages=stages,total_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cpu_threads=1,checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest_path),archive_sha256=digest(archive),script_sha256=digest(__file__),test_evaluated=False)
    (out/'training_run.json').write_text(json.dumps(result,indent=2)+'\n')
    result['cache_manifest_sha256']=digest(ROOT/'configs/thoraxis_lanczos_cache.json')
    result['medical_test_predictions']=0
    result['development_image_ids_sha256']=hashlib.sha256(json.dumps([r['name'] for r in records],separators=(',',':')).encode()).hexdigest()
    result['git_commit_before_run']=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (out/'training_run.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/original_512/preparation_seed17.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
