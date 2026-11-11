"""Frozen 224px transfer benchmark with explicitly weak external labels."""
import hashlib,json,csv
import numpy as np
import torch
from scipy.special import expit
from torchvision.models import mobilenet_v3_small
from original_512 import ROOT,normalize,vector
from prepare_original_xray import digest
from high_resolution_probe import evaluate
from high_resolution_robustness import interval

def deduplicate(images,labels):
    first={};conflicts=set();duplicates=[]
    for i,image in enumerate(images):
        h=hashlib.sha256(image.tobytes()).hexdigest()
        if h in first:
            j=first[h];duplicates.append([i,j])
            if not np.array_equal(labels[i],labels[j]):conflicts.add(h)
        else:first[h]=i
    return np.array(sorted(i for h,i in first.items() if h not in conflicts)),duplicates,sorted(conflicts)

def main():
    torch.set_num_threads(1);cfg=json.loads((ROOT/'configs/external_chest224.json').read_text());archive=ROOT/'data/external_chest224/chestmnist_224.npz';h=hashlib.md5()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==cfg['archive_md5'];assert archive.stat().st_size==cfg['archive_bytes']
    out=ROOT/'reports/external_chest224';out.mkdir(exist_ok=False);data=np.load(archive);images=data['test_images'];labels=data['test_labels'];assert images.shape==(22433,224,224);assert images.dtype==np.uint8;assert labels.shape==(22433,14);assert np.isin(labels,[0,1]).all()
    retained,dups,conflicts=deduplicate(images,labels);rng=np.random.default_rng(20261127)
    positive=retained[labels[retained,6]==1];normal=retained[labels[retained].sum(1)==0];other=retained[(labels[retained,6]==0)&(labels[retained].sum(1)>0)]
    chosen_normal=np.sort(rng.choice(normal,min(2000,len(normal)),replace=False));chosen_other=np.sort(rng.choice(other,min(1000,len(other)),replace=False));ids=np.r_[positive,chosen_normal,chosen_other];strata=np.array(['pneumonia']*len(positive)+['no_finding']*len(chosen_normal)+['other_abnormal']*len(chosen_other));y=(strata=='pneumonia').astype(int)
    sampling=dict(total_test=len(images),retained_after_dedup=len(retained),duplicate_pairs=dups,conflicting_pixel_hashes=conflicts,available=dict(pneumonia=len(positive),no_finding=len(normal),other_abnormal=len(other)),selected=dict(pneumonia=len(positive),no_finding=len(chosen_normal),other_abnormal=len(chosen_other)),positive_with_additional_tags=int((labels[positive].sum(1)>1).sum()))
    (out/'sampling.json').write_text(json.dumps(sampling,indent=2)+'\n');np.savez_compressed(out/'sample.npz',indices=ids,strata=strata,labels=y,original_labels=labels[ids]);print('Sample',sampling['selected'],flush=True)
    encoder=ROOT/'data/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth';model=mobilenet_v3_small(weights=None);model.load_state_dict(torch.load(encoder,weights_only=True));head=np.load(ROOT/'reports/original_512/probe_heads.npz');model.classifier=torch.nn.Linear(576,1)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor(head['mobile_224_probe_weight'][None],dtype=torch.float32));model.classifier.bias.copy_(torch.tensor([float(head['mobile_224_probe_bias'])],dtype=torch.float32))
    model.eval();prior=np.load(ROOT/'reports/original_512/predictions.npz');check_ids=prior['calibration_indices'][:16];internal=np.load(ROOT/'data/original_xray/images_224.npy',mmap_mode='r')
    with torch.inference_mode():np.testing.assert_allclose(model(normalize(internal[check_ids])).ravel().numpy(),prior['mobile_224_probe'][check_ids],atol=2e-4,rtol=1e-4)
    p=expit(prior['mobile_224_probe'][prior['calibration_indices']]);cut=float(np.quantile(np.maximum(p,1-p),.2));chunks=[]
    for start in range(0,len(ids),32):
        with torch.inference_mode():chunks.append(model(normalize(images[ids[start:start+32]])).ravel().numpy().astype(float))
        if start%512==0:print('Inference',start,len(ids),flush=True)
    z=np.concatenate(chunks);rows=[];results={};boot={};metrics=['auroc','nll','accuracy','risk_at_80','coverage','retained_error'];rng=np.random.default_rng(20261128)
    for control in ['no_finding','other_abnormal']:
        mask=(strata=='pneumonia')|(strata==control);iy=np.flatnonzero(mask);pos=iy[y[iy]==1];neg=iy[y[iy]==0];point=vector(y[iy],z[iy],cut);rows.append(dict(control=control,threshold=cut,**evaluate(y[iy],z[iy],cut)));draws=[]
        for _ in range(2000):
            sample=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];draws.append(vector(y[sample],z[sample],cut))
        boot[control]=np.array(draws);results[control]={k:interval(point[j],boot[control][:,j]) for j,k in enumerate(metrics)}
    with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'predictions.npz',logits=z,labels=y,indices=ids,strata=strata);np.savez_compressed(out/'bootstrap.npz',**boot)
    summary=dict(software=dict(torch=torch.__version__,numpy=np.__version__),config=cfg,sampling=sampling,threshold=cut,results=results,primary=results['no_finding']['auroc'],provenance=dict(script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'configs/external_chest224.json'),archive_md5=h.hexdigest(),encoder_sha256=digest(encoder),original_heads_sha256=digest(ROOT/'reports/original_512/probe_heads.npz'),internal_reference_predictions_sha256=digest(ROOT/'reports/original_512/predictions.npz'),internal_reference_examples_verified=16))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print('COMPLETE',summary['primary'],flush=True)

if __name__=='__main__':main()
