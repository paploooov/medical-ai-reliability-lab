"""Paired preprocessing stress tests; all model weights and thresholds frozen."""
import io,json,csv,zipfile
import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from torchvision.models import mobilenet_v3_small
from original_512 import ROOT,normalize
from prepare_original_xray import digest
from high_resolution_probe import evaluate
from high_resolution_robustness import interval

CONDITIONS=['baseline','bilinear','center_crop','stretch','gamma_0.8','gamma_1.25']

def variant(image,baseline,condition,size=512):
    if condition=='baseline':return np.asarray(baseline).copy()
    if condition.startswith('gamma_'):
        return np.clip(np.rint((np.asarray(baseline,dtype=float)/255)**float(condition.split('_')[1])*255),0,255).astype(np.uint8)
    image=image.convert('L');w,h=image.size
    if condition=='stretch':return np.asarray(image.resize((size,size),Image.Resampling.LANCZOS))
    if condition=='center_crop':
        edge=min(w,h);left=(w-edge)//2;top=(h-edge)//2;return np.asarray(image.crop((left,top,left+edge,top+edge)).resize((size,size),Image.Resampling.LANCZOS))
    if condition!='bilinear':raise ValueError(condition)
    scale=size/max(w,h);im=image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR);out=Image.new('L',(size,size),0);out.paste(im,((size-im.width)//2,(size-im.height)//2));return np.asarray(out)

def main():
    torch.set_num_threads(1);parent=ROOT/'reports/original_512';out=parent/'preprocessing';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/original_preprocessing_stress.json').read_text());base=json.loads((parent/'summary.json').read_text());records=json.loads((parent/'manifest.json').read_text())['records'];old=np.load(parent/'predictions.npz');cache=ROOT/'data/original_xray';checkpoint=cache/'mobile512_research.pt';assert digest(checkpoint)==base['deployment']['checkpoint_sha256']
    archive=cache/'ChestXRay2017.zip';assert digest(archive)==json.loads((ROOT/'configs/original_512.json').read_text())['archive_sha256'];images=np.load(cache/'images_512.npy',mmap_mode='r');saved=torch.load(checkpoint,weights_only=True,map_location='cpu');model=mobilenet_v3_small(weights=None);model.classifier=torch.nn.Linear(576,1);model.load_state_dict(saved['state_dict']);model.eval();cut=saved['cutoff']
    ids=np.r_[old['calibration_indices'],old['test_indices']];y=old['labels'][ids];splits=np.array([records[i]['split'] for i in ids]);groups=np.array([records[i]['group'] for i in ids]);logits={'baseline':old['mobile_512_finetuned'][ids].astype(float)}
    with torch.inference_mode():check=model(normalize(images[ids[:16]])).ravel().numpy()
    np.testing.assert_allclose(check,logits['baseline'][:16],atol=2e-4,rtol=1e-4)
    zf=zipfile.ZipFile(archive);raw=[Image.open(io.BytesIO(zf.read(records[i]['name']))).convert('L') for i in ids];rows=[]
    for condition in CONDITIONS[1:]:
        chunks=[]
        for start in range(0,len(ids),16):
            batch=np.stack([variant(raw[j],images[ids[j]],condition) for j in range(start,min(start+16,len(ids)))])
            with torch.inference_mode():chunks.append(model(normalize(batch)).ravel().numpy().astype(float))
            if start%256==0:print(condition,start,len(ids),flush=True)
        logits[condition]=np.concatenate(chunks)
    for split in ['calibration','test']:
        mask=splits==split
        for c,z in logits.items():rows.append(dict(split=split,condition=c,flip_fraction=float(np.mean((z[mask]>=0)!=(logits['baseline'][mask]>=0))),mean_abs_probability_change=float(np.mean(np.abs(expit(z[mask])-expit(logits['baseline'][mask])))),**evaluate(y[mask],z[mask],cut)))
    with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    def metric(ix,c):
        v=evaluate(y[ix],logits[c][ix],cut)
        return np.array([np.mean((logits[c][ix]>=0)!=(logits['baseline'][ix]>=0)),v['accuracy'],v['sensitivity'],v['specificity'],v['nll'],v['risk_at_80'],v['auroc']])
    test=np.flatnonzero(splits=='test');members=[test[groups[test]==g] for g in np.unique(groups[test])];point={c:metric(test,c) for c in CONDITIONS};boots={c:[] for c in CONDITIONS};rng=np.random.default_rng(20261126)
    for _ in range(2000):
        ix=np.concatenate([members[j] for j in rng.integers(len(members),size=len(members))])
        for c in CONDITIONS:boots[c].append(metric(ix,c))
    boots={c:np.array(v) for c,v in boots.items()};names=['flip_fraction','accuracy','sensitivity','specificity','nll','risk_at_80','auroc'];summary=dict(config=cfg,threshold=cut,results={},contrasts={},provenance=dict(script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'configs/original_preprocessing_stress.json'),checkpoint_sha256=digest(checkpoint),prior_predictions_sha256=digest(parent/'predictions.npz')))
    for c in CONDITIONS:
        summary['results'][c]={k:interval(point[c][j],boots[c][:,j]) for j,k in enumerate(names)}
        if c!='baseline':summary['contrasts'][c]={k:interval(point[c][j]-point['baseline'][j],boots[c][:,j]-boots['baseline'][:,j]) for j,k in enumerate(names) if k!='flip_fraction'}
    summary['primary']=summary['results']['center_crop']['flip_fraction'];(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');np.savez_compressed(out/'predictions.npz',indices=ids,labels=y,splits=splits,groups=groups,**logits);np.savez_compressed(out/'bootstrap.npz',**boots);print('COMPLETE',summary['primary'],flush=True)

if __name__=='__main__':main()
