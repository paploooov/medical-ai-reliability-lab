"""Class-specific errors and measured CPU inference cost for the new baseline."""
import json,time,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
import torch
from torchvision.models import mobilenet_v3_small,resnet18
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from high_resolution_robustness import interval

def class_metrics(y,z,cut):
    p=expit(z);pred=p>=.5;keep=np.maximum(p,1-p)>=cut
    return np.array([pred[y==1].mean(),(~pred[y==0]).mean(),keep[y==1].mean(),keep[y==0].mean(),np.mean(pred[(y==1)&keep]==0) if np.any((y==1)&keep) else np.nan,np.mean(pred[(y==0)&keep]==1) if np.any((y==0)&keep) else np.nan])

def main():
    out=ROOT/'reports/original_512';cache=ROOT/'data/original_xray';s=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text());saved=np.load(out/'predictions.npz');test=saved['test_indices'];y=saved['labels'][test]
    z={a:saved[a][test] for a in s['results']};cut={}
    for a in z:
        p=expit(saved[a][saved['calibration_indices']]);cut[a]=float(np.quantile(np.maximum(p,1-p),.2))
    groups=np.array([manifest['records'][i]['group'] for i in test]);members=[np.flatnonzero(groups==g) for g in np.unique(groups)];rng=np.random.default_rng(20261122);boots={a:[] for a in z}
    for _ in range(2000):
        ids=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))])
        for a in z:boots[a].append(class_metrics(y[ids],z[a][ids],cut[a]))
    metrics=['sensitivity','specificity','positive_coverage','negative_coverage','accepted_positive_error','accepted_negative_error'];result={}
    for a in z:
        point=class_metrics(y,z[a],cut[a]);b=np.array(boots[a]);pred=z[a]>=0
        result[a]=dict(confusion=dict(tn=int(((y==0)&~pred).sum()),fp=int(((y==0)&pred).sum()),fn=int(((y==1)&~pred).sum()),tp=int(((y==1)&pred).sum())),metrics={m:interval(point[i],b[:,i]) for i,m in enumerate(metrics)})
    (out/'class_breakdown.json').write_text(json.dumps(dict(scope='Secondary, conditional grouped bootstrap; no multiplicity adjustment',results=result),indent=2,allow_nan=False)+'\n')
    torch.set_num_threads(1);timings={}
    for name,size in [('mobile_224_probe',224),('mobile_512_finetuned',512),('resnet18_224_probe',224)]:
        if name.startswith('mobile'):
            model=mobilenet_v3_small(weights=None);model.classifier=torch.nn.Linear(576,1)
            if size==512:model.load_state_dict(torch.load(cache/'mobile512_research.pt',weights_only=True)['state_dict'])
        else:model=resnet18(weights=None);model.fc=torch.nn.Linear(512,1)
        model.eval();x=torch.zeros(1,3,size,size);times=[]
        with torch.inference_mode():
            for _ in range(5):model(x)
            for _ in range(30):
                start=time.perf_counter();model(x);times.append((time.perf_counter()-start)*1000)
        timings[name]=dict(median_ms=float(np.median(times)),p90_ms=float(np.quantile(times,.9)),repeats=len(times),parameters=sum(p.numel() for p in model.parameters()))
    (out/'runtime.json').write_text(json.dumps(dict(scope='Linux CPU, torch threads=1, batch=1, five warmups; normalized synthetic tensor, excludes image decoding/resize/model loading. Non-selected encoder weights randomly initialized for architecture-only timing. Not iPhone latency.',models=timings,checkpoint_bytes=(cache/'mobile512_research.pt').stat().st_size,torchscript_bytes=(cache/'mobile512_research.ts').stat().st_size),indent=2)+'\n');print('Class breakdown and CPU latency saved',flush=True)

if __name__=='__main__':main()
