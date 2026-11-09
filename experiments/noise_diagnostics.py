"""Verify noise preprocessing and inspect zero final-hidden vectors without fitting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import argparse,json,csv,hashlib
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from resolution_study import ResolutionCNN
ROOT=Path(__file__).resolve().parents[1]
def inspect(model,x):
 logits=[];zero=[];fractions={};preacts=[]
 with torch.no_grad():
  for batch in x.split(128):
   z=batch
   for i,layer in enumerate(model.layers):
    if i==len(model.layers)-1:
     zero.append((z==0).all(1).numpy())
    z=layer(z)
    if isinstance(layer,nn.ReLU):fractions.setdefault(str(i),[]).append((z==0).flatten(1).float().mean(1).numpy())
   logits.append(z.squeeze(1).numpy())
 return np.concatenate(logits),np.concatenate(zero),{i:np.concatenate(v) for i,v in fractions.items()}
def main(report_dir,model_dir,tag,output_dir=ROOT/"reports/budget"):
 torch.set_num_threads(2);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 choices=json.loads((report_dir/'run.json').read_text())['choices']
 manifest=json.loads((ROOT/'configs/resolution_manifest.json').read_text());indices=manifest['indices']['test'];saved=np.load(report_dir/'predictions.npz');rows=[];arrays={};processing=[]
 for r in sorted({c['resolution'] for c in choices}):
  data=np.load(ROOT/'data'/('pneumoniamnist.npz' if r==28 else 'pneumoniamnist_64.npz'));clean=data['test_images'][indices].astype(np.float32)/255
  noise=np.random.default_rng(42).normal(0,.1,clean.shape);before=clean+noise;noisy=np.clip(before,0,1)
  processing.append(dict(resolution=r,clean_min=float(clean.min()),clean_max=float(clean.max()),noise_mean=float(noise.mean()),noise_sd=float(noise.std()),clipped_fraction=float(((before<0)|(before>1)).mean()),rms_change=float(np.sqrt(np.mean((noisy-clean)**2)))))
  for seed in [17,42,83]:
   model=ResolutionCNN();path=model_dir/f'r{r}_s{seed}.pt';model.load_state_dict(torch.load(path,weights_only=True));model.eval();bias=float(model.layers[-1].bias.detach()[0])
   for condition,im in [('clean',clean),('noise',noisy)]:
    x=torch.tensor(im[:,None],dtype=torch.float32);z,allzero,fracs=inspect(model,x);prob=expit(z)
    np.testing.assert_allclose(prob,saved[f'r{r}_s{seed}_{condition}'],atol=1e-7,rtol=1e-6)
    if allzero.any():np.testing.assert_allclose(z[allzero],bias,atol=1e-7)
    arrays[f'r{r}_s{seed}_{condition}_zero_hidden']=allzero;arrays[f'r{r}_s{seed}_{condition}_logits']=z
    for label in [0,1]:
     keep=saved['labels']==label
     rows.append(dict(budget=tag,resolution=r,seed=seed,condition=condition,true_label=label,cases=int(keep.sum()),all_zero_hidden=int(allzero[keep].sum()),predicted_positive=int((prob[keep]>=.5).sum()),logit_sd=float(z[keep].std()),final_bias=bias,bias_probability=float(expit(bias)),**{f'relu_{i}_zero_fraction':float(a[keep].mean()) for i,a in fracs.items()}))
 with (out/f'noise_diagnostics_{tag}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(out/f'noise_diagnostics_{tag}.npz',labels=saved['labels'],test_indices=indices,**arrays)
 (out/f'noise_processing_{tag}.json').write_text(json.dumps(processing,indent=2)+'\n')
 for r in sorted({c['resolution'] for c in choices}):
  for seed in [17,42,83]:
   a=[v for v in rows if v['resolution']==r and v['seed']==seed and v['condition']=='noise'];print(tag,r,seed,'zero hidden',sum(v['all_zero_hidden'] for v in a),'/',len(indices),'bias P',a[0]['bias_probability'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,default=ROOT/'reports/resolution');p.add_argument('--models',type=Path,default=ROOT/'models/resolution');p.add_argument('--tag',default='15');p.add_argument('--output',type=Path,default=ROOT/'reports/budget');args=p.parse_args();main(args.reports,args.models,args.tag,args.output)
