"""Exploratory severity check on frozen 15-budget models, no fitting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import csv,json
import torch,numpy as np
from scipy.special import expit
from resolution_study import ResolutionCNN
from noise_diagnostics import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/budget'
def main():
 torch.set_num_threads(2);idx=json.loads((ROOT/'configs/resolution_manifest.json').read_text())['indices']['test'];rows=[];levels=[0,.025,.05,.1,.15,.2]
 for r in [28,64]:
  d=np.load(ROOT/'data'/('pneumoniamnist.npz' if r==28 else 'pneumoniamnist_64.npz'));im=d['test_images'][idx].astype(np.float32)/255;y=d['test_labels'][idx].ravel()
  fields={s:np.random.default_rng(s).normal(0,1,im.shape) for s in [7,42,99]}
  for seed in [17,42,83]:
   m=ResolutionCNN();m.load_state_dict(torch.load(ROOT/f'models/resolution/r{r}_s{seed}.pt',weights_only=True));m.eval()
   for noise_seed,field in fields.items():
    for sigma in levels:
     x=torch.tensor(np.clip(im+sigma*field,0,1)[:,None],dtype=torch.float32);z,zero,_=inspect(m,x);p=expit(z)
     rows.append(dict(resolution=r,training_seed=seed,noise_seed=noise_seed,sigma=sigma,accuracy=float(((p>=.5)==y).mean()),zero_hidden_fraction=float(zero.mean()),predicted_positive_fraction=float((p>=.5).mean())))
   print('diagnostic sweep complete',r,seed,flush=True)
 with (OUT/'noise_severity.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for i,r in enumerate([28,64]):
  for j,(metric,title) in enumerate([('accuracy','Accuracy'),('zero_hidden_fraction','All-zero last hidden vector')]):
   ax=axes[i,j]
   for seed in [17,42,83]:
    values=np.array([[next(v[metric] for v in rows if v['resolution']==r and v['training_seed']==seed and v['noise_seed']==ns and v['sigma']==level) for level in levels] for ns in [7,42,99]])
    line=ax.plot(levels,values.mean(0),'o-',label=f'Training seed {seed}')[0];ax.fill_between(levels,values.min(0),values.max(0),color=line.get_color(),alpha=.15)
   ax.set(title=f'{r}px — {title}',xlabel='Noise standard deviation on [0,1]',ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Frozen 15-epoch-budget models; bands: range of three noise realizations');fig.tight_layout();fig.savefig(OUT/'noise_severity.png',dpi=180);fig.savefig(OUT/'noise_severity.pdf')
if __name__=='__main__':main()
