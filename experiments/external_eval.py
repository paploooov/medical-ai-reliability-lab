"""Frozen-model external evaluation on the RSNA sample (see reports/external/PROTOCOL.md).

No training, no new thresholds: weights and acceptance cutoffs come from run.json.
Run after experiments/external_data.py:  python experiments/external_eval.py
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import csv,json,hashlib,subprocess,sys
import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/external'
sys.path.insert(0,str(ROOT/'experiments'))
BINS=15;COVERAGES=np.round(np.arange(.05,1.0001,.05),2);BOOT=2000;BOOT_SEED=2026
CONDITIONS=['clean','noise','relative_blur','contrast']
ARMS={'control_28':('budget',28),'augmented_28':('augmentation',28),'control_64':('budget',64)}
SEEDS=[17,42,83]

def perturb(images,condition,res):
 """Repository perturbations on float images in [0,1] (N,H,W)."""
 if condition=='clean':return images
 if condition=='noise':return np.clip(images+np.random.default_rng(42).normal(0,.1,images.shape),0,1)
 if condition=='relative_blur':return gaussian_filter(images,(0,res/28,res/28),mode='reflect')
 if condition=='contrast':return np.clip(.5+.5*(images-.5),0,1)
 raise ValueError(condition)

def auroc(y,p):
 y=np.asarray(y).astype(bool);n1=y.sum();n0=len(y)-n1
 if n1==0 or n0==0:return float('nan')
 r=rankdata(p);return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))

def reliability(y,p,bins=BINS):
 """Top-label confidence vs accuracy in equal-width bins on [0,1]."""
 conf=np.maximum(p,1-p);correct=((p>=.5)==y).astype(float)
 idx=np.clip(np.ceil(conf*bins).astype(int)-1,0,bins-1)
 n=np.bincount(idx,minlength=bins).astype(float)
 acc=np.divide(np.bincount(idx,correct,bins),n,out=np.full(bins,np.nan),where=n>0)
 mc=np.divide(np.bincount(idx,conf,bins),n,out=np.full(bins,np.nan),where=n>0)
 return n,acc,mc

def ece(y,p,bins=BINS):
 n,acc,mc=reliability(y,p,bins);m=n>0
 return float(np.sum(n[m]*np.abs(acc[m]-mc[m]))/n.sum())

def risk_at(y,p,coverage):
 """Error among the most confident `coverage` share; ties shared fractionally (as selective_review.risk)."""
 conf=np.maximum(p,1-p);wrong=((p>=.5)!=y).astype(float);target=coverage*len(p)
 b=np.sort(conf)[::-1][int(np.ceil(target))-1];above=conf>b;tied=conf==b
 w=above+tied*((target-above.sum())/tied.sum());return float(w@wrong/w.sum())

def risk_coverage(y,p,coverages=COVERAGES):return np.array([risk_at(y,p,c) for c in coverages])

def summarize(y,p,cut):
 pred=p>=.5;wrong=pred!=y;keep=np.maximum(p,1-p)>=cut;pos=y==1
 rc=risk_coverage(y,p)
 return dict(n=int(len(y)),prevalence=float(pos.mean()),accuracy=float(1-wrong.mean()),auroc=auroc(y,p),
  sensitivity=float(pred[pos].mean()) if pos.any() else float('nan'),specificity=float((~pred[~pos]).mean()) if (~pos).any() else float('nan'),
  ece=ece(y,p),mean_confidence=float(np.maximum(p,1-p).mean()),coverage=float(keep.mean()),
  retained_error=float(wrong[keep].mean()) if keep.any() else float('nan'),predicted_positive=float(pred.mean()),
  aurc=float(np.trapezoid(rc,COVERAGES)),risk_at_80=float(risk_at(y,p,.8)))

BOOT_KEYS=['accuracy','auroc','ece','coverage','retained_error','sensitivity','specificity']
def fast(y,p,cut):
 pred=p>=.5;wrong=pred!=y;keep=np.maximum(p,1-p)>=cut;pos=y==1
 return [1-wrong.mean(),auroc(y,p),ece(y,p),keep.mean(),wrong[keep].mean() if keep.any() else np.nan,pred[pos].mean(),(~pred[~pos]).mean()]

def load_models():
 import torch
 from resolution_study import ResolutionCNN
 out={}
 for arm,(folder,res) in ARMS.items():
  run=json.loads((ROOT/'reports'/folder/'run.json').read_text())
  for c in run['choices']:
   if c['resolution']!=res:continue
   path=ROOT/'models'/folder/f'r{res}_s{c["seed"]}.pt'
   assert hashlib.sha256(path.read_bytes()).hexdigest()==c['weights_sha256'],path
   m=ResolutionCNN();m.load_state_dict(torch.load(path,weights_only=True));m.eval()
   out[arm,c['seed']]=(m,c['threshold'],c['weights_sha256'])
 assert len(out)==9;return out

def predict(model,images):
 import torch
 from resolution_study import logits
 return expit(logits(model,torch.tensor(images[:,None],dtype=torch.float32)))

def main():
 import torch;torch.set_num_threads(2)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 OUT.mkdir(parents=True,exist_ok=True);models=load_models()
 manifest=json.loads((ROOT/'configs/resolution_manifest.json').read_text());ti=manifest['indices']['test']
 data={}
 for res in [28,64]:
  d=np.load(ROOT/'data'/('pneumoniamnist.npz' if res==28 else 'pneumoniamnist_64.npz'))
  e=np.load(ROOT/f'data/external/rsna_{res}.npz')
  data['internal',res]=(d['test_images'][ti].astype(np.float32)/255,d['test_labels'][ti].ravel().astype(int))
  prim=e['arm']=='primary'
  data['external',res]=(e['images'][prim].astype(np.float32)/255,e['labels'][prim].astype(int))
  data['external_secondary',res]=(e['images'][~prim].astype(np.float32)/255,e['labels'][~prim].astype(int))
 sample=[r for r in csv.DictReader((ROOT/'configs/external_rsna_sample.csv').open()) if r['arm']=='primary']
 adult=np.array([int(r['age'])>=18 for r in sample]);view=np.array([r['view'] for r in sample])
 pred={};rows=[]
 for (arm,seed),(m,cut,_) in models.items():
  res=ARMS[arm][1]
  for ds in ['internal','external','external_secondary']:
   x,y=data[ds,res]
   for c in CONDITIONS:
    p=predict(m,perturb(x,c,res));pred[f'{ds}|{arm}|s{seed}|{c}']=p
    rows.append(dict(dataset=ds,arm=arm,seed=seed,condition=c,threshold=cut,**summarize(y,p,cut)))
    if ds=='external':
     for name,mask in [('external_adult',adult),('external_AP',view=='AP'),('external_PA',view=='PA')]:
      rows.append(dict(dataset=name,arm=arm,seed=seed,condition=c,threshold=cut,**summarize(y[mask],p[mask],cut)))
 # Consistency with stored repository predictions (internal test, frozen models).
 old=np.load(ROOT/'reports/augmentation/comparison_predictions.npz');budget=np.load(ROOT/'reports/budget/predictions.npz');checked=0
 for (arm,seed) in models:
  for c in CONDITIONS:
   if arm in ['control_28','augmented_28']:
    key=f"{'control' if arm=='control_28' else 'augmentation'}_s{seed}_{c}"
    np.testing.assert_allclose(pred[f'internal|{arm}|s{seed}|{c}'],old[key],atol=1e-5);checked+=1
   elif f'r64_s{seed}_{c}' in budget.files:
    np.testing.assert_allclose(pred[f'internal|{arm}|s{seed}|{c}'],budget[f'r64_s{seed}_{c}'],atol=1e-5);checked+=1
 with (OUT/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(OUT/'predictions.npz',**{k.replace('|','__'):v.astype(np.float32) for k,v in pred.items()},**{f'labels__{ds}':data[ds,28][1] for ds in ['internal','external','external_secondary']})
 # Paired bootstrap over images, conditional on fixed models and cutoffs.
 rng=np.random.default_rng(BOOT_SEED);summary={'protocol':'reports/external/PROTOCOL.md','bootstrap':dict(replicates=BOOT,seed=BOOT_SEED,unit='image (one per RSNA patientId); internal: exact-pixel group representative',scope='conditional on fixed models and cutoffs; percentile 95% intervals; mean over seeds 17/42/83'),'results':{}}
 for ds in ['internal','external']:
  y=data[ds,28][1];n=len(y);draws=[rng.integers(0,n,n) for _ in range(BOOT)]
  for c in CONDITIONS:
   point={arm:np.mean([fast(y,pred[f'{ds}|{arm}|s{s}|{c}'],models[arm,s][1]) for s in SEEDS],0) for arm in ARMS}
   boots={arm:np.empty((BOOT,len(BOOT_KEYS))) for arm in ARMS}
   for b,ix in enumerate(draws):
    for arm in ARMS:boots[arm][b]=np.mean([fast(y[ix],pred[f'{ds}|{arm}|s{s}|{c}'][ix],models[arm,s][1]) for s in SEEDS],0)
   entry={}
   for arm in ARMS:
    lo,hi=np.nanquantile(boots[arm],[.025,.975],axis=0)
    entry[arm]={k:dict(point=float(point[arm][i]),ci=[float(lo[i]),float(hi[i])]) for i,k in enumerate(BOOT_KEYS)}
   diff=boots['augmented_28']-boots['control_28'];lo,hi=np.nanquantile(diff,[.025,.975],axis=0);dp=point['augmented_28']-point['control_28']
   entry['augmented_minus_control_28']={k:dict(point=float(dp[i]),ci=[float(lo[i]),float(hi[i])]) for i,k in enumerate(BOOT_KEYS)}
   summary['results'][f'{ds}|{c}']=entry
 summary['consistency']=dict(internal_predictions_matching_repository=checked,weights_verified=len(models))
 summary['provenance']=dict(sample_csv_sha256=hashlib.sha256((ROOT/'configs/external_rsna_sample.csv').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch=torch.__version__)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Figures.
 plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
 colors={'control_28':'#345995','augmented_28':'#bd5735','control_64':'#6a994e'}
 labels={'control_28':'Control 28px','augmented_28':'Noise-augmented 28px','control_64':'Control 64px'}
 fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
 for ax,ds,title in zip(axes,['internal','external'],['Internal test (PneumoniaMNIST, children)','External test (RSNA, adults)']):
  y=data[ds,28][1];ax.plot([0,1],[0,1],':',color='gray')
  for arm in ['control_28','augmented_28']:
   p=np.concatenate([pred[f'{ds}|{arm}|s{s}|clean'] for s in SEEDS]);yy=np.tile(y,3);n,acc,mc=reliability(yy,p);m=n>=10
   e=np.mean([ece(y,pred[f'{ds}|{arm}|s{s}|clean']) for s in SEEDS])
   ax.plot(mc[m],acc[m],'o-',color=colors[arm],label=f'{labels[arm]} (ECE {e:.3f})')
  ax.set(title=title,xlabel='Confidence (top label)',xlim=(.45,1.02),ylim=(0,1.02));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
 axes[0].set_ylabel('Accuracy in bin')
 fig.suptitle('Reliability, clean images: 15 equal-width bins, three seeds pooled, bins with ≥10 cases');fig.tight_layout();fig.savefig(OUT/'reliability.png',dpi=180);fig.savefig(OUT/'reliability.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
 for ax,c in zip(axes,['clean','noise']):
  for arm in ['control_28','augmented_28']:
   for ds,style in [('internal','--'),('external','-')]:
    y=data[ds,28][1];v=np.array([risk_coverage(y,pred[f'{ds}|{arm}|s{s}|{c}']) for s in SEEDS])
    ax.plot(COVERAGES,v.mean(0),style,color=colors[arm],label=f'{labels[arm]}, {ds}');ax.fill_between(COVERAGES,v.min(0),v.max(0),color=colors[arm],alpha=.1)
  ax.set(title=f'{c} images',xlabel='Coverage (share of most confident cases retained)',ylim=(0,.7));ax.axvline(.8,color='gray',linestyle=':');ax.grid(alpha=.2)
 axes[0].set_ylabel('Error among retained cases');axes[0].legend(fontsize=8)
 fig.suptitle('Risk–coverage, internal (dashed) vs external (solid); mean and seed range');fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=180);fig.savefig(OUT/'risk_coverage.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.4))
 for ax,metric,title in zip(axes,['accuracy','retained_error','coverage'],['Accuracy','Error among accepted (fixed cutoff)','Accepted fraction (fixed cutoff)']):
  for k,arm in enumerate(['control_28','augmented_28']):
   for j,ds in enumerate(['internal','external']):
    v=np.array([[next(r[metric] for r in rows if r['dataset']==ds and r['arm']==arm and r['seed']==s and r['condition']==c) for c in CONDITIONS] for s in SEEDS])
    x=np.arange(4)+(-.27+.18*(2*j+k));ax.errorbar(x,v.mean(0),yerr=[v.mean(0)-v.min(0),v.max(0)-v.mean(0)],fmt='o' if ds=='internal' else 's',mfc='white' if ds=='internal' else colors[arm],color=colors[arm],capsize=2,label=f'{labels[arm]}, {ds}')
  ax.set(title=title,xticks=range(4),xticklabels=CONDITIONS,ylim=(0,1));ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.suptitle('All conditions: mean and seed range (not confidence intervals)');fig.tight_layout();fig.savefig(OUT/'conditions.png',dpi=180);plt.close(fig)
 for ds in ['internal','external','external_secondary','external_adult']:
  for arm in ARMS:
   r=[x for x in rows if x['dataset']==ds and x['arm']==arm and x['condition']=='clean']
   print(ds,arm,{k:round(float(np.mean([x[k] for x in r])),4) for k in ['accuracy','auroc','ece','coverage','retained_error','sensitivity','specificity','predicted_positive']})
 print('consistency',summary['consistency'])
if __name__=='__main__':main()
