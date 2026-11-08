"""Equal-coverage ranking diagnostics. No fitting or label-driven tie breaking."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import json,csv,hashlib,subprocess
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/selective_review'
def acceptance_weights(p,coverage):
 p=np.asarray(p)
 if not 0<coverage<=1:raise ValueError('coverage must be in (0, 1]')
 if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('finite probability vector required')
 confidence=np.maximum(p,1-p);target=coverage*len(p)
 boundary=np.sort(confidence)[::-1][int(np.ceil(target))-1]
 above=confidence>boundary;tied=confidence==boundary
 return above.astype(float)+tied*((target-above.sum())/tied.sum())
def risk(y,p,coverage):
 w=acceptance_weights(p,coverage)
 return float(np.dot(w,(np.asarray(p)>=.5)!=np.asarray(y))/w.sum())
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 config=json.loads((ROOT/'configs/selective_review.json').read_text());source=ROOT/'reports/resolution/predictions.npz';a=np.load(source);y=a['labels'];seeds=config['seeds'];rows=[];classrows=[]
 for r in config['resolutions']:
  for seed in seeds:
   for condition in config['conditions']:
    p=a[f'r{r}_s{seed}_{condition}']
    for q in config['curve_coverages']:
     w=acceptance_weights(p,q);wrong=(p>=.5)!=y
     rows.append(dict(resolution=r,seed=seed,condition=condition,coverage=q,risk=float(np.dot(w,wrong)/w.sum()),expected_accepted=float(w.sum())))
     if q==config['primary_coverage']:
      for label in [0,1]:
       mask=y==label;classrows.append(dict(resolution=r,seed=seed,condition=condition,true_label=label,total=int(mask.sum()),expected_accepted=float(w[mask].sum()),expected_accepted_errors=float(np.dot(w[mask],wrong[mask])),expected_rejected=float((1-w[mask]).sum())))
 def contrasts(indices):
  rr={}
  for r in [28,64]:
   for c in ['clean','relative_blur']:rr[r,c]=np.mean([risk(y[indices],a[f'r{r}_s{s}_{c}'][indices],.8) for s in seeds])
  return [rr[64,'clean']-rr[28,'clean'],rr[64,'relative_blur']-rr[28,'relative_blur'],(rr[64,'relative_blur']-rr[64,'clean'])-(rr[28,'relative_blur']-rr[28,'clean'])]
 point=contrasts(np.arange(len(y)));rng=np.random.default_rng(config['bootstrap_seed']);boots=np.array([contrasts(rng.integers(0,len(y),len(y))) for _ in range(config['bootstrap_replicates'])]);ci=np.quantile(boots,[.025,.975],axis=0)
 primary=dict(estimands=['clean_risk_64_minus_28','blur_risk_64_minus_28','difference_in_changes_64_minus_28'],point=point,ci_low=ci[0].tolist(),ci_high=ci[1].tolist(),bootstrap_replicates=len(boots),coverage=.8,scope='Retrospective ranking analysis conditional on six fixed fitted models; exact-pixel groups, not patients.')
 for name,records in [('matched_coverage',rows),('class_breakdown',classrows)]:
  with (OUT/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 (OUT/'primary.json').write_text(json.dumps(primary,indent=2)+'\n')
 (OUT/'provenance.json').write_text(json.dumps(dict(config=config,predictions_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2)+'\n')
 plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
 for ax,c in zip(axes,['clean','relative_blur']):
  for r,color in [(28,'#345995'),(64,'#bd5735')]:
   values=np.array([[next(z['risk'] for z in rows if z['resolution']==r and z['seed']==seed and z['condition']==c and z['coverage']==q) for q in config['curve_coverages']] for seed in seeds])
   ax.plot(config['curve_coverages'],values.mean(0),'o-',color=color,label=f'{r}px mean');ax.fill_between(config['curve_coverages'],values.min(0),values.max(0),color=color,alpha=.13)
  ax.set(title=c.replace('_',' '),xlabel='Retained fraction (retrospectively matched)',ylabel='Errors among retained predictions',ylim=(0,.45));ax.axvline(.8,color='gray',linestyle=':');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Equal coverage: mean and min–max over three seeds, not confidence bands');fig.tight_layout();fig.savefig(OUT/'equal_coverage.png',dpi=180);fig.savefig(OUT/'equal_coverage.pdf');plt.close(fig)
 # Error atlas: first four original-index false positives/negatives, seed 17 at 64px.
 # Choosing by index avoids presenting only the visually most striking cases.
 data=np.load(ROOT/'data/pneumoniamnist_64.npz');images=data['test_images'][a['test_indices']].astype(np.float32)/255
 atlas=[];panels=[]
 for c in ['clean','relative_blur']:
  p=a[f'r64_s17_{c}'];im=images if c=='clean' else gaussian_filter(images,(0,64/28,64/28),mode='reflect')
  for label in [0,1]:
   ix=np.flatnonzero((y==label)&((p>=.5)!=y))[:4]
   for i in ix:panels.append((im[i],c,int(a['test_indices'][i]),int(y[i]),float(p[i])));atlas.append(dict(condition=c,test_index=int(a['test_indices'][i]),label=int(y[i]),probability_pneumonia=float(p[i]),model='64px seed17',selection='first four original-index errors per label and condition'))
 fig,axes=plt.subplots(4,4,figsize=(12,12))
 for ax in axes.flat:ax.axis('off')
 for ax,(im,c,index,label,prob) in zip(axes.flat,panels):ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(f'{c} · index {index}\nlabel={label}, P(pneumonia)={prob:.3f}',fontsize=9)
 fig.suptitle('Selected errors, 64px seed17 — dataset labels, not a diagnostic interpretation\nFirst four errors per class/condition by source index; fewer panels when fewer errors exist');fig.tight_layout();fig.savefig(OUT/'error_atlas.png',dpi=160);plt.close(fig)
 (OUT/'error_atlas.json').write_text(json.dumps(atlas,indent=2)+'\n')
 # Summarize all perturbations: do not hide noise failures.
 raw=list(csv.DictReader((ROOT/'reports/resolution/results.csv').open()))
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,r in zip(axes,[28,64]):
  values=np.array([[float(next(v['accuracy'] for v in raw if int(v['resolution'])==r and int(v['seed'])==s and v['condition']==c)) for c in config['conditions']] for s in seeds]);im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis');ax.set(xticks=range(5),xticklabels=config['conditions'],yticks=range(3),yticklabels=seeds,title=f'{r}px accuracy',ylabel='Training seed');ax.tick_params(axis='x',rotation=45)
  for i in range(3):
   for j in range(5):ax.text(j,i,f'{values[i,j]:.1%}',ha='center',va='center',color='white' if values[i,j]<.6 else 'black')
 fig.tight_layout();fig.savefig(OUT/'all_conditions.png',dpi=180);fig.savefig(OUT/'all_conditions.pdf');plt.close(fig)
 print(json.dumps(primary,indent=2))
 for r in [28,64]:
  for c in ['clean','relative_blur']:print(r,c,np.mean([z['risk'] for z in rows if z['resolution']==r and z['condition']==c and z['coverage']==.8]))
if __name__=='__main__':main()
