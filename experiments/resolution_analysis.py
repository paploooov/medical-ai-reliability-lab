"""Paired conditional uncertainty and figures for the locked resolution study."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/resolution'
def retained_risk(y,p,cut,indices):
 keep=np.maximum(p[indices],1-p[indices])>=cut
 return float(((p[indices][keep]>=.5)!=y[indices][keep]).mean()) if keep.any() else float('nan')
def main():
 p=np.load(OUT/'predictions.npz');run=json.loads((OUT/'run.json').read_text());config=run['config'];y=p['labels'];n=len(y)
 cuts={(c['resolution'],c['seed']):c['threshold'] for c in run['choices']}
 def contrasts(indices):
  deltas={}
  for r in config['resolutions']:
   for s in config['seeds']:
    deltas[r,s]=retained_risk(y,p[f'r{r}_s{s}_relative_blur'],cuts[r,s],indices)-retained_risk(y,p[f'r{r}_s{s}_clean'],cuts[r,s],indices)
  return [np.mean([deltas[r,s] for s in config['seeds']]) for r in config['resolutions']]+[np.mean([deltas[64,s]-deltas[28,s] for s in config['seeds']])]
 point=contrasts(np.arange(n));rng=np.random.default_rng(config['bootstrap_seed']);draws=np.array([contrasts(rng.integers(0,n,n)) for _ in range(config['bootstrap_replicates'])]);valid=np.isfinite(draws).all(1);ci=np.quantile(draws[valid],[.025,.975],axis=0)
 stats={'estimands':['blur_minus_clean_risk_28','blur_minus_clean_risk_64','difference_in_changes_64_minus_28'],'point':point,'ci_low':ci[0].tolist(),'ci_high':ci[1].tolist(),'bootstrap_replicates':len(draws),'discarded_empty_subset_draws':int((~valid).sum()),'interpretation':'Paired equal-exact-group percentile intervals, conditional on six fitted models. Not patient-level or training-population uncertainty.'}
 (OUT/'primary_analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
 rows=list(csv.DictReader((OUT/'results.csv').open()));history=list(csv.DictReader((OUT/'learning_curves.csv').open()))
 plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
 fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
 for row,r in enumerate([28,64]):
  for col,s in enumerate(config['seeds']):
   h=[v for v in history if int(v['resolution'])==r and int(v['seed'])==s];ax=axes[row,col]
   ax.plot([int(v['epoch']) for v in h],[float(v['training_online_bce']) for v in h],label='Training (online)');ax.plot([int(v['epoch']) for v in h],[float(v['selection_bce']) for v in h],label='Model selection');ax.set(title=f'{r}×{r}, seed {s}',xlabel='Epoch',ylabel='Binary cross-entropy');ax.grid(alpha=.2)
 axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);fig.savefig(OUT/'learning_curves.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,metric,title in zip(axes,['retained_error_rate','coverage'],['Errors among accepted predictions','Fraction of predictions accepted']):
  for r,color in [(28,'#345995'),(64,'#c45a36')]:
   for s in config['seeds']:
    vals=[float(next(v[metric] for v in rows if int(v['resolution'])==r and int(v['seed'])==s and v['condition']==c)) for c in ['clean','relative_blur']]
    ax.plot([0,1],vals,'o-',color=color,alpha=.7,label=f'{r}px' if s==17 else None)
  ax.set(xticks=[0,1],xticklabels=['Clean','Width-relative blur'],title=title,ylim=(0,1));ax.grid(alpha=.2);ax.legend()
 fig.suptitle('One line per seed; separate clean-calibration cutoffs');fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=180);fig.savefig(OUT/'risk_coverage.pdf');plt.close(fig)
 lines=['# Resolution and rejection — exploratory study','', 'A duplicate-aware sensitivity analysis, separate from the historical official-split pilots. No exact pixel group crosses fitting, model selection, calibration or testing. Unknown patient identities remain a limitation.','', '## Design','', '4,682 fitting cases; 262 model-selection cases; 262 calibration cases; 621 unique test representatives. Six runs: two resolutions × three seeds × 15 epochs. Identical adaptive-pooling architecture and parameter count. Thresholds target 80% clean calibration coverage; actual test coverage may differ.','', '## Clean versus width-relative blur','', '| Resolution | Condition | Accuracy, mean | Retained error, mean | Coverage, mean |','|---|---|---:|---:|---:|']
 for r in [28,64]:
  for c in ['clean','relative_blur']:
   sel=[v for v in rows if int(v['resolution'])==r and v['condition']==c]
   vals=[np.mean([float(v[k]) for v in sel]) for k in ['accuracy','retained_error_rate','coverage']]
   lines.append(f'| {r} | {c} | '+ ' | '.join(f'{v:.1%}' for v in vals)+' |')
 lines+=['','## Primary contrast','',f'The difference in blur-minus-clean retained error risk (64 minus 28), averaged over matched seeds, is **{point[2]*100:.2f} percentage points**, with a paired conditional 95% percentile interval of **[{ci[0,2]*100:.2f}, {ci[1,2]*100:.2f}] points**. Negative values favor 64 pixels for this particular risk change, not necessarily for clean accuracy or coverage.', '',f'Bootstrap: {len(draws)} equal-group paired replicates; {int((~valid).sum())} discarded for empty retained subsets. Seeds are held fixed; patients are not known. This interval does not include retraining or calibration-set sampling uncertainty.', '', 'The accepted cohorts may differ across conditions and models. Interpret risk jointly with coverage; greater rejection can reduce observed risk. A single predefined blur contrast does not demonstrate general robustness. Secondary noise, contrast and fixed-pixel blur results are in results.csv, not independent confirmations.', '', '![Risk and coverage](../../reports/resolution/risk_coverage.png)','','![Learning curves](../../reports/resolution/learning_curves.png)','','## Reproduce and inspect','', '`configs/resolution_manifest.json` records every exact group and selected index. `configs/resolution_study.json` and [the protocol](RESOLUTION_PROTOCOL.md) were committed before fitting. Run `python experiments/resolution_study.py` then `python experiments/resolution_analysis.py` after installing base dependencies and CPU PyTorch 2.5.1. Training replaces this study’s outputs, not older studies. Full histories, model hashes, timings, software environment and predictions are in reports/resolution; weights are in models/resolution. PDF figures are available beside PNG previews.','','## Limits and next checks','','This known pediatric benchmark is not an external test. Exact deduplication is not patient separation. Fifteen epochs are a fixed experimental budget rather than a proven optimum. Equal epochs imply different computational cost. Spatial receptive fields still differ relative to image width, so this compares resolution-specific pipelines, not a perfectly isolated information-content intervention. Gaussian blur is synthetic; physical acquisition degradation and clinical usefulness remain untested. Calibration uncertainty with only 262 cases may be substantial. A further independent dataset and clinical input are required before medical conclusions.']
 (ROOT/'docs/research/RESOLUTION_STUDY.md').write_text('\n'.join(lines)+'\n');print(json.dumps(stats,indent=2));print('\n'.join(lines[8:16]))
if __name__=='__main__':main()
