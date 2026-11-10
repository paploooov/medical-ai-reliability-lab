"""Combine historical and additional paired seeds without selecting runs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from external_eval import risk_coverage, COVERAGES

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[17,42,83,123,2026]
CONDITIONS=['clean','noise','relative_blur','contrast']


def seed_interval(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('At least two finite paired seed contrasts required')
    mean=float(values.mean());half=float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean,ci=[mean-half,mean+half],minimum=float(values.min()),maximum=float(values.max()),
                n=len(values),scope='Paired-seed t interval on this fixed test split; small n, approximate normality assumption')


def retained_error(y,p,threshold,indices=None):
    if indices is not None:y,p=y[indices],p[indices]
    keep=np.maximum(p,1-p)>=threshold
    return float(((p>=.5)!=y)[keep].mean()) if keep.any() else float('nan')


def main(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    predictions={};cutoffs={};rows=[];sources={};y=None
    for arm in ['control','augmentation']:
        for historical,folder in [(True,'budget' if arm=='control' else 'augmentation'),(False,f'five_seed/{arm}')]:
            directory=ROOT/'reports'/folder;run=json.loads((directory/'run.json').read_text())
            source=np.load(directory/'predictions.npz')
            if y is None:y=source['labels']
            else:np.testing.assert_array_equal(y,source['labels'])
            np.testing.assert_array_equal(source['test_indices'],json.loads((ROOT/'configs/resolution_manifest.json').read_text())['indices']['test'])
            for file in ['run.json','predictions.npz','results.csv']:
                path=directory/file;sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            seeds=SEEDS[:3] if historical else SEEDS[3:]
            for seed in seeds:
                choice=next(c for c in run['choices'] if c['resolution']==28 and c['seed']==seed)
                weight=ROOT/'models'/folder/f'r28_s{seed}.pt'
                assert hashlib.sha256(weight.read_bytes()).hexdigest()==choice['weights_sha256']
                cutoffs[arm,seed]=choice['threshold']
                for condition in CONDITIONS:
                    predictions[arm,seed,condition]=source[f'r28_s{seed}_{condition}']
            with (directory/'results.csv').open() as f:
                rows.extend(dict(arm=arm,**r) for r in csv.DictReader(f)
                            if int(r['resolution'])==28 and int(r['seed']) in seeds and r['condition'] in CONDITIONS)
    point=[];seed_rows=[]
    for seed in SEEDS:
        risks={(arm,c):retained_error(y,predictions[arm,seed,c],cutoffs[arm,seed])
               for arm in ['control','augmentation'] for c in ['clean','noise']}
        contrast=(risks['augmentation','noise']-risks['augmentation','clean'])-(risks['control','noise']-risks['control','clean'])
        point.append(contrast)
        seed_rows.append(dict(seed=seed,historical=seed in SEEDS[:3],primary_contrast=contrast,
                              **{f'{a}_{c}_risk':v for (a,c),v in risks.items()}))
    # Image resampling is separate from seed uncertainty. Same IDs for all models.
    rng=np.random.default_rng(20261112);boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y));values=[]
        for seed in SEEDS:
            risks={(a,c):retained_error(y,predictions[a,seed,c],cutoffs[a,seed],ix)
                   for a in ['control','augmentation'] for c in ['clean','noise']}
            values.append((risks['augmentation','noise']-risks['augmentation','clean'])-(risks['control','noise']-risks['control','clean']))
        boot.append(np.mean(values))
    boot=np.array(boot);assert np.isfinite(boot).all()
    summary=dict(seeds=SEEDS,per_seed=seed_rows,primary=seed_interval(point),
                 historical_mean=float(np.mean(point[:3])),additional_mean=float(np.mean(point[3:])),
                 fixed_model_image_bootstrap=dict(mean=float(np.mean(point)),ci=np.quantile(boot,[.025,.975]).tolist(),
                  replicates=2000,seed=20261112,scope='Paired test-representative bootstrap conditional on ten models and cutoffs; not combined training and image uncertainty'),
                 descriptive={},source_sha256=sources)
    for arm in ['control','augmentation']:
        for c in CONDITIONS:
            selected=[r for r in rows if r['arm']==arm and r['condition']==c]
            assert len(selected)==5
            summary['descriptive'][f'{arm}|{c}']={k:dict(mean=float(np.mean([float(r[k]) for r in selected])),
                  minimum=min(float(r[k]) for r in selected),maximum=max(float(r[k]) for r in selected))
                  for k in ['accuracy','auc','coverage','retained_error_rate']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for name,items in [('per_seed',seed_rows),('results',rows)]:
        with (output/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=items[0]);w.writeheader();w.writerows(items)
    np.savez_compressed(output/'image_bootstrap.npz',primary=boot)
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    for ax,c in zip(axes.flat,CONDITIONS):
        for arm,color in [('control','#345995'),('augmentation','#bd5735')]:
            curves=np.array([risk_coverage(y,predictions[arm,seed,c]) for seed in SEEDS])
            ax.plot(COVERAGES,curves.mean(axis=0),color=color,label=arm)
            ax.fill_between(COVERAGES,curves.min(axis=0),curves.max(axis=0),color=color,alpha=.18)
        ax.set(title=c,xlabel='Coverage',ylabel='Error among retained cases',ylim=(0,.8));ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Five paired seeds, internal test\nMean and min–max over seeds; shaded ranges are not confidence intervals')
    fig.tight_layout();fig.savefig(output/'risk_coverage.png',dpi=180);fig.savefig(output/'risk_coverage.pdf');plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['primary','historical_mean','additional_mean','fixed_model_image_bootstrap']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/five_seed/analysis')
    main(parser.parse_args().output)
