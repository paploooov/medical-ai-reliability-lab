"""Post-hoc AP/PA standardization with a paired, stratified image bootstrap."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from external_eval import auroc

ROOT = Path(__file__).resolve().parents[1]


def standardized_metrics(cell_rates):
    """Cell rates (..., label, [correct, accepted, accepted_wrong])."""
    rates = np.asarray(cell_rates, dtype=float)
    if rates.shape[-2:] != (2, 3):
        raise ValueError('Expected two label cells and three rates')
    average = rates.mean(axis=-2)
    accepted = average[..., 1]
    risk = np.divide(average[..., 2], accepted,
                     out=np.full(accepted.shape, np.nan), where=accepted > 0)
    return np.stack([average[..., 0], accepted, risk], axis=-1)


def interval(point, samples):
    finite = np.isfinite(samples)
    valid = samples[finite]
    return dict(point=float(point), ci=np.quantile(valid, [.025,.975]).tolist() if len(valid) else None,
                defined_replicates=int(finite.sum()), undefined_replicates=int((~finite).sum()))


def main(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT/'configs/external_view_analysis.json').read_text())
    sample_path = ROOT/'configs/external_rsna_sample.csv'
    pred_path = ROOT/'reports/external/predictions.npz'
    with sample_path.open() as f:
        sample = [r for r in csv.DictReader(f) if r['arm']=='primary']
    source = np.load(pred_path)
    y = np.array([int(r['label']) for r in sample]); view=np.array([r['view'] for r in sample])
    np.testing.assert_array_equal(y,source['labels__external'])
    assert set(view)=={'AP','PA'} and len({r['patientId'] for r in sample})==len(y)
    with (ROOT/'reports/external/results.csv').open() as f:
        results=list(csv.DictReader(f))
    rng=np.random.default_rng(cfg['bootstrap_seed']); draws={}; counts={}
    for v in ('AP','PA'):
        for label in (0,1):
            ix=np.flatnonzero((view==v)&(y==label))
            counts[f'{v}_{label}']=len(ix)
            draws[v,label]=rng.integers(0,len(ix),(cfg['bootstrap_replicates'],len(ix)))
    summary=dict(config=cfg,counts=counts,results={},contrasts={},baseline={})
    boot_store={}; point_store={}; rows=[]
    names=['balanced_accuracy','standardized_coverage','standardized_retained_error']
    for arm in cfg['arms']:
        for condition in cfg['conditions']:
            for v in ('AP','PA'):
                seed_points=[];seed_boot=[]
                for seed in cfg['seeds']:
                    p=source[f'external__{arm}__s{seed}__{condition}']
                    threshold=float(next(r['threshold'] for r in results if r['dataset']=='external'
                                         and r['arm']==arm and int(r['seed'])==seed and r['condition']==condition))
                    keep=np.maximum(p,1-p)>=threshold;correct=(p>=.5)==y
                    cells=[];boot_cells=[]
                    for label in (0,1):
                        mask=(view==v)&(y==label)
                        values=np.column_stack([correct[mask],keep[mask],keep[mask]&~correct[mask]]).astype(float)
                        cells.append(values.mean(axis=0))
                        # Bounded temporary memory; same draws for every model/condition.
                        boot_cells.append(np.concatenate([values[ix].mean(axis=1)
                                                          for ix in np.array_split(draws[v,label],20)]))
                    seed_points.append(standardized_metrics(np.array(cells)))
                    seed_boot.append(standardized_metrics(np.stack(boot_cells,axis=1)))
                    mask=view==v
                    rows.append(dict(arm=arm,seed=seed,condition=condition,view=v,
                                     auroc=auroc(y[mask],p[mask]),
                                     **dict(zip(names,seed_points[-1]))))
                # Do not silently omit a seed with an empty accepted set.
                point=np.mean(seed_points,axis=0);boots=np.mean(seed_boot,axis=0)
                key=f'{arm}|{condition}|{v}'
                point_store[key]=point;boot_store[key]=boots
                summary['results'][key]={name:interval(point[i],boots[:,i]) for i,name in enumerate(names)}
            a=f'{arm}|{condition}|AP';b=f'{arm}|{condition}|PA'
            summary['contrasts'][f'{arm}|{condition}|PA_minus_AP']={
                name:interval(point_store[b][i]-point_store[a][i],boot_store[b][:,i]-boot_store[a][:,i])
                for i,name in enumerate(names)}
    # The view-only rule was chosen after inspecting the association: descriptive only.
    baseline=(view=='AP').astype(int)
    summary['baseline']=dict(accuracy=float(np.mean(baseline==y)),auroc=auroc(y,baseline),
                             within_view_balanced_accuracy=0.5,
                             scope='Post-hoc rule, no independent validation; does not establish what CNNs use')
    summary['provenance']=dict(sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                               predictions_sha256=hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               config_sha256=hashlib.sha256((ROOT/'configs/external_view_analysis.json').read_bytes()).hexdigest(),
                               commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (output/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(output/'bootstrap.npz',**{k.replace('|','__'):v for k,v in boot_store.items()})
    print('View-only baseline:',summary['baseline'])
    for k,v in summary['contrasts'].items():
        if '|clean|' in k:print(k,v)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'reports/external/view_analysis')
    main(parser.parse_args().output)
