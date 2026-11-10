"""Joint empirical calibration/test resampling with temperature refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from temperature_scaling import fit_temperature,nll
ROOT=Path(__file__).resolve().parents[1]


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/temperature_sensitivity.json').read_text())
    source=ROOT/'reports/temperature_scaling/logits.npz'
    data=np.load(source);previous=json.loads((ROOT/'reports/temperature_scaling/summary.json').read_text())
    models=previous['models'];cy=data['calibration_labels'];y=data['test_labels']
    rng=np.random.default_rng(cfg['seed']);temperatures=[];differences=[]
    for _ in range(cfg['replicates']):
        ci=rng.integers(0,len(cy),len(cy));ti=rng.integers(0,len(y),len(y))
        temps=[];diff=[]
        for model in models:
            prefix=f"{model['arm']}_s{model['seed']}";zc=data[prefix+'_calibration_logits'];z=data[prefix+'_clean_logits']
            t=fit_temperature(cy[ci],zc[ci],cfg['temperature_bounds'])
            temps.append(t);diff.append(nll(y[ti],z[ti]/t)-nll(y[ti],z[ti]))
        temperatures.append(temps);differences.append(diff)
    temperatures=np.array(temperatures);differences=np.array(differences)
    summary=dict(config=cfg,temperatures=[],results={},provenance=dict(
        source_logits_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((ROOT/'configs/temperature_sensitivity.json').read_bytes()).hexdigest()))
    for i,m in enumerate(models):
        v=temperatures[:,i]
        summary['temperatures'].append(dict(arm=m['arm'],seed=m['seed'],point=m['temperature'],
           median=float(np.median(v)),range_95=np.quantile(v,[.025,.975]).tolist(),
           boundary_replicates=int(np.isin(v,cfg['temperature_bounds']).sum())))
    for arm in ['control','augmentation']:
        ix=[i for i,m in enumerate(models) if m['arm']==arm];values=differences[:,ix].mean(axis=1)
        summary['results'][arm]=dict(point=previous['results'][arm+'|clean']['nll']['difference'],
           range_95=np.quantile(values,[.025,.975]).tolist(),median=float(np.median(values)),
           fraction_positive=float(np.mean(values>0)))
    np.savez_compressed(output/'bootstrap.npz',temperatures=temperatures,nll_differences=differences)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary['results'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/temperature_scaling/sensitivity')
    main(p.parse_args().output)
