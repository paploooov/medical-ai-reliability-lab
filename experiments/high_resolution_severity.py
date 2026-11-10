"""Frozen-model severity curves with paired uncertainty over a prespecified grid."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import argparse,csv,hashlib,json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from high_resolution_augmentation import standardize
from high_resolution_robustness import quantize,metric_vector,interval,METRICS
from high_resolution_probe import evaluate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perturb(image,index,family,level):
    if family not in ['noise','blur'] or level<0:raise ValueError('Invalid perturbation')
    if level==0:return image.copy()
    x=image.astype(np.float64)/255
    if family=='noise':x=x+np.random.default_rng(np.random.SeedSequence([42,int(index)])).normal(0,level,image.shape)
    else:x=gaussian_filter(x,level,mode='reflect')
    return quantize(x)


def logit_risk(y,z,coverage=.8):
    score=np.abs(z);target=coverage*len(y)
    boundary=np.sort(score)[::-1][int(np.ceil(target))-1]
    above=score>boundary;tied=score==boundary
    weights=above+tied*((target-above.sum())/tied.sum())
    return float(weights@((z>=0)!=y)/target)


def joint_band(point,draws):
    point=np.asarray(point);draws=np.asarray(draws)
    if draws.ndim!=2 or draws.shape[1]!=len(point) or not np.isfinite(draws).all():raise ValueError('Finite aligned grid required')
    radius=float(np.quantile(np.max(np.abs(draws-point[None,:]),axis=1),.95))
    return dict(radius=radius,low=(point-radius).tolist(),high=(point+radius).tolist())


def main(output):
    import torch
    from torchvision.models import resnet18
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg_path=ROOT/'configs/high_resolution_severity.json';cfg=json.loads(cfg_path.read_text())
    parent_path=ROOT/'reports/high_resolution/augmentation/summary.json';parent=json.loads(parent_path.read_text())
    base=json.loads((ROOT/'reports/high_resolution/summary.json').read_text())
    probe_path=ROOT/'reports/high_resolution/predictions_and_probes.npz';probes=np.load(probe_path)
    head_path=ROOT/'reports/high_resolution/augmentation/augmented_head.npz';head=np.load(head_path)
    old=np.load(ROOT/'reports/high_resolution/augmentation/predictions.npz')
    archive=ROOT/'data/pneumoniamnist_224.npz';checkpoint=ROOT/'data/torch/hub/checkpoints/resnet18-f37072fd.pth'
    assert sha(archive)==parent['provenance']['archive_sha256']
    assert sha(checkpoint)==parent['provenance']['encoder_sha256']
    np.testing.assert_array_equal(head['scaler_mean'],probes['native_224_scaler_mean'])
    np.testing.assert_array_equal(head['scaler_scale'],probes['native_224_scaler_scale'])
    indices=old['test_indices'];data=np.load(archive);images=data['test_images'][indices];y=data['test_labels'][indices].ravel()
    np.testing.assert_array_equal(y,old['labels'])
    provenance=dict(parent_sha256=sha(parent_path),original_probes_sha256=sha(probe_path),head_sha256=sha(head_path),
                    archive_sha256=sha(archive),encoder_sha256=sha(checkpoint),config_sha256=sha(cfg_path),script_sha256=sha(__file__),
                    helpers_sha256={n:sha(ROOT/'experiments'/n) for n in ['high_resolution_augmentation.py','high_resolution_robustness.py','high_resolution_probe.py','external_eval.py','temperature_scaling.py']})
    cache=ROOT/'data/high_resolution_severity';cache.mkdir(exist_ok=True);meta=cache/'provenance.json'
    if meta.exists():assert json.loads(meta.read_text())==provenance,'Stale cache'
    else:meta.write_text(json.dumps(provenance,indent=2)+'\n')
    torch.set_num_threads(2);encoder=resnet18(weights=None);encoder.load_state_dict(torch.load(checkpoint,weights_only=True))
    encoder.fc=torch.nn.Identity();encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    grid=[('clean','noise',0)]+[(f'noise_{x}','noise',x) for x in cfg['noise_sigma'] if x>0]+[(f'blur_{x}','blur',x) for x in cfg['blur_sigma_pixels'] if x>0]
    anchors={'clean':('data/high_resolution_features/native_224_test.npz','clean'),
             'noise_0.1':('data/high_resolution_augmentation/test_noise_42.npz','noise_42'),
             'blur_8':('data/high_resolution_augmentation/test_relative_blur.npz','relative_blur')}
    logits={};rows=[];cutoffs={'control':base['results']['native_224']['threshold'],'augmentation':parent['selection']['augmented_cutoff']}
    for name,family,level in grid:
        if name in anchors:
            file,previous=anchors[name];path=ROOT/file
            expected=(base['provenance']['feature_sha256'][path.name] if name=='clean' else parent['feature_sha256'][path.name])
            assert sha(path)==expected
            saved=np.load(path);features=saved['features'];np.testing.assert_array_equal(saved['indices'],indices)
        else:
            path=cache/f'{name}.npz'
            if path.exists():
                saved=np.load(path);features=saved['features'];np.testing.assert_array_equal(saved['indices'],indices)
            else:
                chunks=[]
                for start in range(0,len(y),32):
                    altered=np.stack([perturb(im,i,family,level) for im,i in zip(images[start:start+32],indices[start:start+32])])
                    x=torch.tensor(altered[:,None],dtype=torch.float32).repeat(1,3,1,1)/255
                    with torch.inference_mode():chunks.append(encoder((x-mean)/std).numpy())
                features=np.concatenate(chunks);np.savez_compressed(path,features=features,indices=indices)
        x=standardize(features,probes)
        for arm,coef,intercept in [('control',probes['native_224_coefficient'],probes['native_224_intercept']),('augmentation',head['coefficient'],head['intercept'])]:
            z=x@coef.ravel()+intercept[0];logits[arm,name]=z
            if name in anchors:np.testing.assert_allclose(z,old[arm+'__'+anchors[name][1]],atol=1e-10)
            p=expit(z)
            rows.append(dict(arm=arm,condition=name,family=family,severity=level,threshold=cutoffs[arm],
                             **evaluate(y,z,cutoffs[arm]),risk80_logit_rank=logit_risk(y,z),saturated=int(((p==0)|(p==1)).sum())))
        print('Evaluated',name,flush=True)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    point={key:metric_vector(y,z,cutoffs[key[0]]) for key,z in logits.items()}
    rng=np.random.default_rng(cfg['bootstrap_seed']);boot={key:[] for key in logits}
    for _ in range(cfg['bootstrap_replicates']):
        ix=rng.integers(0,len(y),len(y))
        for key,z in logits.items():boot[key].append(metric_vector(y[ix],z[ix],cutoffs[key[0]]))
    boot={key:np.array(v) for key,v in boot.items()};summary=dict(config=cfg,provenance=provenance,results={},differences={})
    for name,_,_ in grid:
        for arm in cutoffs:
            key=arm,name;summary['results'][arm+'|'+name]={m:interval(point[key][j],boot[key][:,j]) for j,m in enumerate(METRICS)}
        a,b=('augmentation',name),('control',name)
        summary['differences'][name]={m:interval(point[a][j]-point[b][j],boot[a][:,j]-boot[b][:,j]) for j,m in enumerate(METRICS)}
    summary['primary']=summary['differences']['noise_0.2']['risk_at_80']
    dp=np.array([summary['differences'][name]['risk_at_80']['point'] for name,_,_ in grid])
    db=np.column_stack([boot['augmentation',name][:,-1]-boot['control',name][:,-1] for name,_,_ in grid])
    summary['joint_grid_band']=dict(grid=[n for n,_,_ in grid],**joint_band(dp,db))
    summary['numerical_diagnostic']=dict(saturated_model_case_pairs=sum(r['saturated'] for r in rows),
        max_risk80_probability_vs_logit_difference=max(abs(r['risk_at_80']-r['risk80_logit_rank']) for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'predictions.npz',labels=y,test_indices=indices,**{a+'__'+c:z for (a,c),z in logits.items()})
    np.savez_compressed(out/'bootstrap.npz',risk_difference_grid=db,**{a+'__'+c:v for (a,c),v in boot.items()})
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharey='col')
    for i,(family,levels) in enumerate([('noise',cfg['noise_sigma']),('blur',cfg['blur_sigma_pixels'])]):
        names=['clean' if v==0 else f'{family}_{v}' for v in levels]
        for j,(metric,title) in enumerate([('risk_at_80','Error at 80% coverage'),('coverage','Original-cutoff coverage'),('retained_error','Original-cutoff accepted error')]):
            for arm,color in [('control','#345995'),('augmentation','#bd5735')]:
                values=[summary['results'][arm+'|'+name][metric]['point'] for name in names]
                axes[i,j].plot(levels,values,'o-',color=color,label=arm)
            axes[i,j].set(title=title,xlabel='Noise SD' if family=='noise' else 'Blur sigma (pixels)',ylim=(0,1));axes[i,j].grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Fixed 224px probes and cutoffs; lines connect evaluated grid points only')
    fig.tight_layout();fig.savefig(out/'severity.png',dpi=180);fig.savefig(out/'severity.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    band=summary['joint_grid_band'];lookup={name:k for k,(name,_,_) in enumerate(grid)}
    for ax,family,levels in zip(axes,['noise','blur'],[cfg['noise_sigma'],cfg['blur_sigma_pixels']]):
        ix=[lookup['clean' if v==0 else f'{family}_{v}'] for v in levels]
        ax.plot(levels,dp[ix]*100,'o-',color='#345995')
        ax.fill_between(levels,np.array(band['low'])[ix]*100,np.array(band['high'])[ix]*100,alpha=.2,color='#345995')
        ax.axhline(0,color='black',linestyle=':');ax.set(title=family,xlabel='Noise SD' if family=='noise' else 'Blur sigma (pixels)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Augmented minus control risk at 80% (pp)')
    fig.suptitle('Exploratory joint bootstrap band over the 10-point grid, fixed models')
    fig.tight_layout();fig.savefig(out/'difference_band.png',dpi=180);fig.savefig(out/'difference_band.pdf');plt.close(fig)
    print('Primary',summary['primary'],flush=True)
    for row in rows:print({k:row[k] for k in ['arm','condition','risk_at_80','coverage','retained_error']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/high_resolution/severity')
    main(p.parse_args().output)
