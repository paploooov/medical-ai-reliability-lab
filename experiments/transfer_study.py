"""Versuch 10: Transferdiagnose mit unveränderten Gewichten und Originalkalibrierung."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/medical-ai-mpl')
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanistic_ties import sha, keys, fit_rule, apply_rule, counts, write_csv
from noise_diagnostics import inspect
from resolution_study import ResolutionCNN
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/transfer'
CONDITIONS = ['clean', 'noise', 'blur', 'contrast']
POLICIES = ['standard', 'tie_break_reject_zero']


def variants(images):
    """Identische Definitionen wie Versuch 07; Blur-Varianten sind bei 28px gleich."""
    return dict(clean=images.copy(),
        noise=np.clip(images + np.random.default_rng(42).normal(0, .1, images.shape), 0, 1),
        blur=gaussian_filter(images, (0, 1, 1), mode='reflect'),
        contrast=np.clip(.5 + .5 * (images - .5), 0, 1))


def evaluate(probability, zero, labels, identifiers, threshold, rule):
    conf = np.maximum(probability, 1 - probability)
    wrong = (probability >= .5) != labels
    common = dict(cases=len(labels), correct=int((~wrong).sum()), accuracy=float((~wrong).mean()),
        zero_count=int(zero.sum()), zero_fraction=float(zero.mean()),
        threshold_tie_count=int((conf == threshold).sum()),
        threshold_tie_fraction=float((conf == threshold).mean()),
        ties_with_zero=int(((conf == threshold) & zero).sum()))
    masks = [conf >= threshold, apply_rule(conf, identifiers, rule) & ~zero]
    return [dict(policy=policy, **common, **counts(mask, wrong)) for policy, mask in zip(POLICIES, masks)], masks


def prior_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ['models', 'reports']
            for p in sorted((ROOT / folder).rglob('*')) if p.is_file() and OUT not in p.parents}


def plots(out, rows):
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    specs = [('mechanism', [('accuracy', 'Accuracy', 'standard'), ('zero_fraction', 'Nullvektor-Anteil', 'standard'),
                          ('threshold_tie_fraction', 'Exakter Gleichstand an Originalschwelle', 'standard')]),
             ('selection', [('coverage', 'Akzeptanz: Standard', 'standard'), ('risk', 'Fehlerrate: Standard', 'standard'),
                            ('coverage', 'Akzeptanz: Tie-Breaking + Nullausschluss', POLICIES[1]),
                            ('risk', 'Fehlerrate: Tie-Breaking + Nullausschluss', POLICIES[1])])]
    for filename, panels in specs:
        fig, axes = plt.subplots(1 if filename == 'mechanism' else 2, 3 if filename == 'mechanism' else 2,
                                 figsize=(13, 4.8) if filename == 'mechanism' else (11, 8))
        for ax, (metric, title, policy) in zip(np.array(axes).ravel(), panels):
            for i, (arm, seed) in enumerate((a, s) for a in ['control', 'augmentation'] for s in [17,42,83]):
                values = [next(r[metric] for r in rows if r['arm']==arm and r['seed']==seed and r['condition']==c and r['policy']==policy) for c in CONDITIONS]
                x = np.arange(4) + (i-2.5)*.045
                label = ('Kontrolle' if arm=='control' else 'Augmentation') + f' / {seed}'
                ax.scatter(x, [v if v is not None else np.nan for v in values],
                           color='#345995' if arm=='control' else '#bd5735', marker=['o','s','^'][i%3], label=label)
                for j,v in enumerate(values):
                    if v is None: ax.annotate(f'undef. {seed}', (x[j], .03+i*.04), fontsize=7, rotation=90)
            ax.set(title=title, xticks=range(4), xticklabels=['Original','Rauschen','Blur','Kontrast'], ylim=(-.03,1.05))
            ax.grid(alpha=.2)
        handles, labels = np.array(axes).ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3)
        fig.suptitle('Versuch 10: einzelne Trainingsseeds; keine Konfidenzintervalle')
        fig.tight_layout(rect=(0,.1,1,.94))
        fig.savefig(out / f'{filename}.png', dpi=180)
        fig.savefig(out / f'{filename}.pdf', metadata={'CreationDate':None, 'ModDate':None})
        plt.close(fig)


def main(out):
    if out.exists():
        raise FileExistsError('Neuen Ausgabeordner wählen; bestehende Artefakte bleiben erhalten')
    previous = prior_hashes()
    out.mkdir(parents=True)
    (out/'previous_artifacts.json').write_text(json.dumps(previous, indent=2)+'\n')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source = dict(np.load(ROOT/'reports/calibration/predictions.npz'))
    historical = dict(np.load(ROOT/'reports/augmentation/comparison_predictions.npz'))
    manifest = json.loads((ROOT/'configs/resolution_manifest.json').read_text())['indices']
    meta = json.loads((ROOT/'reports/calibration/provenance.json').read_text())['models']
    data = np.load(ROOT/'data/pneumoniamnist.npz')
    ci, ti = source['calibration_indices'], source['test_indices']
    np.testing.assert_array_equal(ci, manifest['calibration'])
    np.testing.assert_array_equal(ti, manifest['test'])
    assert len(ci)==262 and len(ti)==621
    y = data['test_labels'][ti].ravel()
    np.testing.assert_array_equal(y, source['test_labels'])
    images = variants(data['test_images'][ti].astype(np.float32)/255)
    cal = data['val_images'][ci].astype(np.float32)/255
    rows, calibration, arrays, inputs = [], [], dict(labels=y,test_indices=ti,calibration_indices=ci), {}
    for i,m in enumerate(meta):
        arm, seed = m['arm'],m['seed']
        name = f'{arm}_s{seed}'
        path = ROOT/f"models/{'budget' if arm=='control' else 'augmentation'}/r28_s{seed}.pt"
        assert sha(path)==m['weights_sha256']
        inputs[str(path.relative_to(ROOT))]=sha(path)
        model = ResolutionCNN()
        model.load_state_dict(torch.load(path,weights_only=True)); model.eval()
        z, zero, _ = inspect(model, torch.tensor(cal[:,None],dtype=torch.float32))
        cp = expit(z)
        np.testing.assert_array_equal(cp,source['calibration_probabilities'][i])
        conf = np.maximum(cp,1-cp)
        threshold = float(np.quantile(conf,.2,method='linear'))
        rule = fit_rule(conf,keys('val',ci))
        calibration.append(dict(arm=arm,seed=seed,threshold=threshold,rank_threshold=rule[0],rank_hash=rule[1],bias=float(model.layers[-1].bias.detach()[0])))
        arrays[name+'_calibration_probabilities']=cp
        for condition,im in images.items():
            z,zero,_=inspect(model,torch.tensor(im[:,None],dtype=torch.float32))
            p=expit(z)
            old_condition='relative_blur' if condition=='blur' else condition
            np.testing.assert_array_equal(p,historical[f'{name}_{old_condition}'])
            if condition=='blur':np.testing.assert_array_equal(p,historical[f'{name}_fixed_blur'])
            np.testing.assert_array_equal(z[zero],np.full(zero.sum(),calibration[-1]['bias'],dtype=z.dtype))
            values,masks=evaluate(p,zero,y,keys('test',ti),threshold,rule)
            for row in values:rows.append(dict(arm=arm,seed=seed,condition=condition,**row))
            prefix=f'{name}_{condition}'
            arrays[prefix+'_logits']=z; arrays[prefix+'_zero']=zero; arrays[prefix+'_probabilities']=p
            for policy,mask in zip(POLICIES,masks):arrays[prefix+'_'+policy]=mask
    write_csv(out/'results.csv',rows)
    write_csv(out/'calibration.csv',calibration)
    np.savez_compressed(out/'predictions.npz',**arrays)
    paths=['reports/calibration/predictions.npz','reports/calibration/provenance.json',
           'reports/augmentation/comparison_predictions.npz','configs/resolution_manifest.json','data/pneumoniamnist.npz',
           'experiments/transfer_study.py','experiments/mechanistic_ties.py','experiments/noise_diagnostics.py',
           'experiments/resolution_study.py','experiments/augmentation_analysis.py']
    inputs.update({p:sha(ROOT/p) for p in paths})
    (out/'provenance.json').write_text(json.dumps(dict(inputs_sha256=inputs,models=meta,conditions=CONDITIONS,
        torch=torch.__version__,numpy=np.__version__,threads=2,deterministic_algorithms=True,
        rule='Originalkalibrierung; Study09-Hash-Salt; keine Nachkalibrierung unter Störung',
        blur='Gaussian sigma=1 pixel, reflect; relative_blur == fixed_blur at 28px',
        noise='Gaussian SD=0.1, seed=42, clip[0,1]',contrast='0.5+0.5*(image-0.5)'),indent=2)+'\n')
    plots(out,rows)
    for path,digest in previous.items():assert sha(ROOT/path)==digest,path
    print(f'{len(rows)} Ergebniszeilen; {len(previous)} frühere Artefakte unverändert')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=OUT)
    main(parser.parse_args().output)
