"""Empirical calibration-sample sensitivity; frozen models, no training."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import json,csv,hashlib,subprocess
import numpy as np
import torch
from scipy.special import expit
from resolution_study import ResolutionCNN,logits,tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/calibration'
NAMES=['control_clean_risk','control_noise_risk','augmentation_clean_risk','augmentation_noise_risk','control_clean_coverage','control_noise_coverage','augmentation_clean_coverage','augmentation_noise_coverage','primary_risk_change_difference','clean_risk_difference']
def cutoffs(confidence,indices):
 return np.quantile(confidence[:,indices],.2,axis=1,method='linear')
def summarize(test_confidence,wrong,thresholds,indices):
 keep=test_confidence[:,:,indices]>=thresholds[:,None,None];den=keep.sum(2);num=(keep&wrong[:,:,indices]).sum(2)
 risks=np.divide(num,den,out=np.full(num.shape,np.nan),where=den>0).reshape(2,3,2).mean(1)
 coverage=keep.mean(2).reshape(2,3,2).mean(1)
 contrast=(risks[1,1]-risks[1,0])-(risks[0,1]-risks[0,0])
 return np.r_[risks.ravel(),coverage.ravel(),contrast,risks[1,0]-risks[0,0]]
def percentile_summary(values):
 finite=np.isfinite(values);valid=values[finite]
 return {'finite':int(finite.sum()),'undefined':int((~finite).sum()),'low':float(np.quantile(valid,.025)) if len(valid) else None,'high':float(np.quantile(valid,.975)) if len(valid) else None,'median':float(np.median(valid)) if len(valid) else None}
def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 config=json.loads((ROOT/'configs/calibration_study.json').read_text());manifest=json.loads((ROOT/'configs/resolution_manifest.json').read_text());idx=manifest['indices'];d=np.load(ROOT/'data/pneumoniamnist.npz');x=tensor(d['val_images'][idx['calibration']]);source=ROOT/'reports/augmentation/comparison_predictions.npz';stored=np.load(source);y=stored['labels'];np.testing.assert_array_equal(stored['test_indices'],idx['test']);np.testing.assert_array_equal(y,d['test_labels'][idx['test']].ravel())
 cal=[];test=[];metadata=[];threshold_rows=[]
 for arm in config['arms']:
  folder='budget' if arm=='control' else 'augmentation';run=json.loads((ROOT/f'reports/{folder}/run.json').read_text())
  for seed in config['seeds']:
   c=next(v for v in run['choices'] if v['resolution']==28 and v['seed']==seed);path=ROOT/f'models/{folder}/r28_s{seed}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==c['weights_sha256'];m=ResolutionCNN();m.load_state_dict(torch.load(path,weights_only=True));m.eval();cp=expit(logits(m,x));confidence=np.maximum(cp,1-cp);threshold=float(np.quantile(confidence,.2,method='linear'));np.testing.assert_allclose(threshold,c['threshold'],atol=1e-12)
   cal.append(cp);test.append(np.stack([stored[f'{arm}_s{seed}_{condition}'] for condition in config['conditions']]));metadata.append({'arm':arm,'seed':seed,'weights_sha256':c['weights_sha256']})
   threshold_rows.append(dict(arm=arm,seed=seed,threshold=threshold,calibration_count=len(cp),calibration_at_threshold=int((confidence==threshold).sum()),calibration_below_threshold=int((confidence<threshold).sum()),actual_calibration_coverage=float((confidence>=threshold).mean())))
 cal=np.array(cal);test=np.array(test);cc=np.maximum(cal,1-cal);tc=np.maximum(test,1-test);wrong=(test>=.5)!=y;baseline=cutoffs(cc,np.arange(len(x)));point=summarize(tc,wrong,baseline,np.arange(len(y)))
 n=config['bootstrap_replicates'];test_rng=np.random.default_rng(config['test_bootstrap_seed']);cal_rng=np.random.default_rng(config['calibration_bootstrap_seed']);fixed=[];joint=[];thresholds=[];cal_indices=[];test_indices=[];accepted_counts=[]
 for _ in range(n):
  ti=test_rng.integers(0,len(y),len(y));ci=cal_rng.integers(0,len(x),len(x));cut=cutoffs(cc,ci)
  fixed.append(summarize(tc,wrong,baseline,ti));joint.append(summarize(tc,wrong,cut,ti));thresholds.append(cut);cal_indices.append(ci);test_indices.append(ti);accepted_counts.append((tc[:,:,ti]>=cut[:,None,None]).sum(2))
 fixed=np.array(fixed);joint=np.array(joint);thresholds=np.array(thresholds);accepted_counts=np.array(accepted_counts)
 empty_rows=[dict(arm=m['arm'],seed=m['seed'],condition=condition,empty_replicates=int((accepted_counts[:,i,j]==0).sum()),replicates=n) for i,m in enumerate(metadata) for j,condition in enumerate(config['conditions'])]
 with (OUT/'empty_subsets.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=empty_rows[0]);w.writeheader();w.writerows(empty_rows)
 # Verify fixed-threshold analysis reproduces the prior primary interval.
 prior=json.loads((ROOT/'reports/augmentation/primary.json').read_text());np.testing.assert_allclose(point[8],prior['point'][0],atol=1e-12);np.testing.assert_allclose([percentile_summary(fixed[:,8])['low'],percentile_summary(fixed[:,8])['high']],[prior['ci_low'][0],prior['ci_high'][0]],atol=1e-12)
 summary={'point':dict(zip(NAMES,point.tolist())),'fixed_threshold':{name:percentile_summary(fixed[:,i]) for i,name in enumerate(NAMES)},'reestimated_threshold':{name:percentile_summary(joint[:,i]) for i,name in enumerate(NAMES)},'scope':config['interval_scope'],'config':config}
 subset_rng=np.random.default_rng(config['subset_seed']);subset_rows=[];subsets={};subset_stats={};subset_cutoffs={}
 for size in config['subset_sizes']:
  selections=np.array([subset_rng.choice(len(x),size,replace=False) for _ in range(config['subset_replicates'])]);values=[];cuts=[]
  for selection in selections:
   cut=cutoffs(cc,selection);cuts.append(cut);values.append(summarize(tc,wrong,cut,np.arange(len(y))))
  values=np.array(values);subsets[f'n{size}']=selections;subset_stats[f'n{size}']=values;subset_cutoffs[f'n{size}']=np.array(cuts)
  if size==len(x):np.testing.assert_allclose(values,np.broadcast_to(point,values.shape),atol=1e-12)
  for i,name in enumerate(NAMES):subset_rows.append(dict(calibration_size=size,metric=name,**percentile_summary(values[:,i])))
 with (OUT/'thresholds.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=threshold_rows[0]);w.writeheader();w.writerows(threshold_rows)
 with (OUT/'subset_sensitivity.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=subset_rows[0]);w.writeheader();w.writerows(subset_rows)
 np.savez_compressed(OUT/'predictions.npz',calibration_probabilities=cal,calibration_indices=idx['calibration'],test_probabilities=test,test_indices=idx['test'],test_labels=y)
 np.savez_compressed(OUT/'bootstrap.npz',fixed=fixed,joint=joint,thresholds=thresholds,accepted_counts=accepted_counts,calibration_indices=np.array(cal_indices,dtype=np.int16),test_indices=np.array(test_indices,dtype=np.int16))
 np.savez_compressed(OUT/'subsets.npz',**{f'indices_{k}':v for k,v in subsets.items()},**{f'statistics_{k}':v for k,v in subset_stats.items()},**{f'thresholds_{k}':v for k,v in subset_cutoffs.items()})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'provenance.json').write_text(json.dumps({'models':metadata,'metric_order':NAMES,'source_predictions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((ROOT/'configs/resolution_manifest.json').read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 axes[0].hist(fixed[np.isfinite(fixed[:,8]),8]*100,bins=35,alpha=.6,label='Test sampling only');axes[0].hist(joint[np.isfinite(joint[:,8]),8]*100,bins=35,alpha=.6,label=f'Test + calibration ({np.isfinite(joint[:,8]).sum()}/{n} defined)');axes[0].axvline(point[8]*100,color='black',linestyle=':');axes[0].set(xlabel='Primary contrast (percentage points)',ylabel='Replicates',title='Conditional empirical sensitivity');axes[0].legend(fontsize=8)
 for i,m in enumerate(metadata):
  lo,hi=np.quantile(thresholds[:,i],[.025,.975]);axes[1].plot([i,i],[lo,hi],linewidth=3,color='#345995' if m['arm']=='control' else '#bd5735');axes[1].plot(i,baseline[i],'o',color='black')
 axes[1].set(xticks=range(6),xticklabels=[f"{m['arm']}\ns{m['seed']}" for m in metadata],ylabel='Confidence cutoff',title='Calibration bootstrap 2.5–97.5% range',ylim=(.5,1));axes[1].tick_params(axis='x',labelsize=8);fig.tight_layout();fig.savefig(OUT/'bootstrap_sensitivity.png',dpi=180);fig.savefig(OUT/'bootstrap_sensitivity.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,metric in zip(axes,['augmentation_noise_risk','augmentation_noise_coverage']):
  selected=[r for r in subset_rows if r['metric']==metric];xx=np.arange(len(selected));middle=np.array([r['median'] for r in selected]);ax.errorbar(xx,middle,yerr=np.vstack([middle-np.array([r['low'] for r in selected]),np.array([r['high'] for r in selected])-middle]),fmt='o',capsize=5);ax.set(xticks=xx,xticklabels=[r['calibration_size'] for r in selected],xlabel='Subset size within the existing calibration pool',ylabel=metric.replace('_',' '),ylim=(0,1));ax.grid(alpha=.2)
 fig.suptitle('Median and central 95% of subset results; not prospective confidence intervals');fig.tight_layout();fig.savefig(OUT/'subset_sensitivity.png',dpi=180);fig.savefig(OUT/'subset_sensitivity.pdf');plt.close(fig)
 lines=['# Versuch 08 – Wie stabil ist die Kalibrierung?', '', 'Keine neuen Modelle. Die sechs Modelle aus dem Augmentationsvergleich bleiben unverändert. Neu ist, dass nicht nur die Testfälle, sondern auch die 262 Fälle zur Bestimmung der Annahmeschwellen variiert werden.', '', '## Primärer Vergleich', '', 'Kennzahl wie in Versuch 07: Unterschied des Rauschen-minus-Original-Anstiegs der Fehlerrate unter akzeptierten Vorhersagen, Augmentation minus Kontrolle. Negative Werte sprechen unter dieser Regel für die Augmentation.', '', '| Auswertung | Punktschätzer | Empirischer 2,5–97,5%-Bereich | Undefinierte Ziehungen |','|---|---:|---:|---:|']
 for key,label in [('fixed_threshold','Nur Testfälle variieren'),('reestimated_threshold','Test- und Kalibrierungsfälle variieren')]:
  stat=summary[key][NAMES[8]];lines.append(f"| {label} | {point[8]*100:.2f} pp | [{stat['low']*100:.2f}; {stat['high']*100:.2f}] pp | {stat['undefined']} / {n} |")
 lines+=['', 'Der feste Vergleich reproduziert die ursprüngliche Analyse aus Versuch 07 mit denselben Testziehungen. Die neue Analyse setzt in jeder Ziehung für alle Modelle gemeinsam dieselben gezogenen Kalibrierungsfälle ein und berechnet deren Schwellen neu. Modelle und deren Auswahl bleiben fest.', '', '![Empirische Schwellenunsicherheit](../../reports/calibration/bootstrap_sensitivity.png)', '', '## Gleichstände an der Annahmeschwelle', '', '| Arm | Seed | Schwelle | Kalibrierungsfälle genau auf der Schwelle | Tatsächlich akzeptiert im Kalibrierungssatz |','|---|---:|---:|---:|---:|']
 for r in threshold_rows:lines.append(f"| {r['arm']} | {r['seed']} | {r['threshold']:.6f} | {r['calibration_at_threshold']} / {r['calibration_count']} | {r['actual_calibration_coverage']:.1%} |")
 lines+=['', 'Die Regel verwendet `confidence >= cutoff`. Große Gruppen identischer Konfidenz werden deshalb gemeinsam akzeptiert. Ein Zielwert von 80% ergibt nicht zwingend genau 80% Akzeptanz, selbst auf den Kalibrierungsfällen. Solche Punktmassen erschweren die gewöhnliche Deutung von Bootstrap-Perzentilen als nominal verlässliches 95%-Konfidenzintervall. Die Bereiche werden hier bewusst als empirische Sensitivitätsbereiche ausgewiesen.', '', '## Weniger Kalibrierungsfälle', '', 'Je 500 Teilmengen der Größen 32, 64, 128 und 262 wurden ohne Zurücklegen aus dem vorhandenen Pool gezogen; ausgewertet wurde auf allen 621 Testrepräsentanten. Bei 262 ist die Variation definitionsgemäß null: Es handelt sich jedes Mal um denselben vollständigen Pool. Das ist kein Nachweis, dass 262 neue Patienten immer eine stabile Schwelle liefern.', '', '![Teilmengen-Sensitivität](../../reports/calibration/subset_sensitivity.png)', '', '## Nachweise', '', '[Protokoll](CALIBRATION_PROTOCOL.md) · [Zusammenfassung aller Kennzahlen](../../reports/calibration/summary.json) · [Schwellen und Gleichstände](../../reports/calibration/thresholds.csv) · [Teilmengen-Ergebnisse](../../reports/calibration/subset_sensitivity.csv) · [PDF](../../reports/calibration/bootstrap_sensitivity.pdf).', '', 'Alle Kalibrierungsvorhersagen und gezogenen Indexlisten sind gespeichert. Reproduktion: `python experiments/calibration_study.py`. Verwendet die bestehenden Modellgewichte und das geprüfte 28px-Archiv; keine erneute Optimierung.', '', '## Grenzen', '', 'Empirische Wiederziehungen ersetzen keine neue Klinik oder neue Patienten. Die Analyse bleibt bedingt auf sechs feste Modelle, einen bekannten Benchmark und Pixelgruppen. Sie enthält weder Trainings-/Modellwahlunsicherheit noch unbekannte Patientenkorrelation. Die Intervallbreite kann trotz zusätzlicher Variation auch kleiner werden, etwa durch diskrete Schwellen; eine einfache monotone „mehr Unsicherheit = breiter“-Interpretation ist nicht garantiert. Keine garantiert sichere Annahmeschwelle oder klinische Fehlerobergrenze wird abgeleitet.']
 lines+=['', '## Leere Teilmengen sind kein Erfolg', '', f"Bei {summary['reestimated_threshold'][NAMES[8]]['undefined']} der {n} gemeinsamen Wiederziehungen ist der primäre Vergleich undefiniert, weil mindestens eine benötigte akzeptierte Teilmenge leer ist. Der angegebene gemeinsame Perzentilbereich beruht auf den übrigen {summary['reestimated_threshold'][NAMES[8]]['finite']} Ziehungen und ist damit zusätzlich auf Definierbarkeit bedingt. Er darf nicht so gelesen werden, als hätte die Regel in allen Ziehungen funktioniert. Eine Regel, die keine Fälle akzeptiert, hat hier keine gemessene Fehlerrate von null.", '', '[Aufschlüsselung nach Modell und Bedingung](../../reports/calibration/empty_subsets.csv). Die Zahl angenommener Fälle je Modell/Bedingung ist für jede Ziehung im Bootstrap-Archiv gespeichert.', '', '## Methodische Konsequenz', '', 'Vor einer weiteren Modellvergrößerung sollte die Annahmeregel Gleichstände ausdrücklich behandeln. Eine mögliche nächste Untersuchung ist eine nur auf Kalibrierungsdaten bestimmte, zufällige Annahme eines Anteils der Grenzgruppe. Sie würde das Akzeptanzziel im Erwartungswert treffen, aber weder die Fehlerquote garantieren noch Verteilungsänderungen lösen. Die bisherige Regel wird hier nicht rückwirkend geändert.']
 lines+=['', 'Mit Kalibrierung ist in diesem Versuch die Festlegung der Annahmeschwelle gemeint, nicht eine nachträgliche Korrektur der Modellwahrscheinlichkeiten durch Temperature Scaling.', '', 'Zusätzliche Nachprüfung: `python experiments/verify_calibration.py` rechnet beide Varianten aller 2.000 Bootstrap-Ziehungen und sämtliche 2.000 Teilmengen-Auswertungen mit einer separaten skalaren Implementierung nach. [Prüfdatei](../../reports/calibration/verification.json).']
 (ROOT/'docs/research/CALIBRATION_STUDY.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:summary[k][NAMES[8]] for k in ['fixed_threshold','reestimated_threshold']},indent=2));print(threshold_rows)
if __name__=='__main__':main()
