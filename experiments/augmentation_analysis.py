"""Control-matched augmentation evaluation with negative-transfer checks."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
from scipy.special import expit
from scipy.ndimage import gaussian_filter
from resolution_study import ResolutionCNN,logits,tensor
from resolution_analysis import retained_risk
from selective_review import risk as matched_risk
from train import metrics
from error_detection import detection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/augmentation'
def read(p):return list(csv.DictReader(p.open()))
def main():
 torch.set_num_threads(2);manifest=json.loads((ROOT/'configs/resolution_manifest.json').read_text());idx=manifest['indices'];d=np.load(ROOT/'data/pneumoniamnist.npz');im=d['test_images'][idx['test']].astype(np.float32)/255;y=d['test_labels'][idx['test']].ravel();run=json.loads((OUT/'run.json').read_text());control_run=json.loads((ROOT/'reports/budget/run.json').read_text());aug=np.load(OUT/'predictions.npz');old=np.load(ROOT/'reports/budget/predictions.npz');np.testing.assert_array_equal(aug['labels'],y)
 assert run['manifest_sha256']==hashlib.sha256((ROOT/'configs/resolution_manifest.json').read_bytes()).hexdigest()
 variants={'clean':im,'relative_blur':gaussian_filter(im,(0,1,1),mode='reflect'),'fixed_blur':gaussian_filter(im,(0,1,1),mode='reflect'),'noise':np.clip(im+np.random.default_rng(42).normal(0,.1,im.shape),0,1),'contrast':np.clip(.5+.5*(im-.5),0,1)}
 for extra in run['config']['extra_noise']:variants[extra['name']]=np.clip(im+np.random.default_rng(extra['seed']).normal(0,extra['sigma'],im.shape),0,1)
 arrays={};rows=[];choices={'control':[c for c in control_run['choices'] if c['resolution']==28],'augmentation':run['choices']};cuts={}
 for arm in ['control','augmentation']:
  modeldir=ROOT/'models'/('budget' if arm=='control' else 'augmentation')
  for c in choices[arm]:
   seed=c['seed'];cuts[arm,seed]=c['threshold'];path=modeldir/f'r28_s{seed}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==c['weights_sha256'];model=ResolutionCNN();model.load_state_dict(torch.load(path,weights_only=True));model.eval()
   cp=expit(logits(model,tensor(d['val_images'][idx['calibration']])));np.testing.assert_allclose(np.quantile(np.maximum(cp,1-cp),.2),c['threshold'],atol=1e-12)
   for condition,images in variants.items():
    prob=expit(logits(model,torch.tensor(images[:,None],dtype=torch.float32)));key=f'r28_s{seed}_{condition}'
    if arm=='augmentation':np.testing.assert_allclose(prob,aug[key],atol=1e-7)
    elif key in old.files:np.testing.assert_allclose(prob,old[key],atol=1e-7)
    arrays[f'{arm}_s{seed}_{condition}']=prob
    rows.append(dict(arm=arm,seed=seed,condition=condition,threshold=c['threshold'],matched_80_risk=matched_risk(y,prob,.8),**metrics(y,prob),**detection(y,prob,c['threshold'])))
 # Independent comparison with all training-script CSV metrics.
 training_rows=read(OUT/'results.csv')
 for row in training_rows:
  computed=next(v for v in rows if v['arm']=='augmentation' and v['seed']==int(row['seed']) and v['condition']==row['condition'])
  for k,v in computed.items():
   if k not in row or k in ['seed','condition']:continue
   if v is None:assert row[k]==''
   else:np.testing.assert_allclose(float(row[k]),v,atol=1e-12)
 with (OUT/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(OUT/'comparison_predictions.npz',labels=y,test_indices=idx['test'],**arrays)
 def contrast(indices):
  values={}
  for arm in ['control','augmentation']:
   for c in ['clean','noise']:
    values[arm,c]=np.mean([retained_risk(y,arrays[f'{arm}_s{s}_{c}'],cuts[arm,s],indices) for s in [17,42,83]])
  delta=(values['augmentation','noise']-values['augmentation','clean'])-(values['control','noise']-values['control','clean'])
  clean_accuracy_delta=np.mean([np.mean((arrays[f'augmentation_s{s}_clean'][indices]>=.5)==y[indices])-np.mean((arrays[f'control_s{s}_clean'][indices]>=.5)==y[indices]) for s in [17,42,83]])
  return [delta,clean_accuracy_delta]
 point=contrast(np.arange(len(y)));rng=np.random.default_rng(20261110);boots=np.array([contrast(rng.integers(0,len(y),len(y))) for _ in range(2000)]);valid=np.isfinite(boots).all(1);ci=np.quantile(boots[valid],[.025,.975],axis=0)
 primary={'estimands':['augmentation_minus_control_change_in_noise_minus_clean_retained_risk','secondary_clean_accuracy_difference'],'point':point,'ci_low':ci[0].tolist(),'ci_high':ci[1].tolist(),'replicates':2000,'bootstrap_seed':20261110,'discarded':int((~valid).sum()),'scope':'Conditional on fitted models and cutoffs; exact-group resampling, not patients. Secondary interval exploratory without multiplicity correction.'};(OUT/'primary.json').write_text(json.dumps(primary,indent=2)+'\n')
 (OUT/'analysis_provenance.json').write_text(json.dumps({'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'control_predictions_sha256':hashlib.sha256((ROOT/'reports/budget/predictions.npz').read_bytes()).hexdigest(),'augmentation_predictions_sha256':hashlib.sha256((OUT/'predictions.npz').read_bytes()).hexdigest(),'bootstrap_seed':20261110,'primary_protocol':'docs/research/AUGMENTATION_PROTOCOL.md'},indent=2)+'\n')
 (OUT/'verification.json').write_text(json.dumps({'weights_and_cutoffs_verified':6,'augmentation_result_rows_verified':24,'control_existing_condition_outputs_verified':15,'comparison_rows':len(rows),'test_groups':len(y),'manifest_hash_verified':True},indent=2)+'\n')
 fig,axes=plt.subplots(1,3,figsize=(14,4.5));conditions=list(variants)
 for ax,metric,title in zip(axes,['accuracy','retained_error_rate','coverage'],['Accuracy','Errors among accepted','Accepted fraction']):
  for arm,color in [('control','#345995'),('augmentation','#bd5735')]:
   values=np.array([[next(v[metric] for v in rows if v['arm']==arm and v['seed']==s and v['condition']==c) for c in conditions] for s in [17,42,83]])
   mean=values.mean(0);positions=np.arange(len(conditions))+(-.1 if arm=='control' else .1);ax.errorbar(positions,mean,yerr=np.vstack([mean-values.min(0),values.max(0)-mean]),fmt='o',color=color,capsize=3,label=arm)
  ax.set(title=title,xticks=range(len(conditions)),xticklabels=conditions,ylim=(0,1));ax.tick_params(axis='x',rotation=65);ax.grid(alpha=.2)
 axes[0].legend();fig.suptitle('28px, 45-epoch budget: mean and seed range (not confidence bands)');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
 history=read(OUT/'learning_curves.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,s in zip(axes,[17,42,83]):
  h=[v for v in history if int(v['seed'])==s];ax.plot([int(v['epoch']) for v in h],[float(v['training_online_bce']) for v in h],label='Augmented fitting, online');ax.plot([int(v['epoch']) for v in h],[float(v['selection_bce']) for v in h],label='Clean model selection');ax.set(title=f'Seed {s}',xlabel='Epoch',ylabel='BCE');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
 lines=['# Versuch 07 – Gezieltes Training gegen Rauschen','', 'Drei neue 28×28-Modelle mit derselben Architektur und demselben 45-Epochen-Budget wie die Kontrollmodelle aus Versuch 06. Einziger geplanter Trainingsunterschied: Mit Wahrscheinlichkeit 0,5 erhält jedes Trainingsbild frisches Gaußrauschen (SD 0,1) und wird auf [0,1] begrenzt. Modellwahl und Kalibrierung bleiben ausschließlich auf getrennten Originaldaten.','', '## Ergebnisse im Vergleich','', '| Bedingung | Arm | Accuracy | Fehler unter akzeptierten Fällen | Akzeptierter Anteil | Fehler bei rückblickend 80% Akzeptanz |','|---|---|---:|---:|---:|---:|']
 for c in ['clean','noise','noise_7','noise_99','noise_high','relative_blur','contrast']:
  for arm in ['control','augmentation']:
   selected=[v for v in rows if v['arm']==arm and v['condition']==c];values=[np.mean([v[k] for v in selected]) for k in ['accuracy','retained_error_rate','coverage','matched_80_risk']];lines.append('| '+ ' | '.join([c,arm]+[f'{v:.1%}' for v in values])+' |')
 lines+=['', 'Mittelwerte über alle drei Seeds. noise = SD 0,1 / Seed42; noise_7 und noise_99 ändern die Zufallsfolge, noise_high nutzt SD 0,2 / Seed42. Die nachträglich auf 80% angeglichene Akzeptanz ist eine Rangdiagnose, keine Einsatzregel.','',f'Primärer Vergleich: Unterschied des Rauschen-minus-Original-Anstiegs der akzeptierten Fehlerrate, Augmentation minus Kontrolle: **{point[0]*100:.2f} Prozentpunkte**, bedingtes gepaartes 95%-Intervall **[{ci[0,0]*100:.2f}; {ci[1,0]*100:.2f}]**. Sekundäre Differenz der Accuracy auf Originalbildern: **{point[1]*100:.2f} Prozentpunkte**, exploratives Intervall **[{ci[0,1]*100:.2f}; {ci[1,1]*100:.2f}]**.','', '![Vergleich aller Bedingungen](../../reports/augmentation/comparison.png)','','![Lernverläufe](../../reports/augmentation/learning_curves.png)','','## Was der Versuch aussagen kann','','Die Intervention kennt absichtlich dieselbe Rauschfamilie und primäre Stärke wie die Auswertung. Ein Erfolg ist deshalb ein enger Nachweis störungsspezifischer Anpassung, kein Beleg allgemeiner Robustheit. Die anderen Rauschfolgen und stärkere Störung wurden bereits früher betrachtet; sie sind kein unberührter externer Test. Nachteile unter Unschärfe, Kontrast oder auf Originalbildern gehören ebenso zum Ergebnis. Die Trainingskurve bezieht sich auf teilweise verrauschte Bilder, die Auswahlkurve auf Originalbilder; ihr Abstand ist kein unmittelbar vergleichbarer Generalisierungsfehler.','','## Nachweise und Reproduktion','','[Protokoll](AUGMENTATION_PROTOCOL.md) · [48 Vergleichszeilen](../../reports/augmentation/comparison.csv) · [Primäre Analyse](../../reports/augmentation/primary.json) · [Nachprüfung](../../reports/augmentation/verification.json) · [PDF](../../reports/augmentation/comparison.pdf). Die 24 neuen Ergebniszeilen sowie sechs Modellgewichte und Kalibrierungsschwellen beider Arme wurden nachgeprüft; alte Kontrollvorhersagen stimmen mit Versuch 06 überein.','','Training: `python experiments/resolution_study.py --config configs/augmentation_study.json --output reports/augmentation --models models/augmentation`. Auswertung: `python experiments/augmentation_analysis.py`. Vorher Grundabhängigkeiten, CPU-PyTorch und die dokumentierten Archive bereitstellen. Die ursprünglichen Studien bleiben unverändert.','','## Grenzen','','Bekannter Benchmark, nur 28px, drei Seeds, unbekannte Patientenzusammenhänge. Intervalle gelten bedingt auf feste trainierte Modelle und Annahmeschwellen und schließen Kalibrierungs- sowie Trainingsunsicherheit aus. Die Unterdrückung eines synthetischen Fehlermusters ist keine klinische Sicherheitsgarantie. Es fand keine externe Validierung statt.']
 lines+=['', '## Interne Merkmalsdiagnose', '', 'Unter dem primären Rauschen haben die drei augmentierten Modelle bei 85, 89 und 72 Fällen einen vollständig leeren letzten Merkmalsvektor. Nullvektoren sind damit nicht vollständig verschwunden. Sie sind auch für sich genommen kein Beweis einer falschen Entscheidung: Die Bias-Ausgabe kann für normal gelabelte Fälle korrekt sein. Entscheidend bleiben Fehler und Akzeptanz, nicht nur die Zahl aktiver Neuronen.', '', '[Diagnose nach Klasse](../../reports/augmentation/noise_diagnostics_augmented.csv). Reproduktion: `python experiments/noise_diagnostics.py --reports reports/augmentation --models models/augmentation --tag augmented --output reports/augmentation`.']
 (ROOT/'docs/research/AUGMENTATION_STUDY.md').write_text('\n'.join(lines)+'\n');print(json.dumps(primary,indent=2));print('\n'.join(lines[6:15]))
if __name__=='__main__':main()
