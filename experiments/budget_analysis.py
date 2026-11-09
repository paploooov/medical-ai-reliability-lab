"""Matched-budget analysis, checkpoint verification and figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
from pathlib import Path
import json,csv,hashlib
import numpy as np
import torch
from scipy.special import expit
from resolution_study import ResolutionCNN,logits,tensor
from train import metrics
from error_detection import detection
from resolution_analysis import retained_risk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/budget'
def read(path):return list(csv.DictReader(path.open()))
def main():
 torch.set_num_threads(2)
 manifest=json.loads((ROOT/'configs/resolution_manifest.json').read_text());idx=manifest['indices'];oldrun=json.loads((ROOT/'reports/resolution/run.json').read_text());newrun=json.loads((OUT/'run.json').read_text());runs={15:oldrun,45:newrun};pred={15:np.load(ROOT/'reports/resolution/predictions.npz'),45:np.load(OUT/'predictions.npz')};y=pred[15]['labels'];np.testing.assert_array_equal(y,pred[45]['labels']);np.testing.assert_array_equal(pred[15]['test_indices'],pred[45]['test_indices'])
 oldh=read(ROOT/'reports/resolution/learning_curves.csv');newh=read(OUT/'learning_curves.csv');prefix=[]
 for a in oldh:
  b=next(v for v in newh if all(a[k]==v[k] for k in ['resolution','seed','epoch']))
  prefix.append(max(abs(float(a[k])-float(b[k])) for k in ['training_online_bce','selection_bce']))
 assert max(prefix)<1e-7,'Training prefix mismatch: cannot claim matched extension'
 rows={15:read(ROOT/'reports/resolution/results.csv'),45:read(OUT/'results.csv')}
 for row in rows[45]:
  p=pred[45][f"r{row['resolution']}_s{row['seed']}_{row['condition']}"]
  for key,value in {**metrics(y,p),**detection(y,p,float(row['threshold']))}.items():
   if value is None:assert row[key]==''
   else:np.testing.assert_allclose(float(row[key]),value,atol=1e-12)
 cuts={budget:{(v['resolution'],v['seed']):v['threshold'] for v in run['choices']} for budget,run in runs.items()}
 for choice in newrun['choices']:
  r=choice['resolution'];s=choice['seed'];path=ROOT/f'models/budget/r{r}_s{s}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==choice['weights_sha256'];m=ResolutionCNN();m.load_state_dict(torch.load(path,weights_only=True));m.eval();d=np.load(ROOT/'data'/('pneumoniamnist.npz' if r==28 else 'pneumoniamnist_64.npz'))
  p=expit(logits(m,tensor(d['test_images'][idx['test']])));np.testing.assert_allclose(p,pred[45][f'r{r}_s{s}_clean'],atol=1e-7)
  cp=expit(logits(m,tensor(d['val_images'][idx['calibration']])));np.testing.assert_allclose(np.quantile(np.maximum(cp,1-cp),.2),choice['threshold'],atol=1e-12)
 def contrast(indices,r):
  ds={}
  for budget in [15,45]:
   ds[budget]=np.mean([retained_risk(y,pred[budget][f'r{r}_s{s}_noise'],cuts[budget][r,s],indices)-retained_risk(y,pred[budget][f'r{r}_s{s}_clean'],cuts[budget][r,s],indices) for s in [17,42,83]])
  return ds[45]-ds[15]
 point=[contrast(np.arange(len(y)),r) for r in [28,64]];rng=np.random.default_rng(20261109);boots=[]
 for _ in range(2000):
  sample=rng.integers(0,len(y),len(y));boots.append([contrast(sample,r) for r in [28,64]])
 boots=np.array(boots);valid=np.isfinite(boots).all(1);ci=np.quantile(boots[valid],[.025,.975],axis=0)
 primary={'estimand':'45-minus-15 change in noise-minus-clean retained error risk','resolutions':[28,64],'primary_resolution':28,'point':point,'ci_low':ci[0].tolist(),'ci_high':ci[1].tolist(),'replicates':2000,'bootstrap_seed':20261109,'discarded':int((~valid).sum()),'scope':'Paired exact-group bootstrap conditional on models and cutoffs; not patient-level or population training uncertainty.'};(OUT/'primary.json').write_text(json.dumps(primary,indent=2)+'\n')
 verification={'prefix_epochs_checked':len(prefix),'max_prefix_loss_difference':max(prefix),'metric_rows_verified':30,'weights_and_cutoffs_verified':6};(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
 plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for i,r in enumerate([28,64]):
  for j,s in enumerate([17,42,83]):
   ax=axes[i,j];h=[v for v in newh if int(v['resolution'])==r and int(v['seed'])==s]
   ax.plot([int(v['epoch']) for v in h],[float(v['training_online_bce']) for v in h],label='Training, online');ax.plot([int(v['epoch']) for v in h],[float(v['selection_bce']) for v in h],label='Selection');ax.axvline(15,color='gray',linestyle=':');ax.set(title=f'{r}px, seed {s}',xlabel='Epoch',ylabel='BCE');ax.grid(alpha=.2)
 axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);fig.savefig(OUT/'learning_curves.pdf');plt.close(fig)
 diagnostics={b:read(OUT/f'noise_diagnostics_{b}.csv') for b in [15,45]}
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,r in zip(axes,[28,64]):
  for j,s in enumerate([17,42,83]):
   values=[sum(int(v['all_zero_hidden']) for v in diagnostics[b] if int(v['resolution'])==r and int(v['seed'])==s and v['condition']=='noise')/len(y) for b in [15,45]]
   ax.plot([15,45],values,'o-',label=f'Seed {s}')
  ax.set(title=f'{r}px: all-zero final hidden vector under noise',xlabel='Training budget (selected checkpoint)',ylabel='Fraction of test cases',xticks=[15,45],ylim=(0,1));ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'hidden_collapse.png',dpi=180);fig.savefig(OUT/'hidden_collapse.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,key,title in zip(axes,['retained_error_rate','coverage'],['Errors among accepted noisy cases','Accepted fraction under noise']):
  for r,color in [(28,'#345995'),(64,'#bd5735')]:
   for s in [17,42,83]:
    values=[float(next(v[key] for v in rows[b] if int(v['resolution'])==r and int(v['seed'])==s and v['condition']=='noise')) for b in [15,45]]
    ax.plot([15,45],values,'o-',color=color,alpha=.7,label=f'{r}px' if s==17 else None)
  ax.set(title=title,xlabel='Training budget',xticks=[15,45],ylim=(0,1));ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'noise_risk.png',dpi=180);fig.savefig(OUT/'noise_risk.pdf');plt.close(fig)
 lines=['# Versuch 06 – Rauschen, Merkmalsverlust und längeres Training','', 'Nachanalyse und festgelegte Trainingsverlängerung auf demselben bereits untersuchten Benchmark. Die sechs Modelle wurden mit identischer Initialisierung und Reihenfolge erneut trainiert, diesmal mit einem Budget von 45 statt 15 Epochen. Checkpoints werden weiterhin ausschließlich nach Modellwahl-Daten ausgewählt. Kein Rauschtraining, keine neue Architektur.','', '## Was in den früheren Modellen passiert','', 'Bei den 28px-Modellen mit Seed 42 und 83 ist unter dem getesteten Rauschen für 597 bzw. 589 der 621 Fälle der gesamte letzte Merkmalsvektor null. Für diese Fälle reduziert sich der Logit rechnerisch auf den festen Bias der Ausgabeschicht; P(pneumonia) liegt bei etwa 0,422 bzw. 0,439. Das erklärt die identische Negativentscheidung in diesen Fällen, aber nicht automatisch die Ursache aller übrigen Fehler.','', 'Skalierung, Rauschverteilung, Clipping, float32-Verarbeitung und alte Vorhersagen wurden reproduziert. Das ist ein nachvollziehbarer interner Rechenmechanismus, keine klinische Erklärung. Zero-ReLU-Aktivierungen sind inputabhängig; daraus folgt nicht, dass diese Neuronen bei allen Bildern dauerhaft tot sind.','', '## Vergleich der Trainingsbudgets','', '| Auflösung | Budget | Accuracy original | Accuracy Rauschen | Fehler unter akzeptierten Rauschfällen | Akzeptierte Rauschfälle |','|---|---:|---:|---:|---:|---:|']
 for r in [28,64]:
  for b in [15,45]:
   clean=[v for v in rows[b] if int(v['resolution'])==r and v['condition']=='clean'];noise=[v for v in rows[b] if int(v['resolution'])==r and v['condition']=='noise'];vals=[np.mean([float(v['accuracy']) for v in clean])]+[np.mean([float(v[k]) for v in noise]) for k in ['accuracy','retained_error_rate','coverage']]
   lines.append(f'| {r} | {b} | '+' | '.join(f'{v:.1%}' for v in vals)+' |')
 lines+=['', 'Mittelwerte über die drei Seeds. Unterschiedliche Akzeptanzraten müssen zusammen mit dem Fehleranteil gelesen werden. Ein längeres Budget heißt nicht, dass der ausgewählte Checkpoint aus Epoche 45 stammt.','',f'Primärer Vergleich bei 28px: Veränderung des Rauschen-minus-Original-Anstiegs der akzeptierten Fehlerrate, 45er- minus 15er-Budget: **{point[0]*100:.2f} Prozentpunkte**, bedingtes gepaartes 95%-Intervall **[{ci[0,0]*100:.2f}; {ci[1,0]*100:.2f}]**. Negative Werte bedeuten einen geringeren Anstieg unter dieser Regel. Das ist kein externer Wirksamkeitsnachweis.','', '![Rauschrisiko und Akzeptanz](../../reports/budget/noise_risk.png)','','![Merkmalsverlust](../../reports/budget/hidden_collapse.png)','','![Lernverlauf](../../reports/budget/learning_curves.png)','','## Gewählte Checkpoints','', '| Auflösung | Seed | Gewählte Epoche im 45er-Budget | Trainingssekunden |','|---|---:|---:|---:|']
 for c in newrun['choices']:lines.append(f"| {c['resolution']} | {c['seed']} | {c['epoch']} | {c['training_seconds']:.1f} |")
 lines+=['', '## Nachprüfbarkeit','',f"Die ersten {len(prefix)} Epocheneinträge (15 je Lauf) stimmen mit den früheren Verläufen überein; maximale Verlustabweichung {max(prefix):.3g}. Alle 30 neuen Ergebniszeilen, sechs gespeicherten Modelle und Kalibrierungsschwellen wurden nachgeprüft.",'', '[Protokoll](BUDGET_PROTOCOL.md) · [Ergebnis-CSV](../../reports/budget/results.csv) · [Primärer Vergleich](../../reports/budget/primary.json) · [Prüfnachweis](../../reports/budget/verification.json) · [Diagnose pro Klasse, 15er-Budget](../../reports/budget/noise_diagnostics_15.csv) · [Diagnose pro Klasse, 45er-Budget](../../reports/budget/noise_diagnostics_45.csv)','','Reproduktion: `python experiments/noise_diagnostics.py`; `python experiments/resolution_study.py --config configs/budget_study.json --output reports/budget --models models/budget`; `python experiments/noise_diagnostics.py --reports reports/budget --models models/budget --tag 45`; `python experiments/budget_analysis.py`. CPU-PyTorch und Grundabhängigkeiten wie in der README. Neue Läufe ersetzen nur die angegebenen Budget-Ausgaben.','','## Grenzen','','Drei Seeds, ein bekannter Datensatz, synthetisches Gaußrauschen und unbekannte Patientenidentitäten. Die gepaarten Intervalle resampeln eindeutige Pixelgruppen und halten Training sowie Kalibrierung fest; sie erfassen nicht deren Unsicherheit. Die Hauptfrage wurde nach Sichtung der früheren Rauschfehler gewählt. Weitere Bedingungen sind sekundär. Es gibt keine Aussage über allgemeine medizinische Sicherheit, keine garantierte maximale Fehlerrate und keine unabhängige Replikation.']
 lines+=['', '## Warum die Annahmeregel den Merkmalsverlust nicht abfängt', '', 'Bei den alten 28px-Modellen akzeptiert die feste Kalibrierungsschwelle alle 597 bzw. 589 Bias-only-Ausgaben der Seeds 42 und 83; darunter sind 377 bzw. 382 falsche Vorhersagen. Diese Ausgaben müssen nicht hochkonfident im Sinne von mindestens 90% sein. Sie liegen lediglich über oder auf der für das jeweilige Modell gesetzten Annahmeschwelle. Die inklusive Regel `confidence >= cutoff` akzeptiert auch Gleichstände. Eine auf clean-Daten gewählte Schwelle ist deshalb keine Absicherung gegen einen solchen Merkmalsverlust.', '', '[Zählung beider Trainingsbudgets](../../reports/budget/bias_acceptance.csv). Reproduktion: `python experiments/bias_acceptance.py` nach den Aktivierungsdiagnosen.']
 lines+=['', '## Sechs Rauschstärken, drei Rausch-Zufallsfolgen', '', '![Rauschstärken und Merkmalsverlust](../../reports/budget/noise_severity.png)', '', 'Diese Zusatzdiagnose verwendet ausschließlich die eingefrorenen 15er-Modelle: sechs Rauschstärken × drei Rauschseeds × sechs Modelle, insgesamt 108 Auswertungen. Die Bänder zeigen Minimum und Maximum über Rauschrealisationen, keine Konfidenzintervalle. Originalzustand und die frühere Störung mit SD 0,1 und Seed 42 wurden gegen die gespeicherten Ergebnisse geprüft.', '', 'Der Anteil vollständig leerer Merkmalsvektoren ist nicht bei jedem Modell monoton: Bei stärkerem Rauschen können wieder Merkmale aktiviert werden, während die Klassifikation weiterhin schlecht bleibt. Weniger Nullvektoren beweisen deshalb keine Erholung. Auch auf Originalbildern existieren bei einigen Modellen Nullvektoren; ihre bloße Existenz ist kein Fehlernachweis. Die diagnostizierte Bias-Ausgabe erklärt einen Teil des Zusammenbruchs, nicht die gesamte Robustheit.', '', '[108 Diagnosezeilen](../../reports/budget/noise_severity.csv) · [Nachprüfung](../../reports/budget/sweep_verification.json). Reproduktion: `python experiments/noise_sweep.py`.']
 (ROOT/'docs/research/BUDGET_STUDY.md').write_text('\n'.join(lines)+'\n');print(json.dumps(primary,indent=2));print('\n'.join(lines[10:20]))
if __name__=='__main__':main()
