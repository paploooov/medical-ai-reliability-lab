"""Descriptive coverage sweep of saved external logits; no threshold selection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import csv,json
from pathlib import Path
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_high_resolution_robustness import grouped_risk
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'reports/external_chest224';p=np.load(folder/'predictions.npz');summary=json.loads((folder/'summary.json').read_text());rows=list(csv.DictReader((folder/'results.csv').open()));result=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,row,title in zip(axes,rows,['Pneumonie-Label vs. No finding','Pneumonie-Label vs. andere Labels']):
        control=row['control'];mask=(p['strata']=='pneumonia')|(p['strata']==control);y=p['labels'][mask];prob=expit(p['logits'][mask]);coverage=np.arange(1,21)/20
        risk=np.array([grouped_risk(y,prob,c) for c in coverage])
        np.testing.assert_allclose(risk[15],float(row['risk_at_80']),atol=1e-12)
        np.testing.assert_allclose(risk[-1],1-float(row['accuracy']),atol=1e-12)
        keep=np.maximum(prob,1-prob)>=summary['threshold'];fixed_error=np.mean((prob[keep]>=.5)!=y[keep]);np.testing.assert_allclose(fixed_error,float(row['retained_error']),atol=1e-12)
        for c,r in zip(coverage,risk):result.append(dict(control=control,coverage=float(c),error=float(r)))
        ax.plot(coverage*100,risk*100,'o-',color='#24618d',markersize=3,label='Nach Konfidenz sortiert')
        ax.axhline((1-float(row['accuracy']))*100,color='#777777',ls='--',label='Fehler ohne Ablehnung')
        ax.scatter(keep.mean()*100,fixed_error*100,color='#c05c22',marker='D',s=65,zorder=5,label='Feste interne Schwelle')
        ax.set(title=title,xlabel='Akzeptierter Anteil (%)',ylabel='Fehler unter akzeptierten Fällen (%)',xlim=(0,102),ylim=(0,100));ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Externer 224px-Test: Hohe Konfidenz schützt nicht zuverlässig vor Fehlern',fontsize=12)
    fig.savefig(folder/'selective_coverage.png',dpi=180);fig.savefig(folder/'selective_coverage.pdf');plt.close(fig)
    with (folder/'selective_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    (folder/'selective_coverage_verification.json').write_text(json.dumps(dict(passed=True,descriptive_only=True,new_training=False,new_threshold_selection=False,controls=2,coverage_points_per_control=20,existing_80_percent_and_full_coverage_values_verified=True,fixed_threshold_values_verified=True,ties='Fractional weighting of the boundary confidence group'),indent=2)+'\n')
if __name__=='__main__':main()
