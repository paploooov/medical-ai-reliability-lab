"""At most two stage-three figures, from stored results only."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/thoraxis-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/original_512/thoraxis_30'

def main():
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    s=json.loads((OUT/'source_cnn_results.json').read_text());c=json.loads((OUT/'source_logit_correlations.json').read_text());z=np.load(OUT/'source_oof.npz',allow_pickle=False)
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.8));fp,tp,_=roc_curve(z['label'],z['score']);a.plot(fp,tp,lw=2,label=f"OOF-CNN: AUROC {s['auroc']:.3f}")
    for fold in range(5):
        m=z['fold']==fold;f,t,_=roc_curve(z['label'][m],z['score'][m]);a.plot(f,t,alpha=.25,lw=1)
    a.plot([0,1],[0,1],'k--',lw=1);a.set(xlabel='Falsch als alter Testbestand erkannt',ylabel='Alter Testbestand richtig erkannt',title='Ist die Herkunft erkennbar?',xlim=(0,1),ylim=(0,1));a.legend(loc='lower right')
    a.text(.03,.05,'Frühere einfache Bildmerkmale: AUROC 0,849\nAnderes Verfahren; kein gepaarter Überlegenheitstest.',transform=a.transAxes,fontsize=9)
    conditions=[('official','all','Offiziell: alle Normalbilder'),('mixed','all','Gemischt: alle Normalbilder'),('mixed','train','Gemischt: früheres Train'),('mixed','test','Gemischt: früherer Test')]
    for i,(arm,origin,label) in enumerate(conditions):
        row=next(r for r in c['summaries'] if r['arm']==arm and r['origin']==origin)
        values=[r['rho'] for r in c['rows'] if r['origin']==origin and r['source_outer_fold']=='all' and r['run_id'].startswith('official')==(arm=='official') and r['rho'] is not None]
        b.scatter(values,[i]*len(values),s=12,color='#b4b4b4')
        bounds=row['group_bootstrap_of_mean']['percentile95']
        if bounds:b.plot(bounds,[i,i],color='#277da8',lw=3)
        if row['mean'] is not None:b.plot(row['mean'],i,'o',color='#d98921')
    b.axvline(0,color='black',ls='--',lw=1);b.set_yticks(range(4),[x[2] for x in conditions]);b.set(xlim=(-1,1),ylim=(3.5,-.5),xlabel='Spearman-Korrelation',title='Herkunftsscore und Pneumonie-Logit')
    fig.text(.5,.015,'Links: gepoolte OOF-Kurve und fünf Foldkurven. Rechts: Modellwerte, Mittel und bedingter Gruppen-Bootstrap; keine Kausalität.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'stage3_source_correlation.svg');fig.savefig('/tmp/stage3-source.png',dpi=140);plt.close(fig)
    h=json.loads((OUT/'h3_results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.5));splits=['official',*map(str,range(20261112,20261117))]
    for a,metric,title in zip(axes,['R80','specificity'],['Änderung Fehler bei 80 % Abdeckung','Änderung Spezifität']):
        for i,split in enumerate(splits):
            values=np.array([r['delta'][metric]*100 for r in h['rows'] if r['run_id'].startswith(split+'_')]);a.scatter(i+np.linspace(-.12,.12,5),values,color='#277da8',s=25)
            a.plot([i-.2,i+.2],[values.mean()]*2,color='#d98921',lw=3)
        a.axhline(0,ls='--',color='black',lw=1);a.set_xticks(range(6),['Offiziell','Split 12','Split 13','Split 14','Split 15','Split 16'],rotation=25);a.set(ylabel='Bilinear minus Lanczos (Prozentpunkte)',title=title)
    fig.text(.5,.015,'Punkte: fünf Trainingsseeds je Aufteilung. Strich: Mittel. Nur der Resize-Kernel ändert sich; explorativer Vergleich.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(OUT/'stage3_h3.svg');fig.savefig('/tmp/stage3-h3.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
